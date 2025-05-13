#!/usr/bin/env python3

import os
import glob
from tempfile import mkdtemp
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from pathlib import Path
from natsort import natsorted
import shutil

import numpy as np
import trimesh
from trimesh.visual import color as color_visual
from tqdm.auto import tqdm

from icecream import ic
try:
    # python >= 3.8
    from functools import cached_property, partial
except:
    # python < 3.8
    from functools import lru_cache, partial
    def cached_property(func):
        return property(lru_cache()(func))


from isaacgymenvs.tasks.object.util import load_glb, load_npy, load_pkl, sample_stable_poses
from isaacgymenvs.tasks.object.object_set import ObjectSet

import json

DATA_ROOT = os.getenv('PKM_DATA_ROOT', '/input')

URDF_TEMPLATE: str = '''<robot name="robot">
    <link name="base_link">
        <inertial>
            <mass value="{mass}"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}"
            iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{vis_mesh}" scale="1.0 1.0 1.0"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{col_mesh}" scale="1.0 1.0 1.0"/>
            </geometry>
        </collision>
    </link>
</robot>
'''


class MeshObjectSet(ObjectSet):
    @dataclass
    class Config:

        density: float = 300.0

        num_points: int = 512

        data_path: Optional[str] = '/input/isaac_data'
        meta_file: Optional[str] = None
        urdf_path: Optional[str] = None
        hull_path: Optional[str] = None
        cloud_path: Optional[str] = None
        normal_path: Optional[str] = None
        pose_path: Optional[str] = None
        code_path: Optional[str] = None

        def __post_init__(self):
            if self.data_path is not None:
                data_path = self.data_path
                if self.meta_file is None:
                    self.meta_file = F'{data_path}/metadata.json'
                if self.urdf_path is None:
                    self.urdf_path = F'{data_path}/urdf'
                if self.hull_path is None:
                    self.hull_path = F'{data_path}/hull'
                if self.cloud_path is None:
                    self.cloud_path = F'{data_path}/cloud'
                if self.normal_path is None:
                    self.normal_path = F'{data_path}/normal'
                if self.pose_path is None:
                    self.pose_path = F'{data_path}/pose'
                if self.code_path is None:
                    self.code_path = F'{data_path}/code'


    def __init__(self, cfg: Config):
        self.cfg = cfg

        if cfg.data_path is None:
            raise ValueError('cfg.filename should not be None!')
        ic(cfg)
        print('loading metadata')
        with open(cfg.meta_file, 'r') as fp:
            self.__metadata = json.load(fp)
        print('loaded metadata')
        keys = list(self.__metadata.keys())
        # for k in keys:
        #     if not Path(F'{self.cfg.pose_path}/{k}.npy').exists():
        #         self.__metadata.pop(k)
        ic(len(self.keys()))
        
    def keys(self):
        return self.__metadata.keys()
    
    @cached_property
    def poses(self):
        return {k: load_npy(F'{self.cfg.pose_path}/{k}.npy')
                for k in self.keys()}

    @cached_property
    def codes(self):
        # FIXME(ycho): temporary exception
        if Path(self.cfg.code_path).is_file():
            return load_pkl(self.cfg.code_path)
        return {k: load_npy(F'{self.cfg.code_path}/{k}.npy')
                for k in self.keys()}

    @cached_property
    def clouds(self):
        return {k: load_npy(F'{self.cfg.cloud_path}/{k}.npy')
                for k in self.keys()}

    @cached_property
    def normals(self):
        out = {k: load_npy(F'{self.cfg.normal_path}/{k}.npy')
               for k in self.keys()}
        return out

    @cached_property
    def hulls(self) -> Dict[str, trimesh.Trimesh]:
        return {k: load_glb(F'{self.cfg.hull_path}/{k}.glb')
                for k in self.keys()}

    def label(self, key: str) -> str:
        """ Category of this object """
        return key

    def urdf(self, key: str):
        cfg = self.cfg
        return F'{cfg.urdf_path}/{key}.urdf'

    def pose(self, key: str):
        try:
            return self.poses[key]
        except:
            return None

    def code(self, key: str):
        return None

    def cloud(self, key: str):
        return self.clouds[key]

    def normal(self, key: str):
        return self.normals[key]

    def bbox(self, key: str):
        return np.asarray(self.__metadata[key]['bbox'],
                          dtype=np.float32)

    def aabb(self, key: str):
        return np.asarray(self.__metadata[key]['aabb'],
                          dtype=np.float32)

    def obb(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        obb = self.__metadata[key]['obb']
        xfm = np.asarray(obb[0], dtype=np.float32)
        extent = np.asarray(obb[1], dtype=np.float32)
        return (xfm, extent)

    def hull(self, key: str) -> trimesh.Trimesh:
        return self.hulls[key]

    def radius(self, key: str) -> float:
        return self.__metadata[key]['radius']

    def volume(self, key: str) -> float:
        return self.__metadata[key]['volume']

    def num_verts(self, key: str) -> float:
        return self.__metadata[key]['num_verts']

    def num_faces(self, key: str) -> float:
        return self.__metadata[key]['num_faces']

    def num_hulls(self, key: str) -> float:
        # return self.__metadata[key]['num_chulls']
        return self.__metadata[key]['num_hulls']


def main():
    # _convert_from_previous_version()
    dataset = BenchObjectSet(
        BenchObjectSet.Config())
    for attr in dir(dataset):
        print(attr)
        if hasattr(dataset, attr):
            (getattr(dataset, attr))
    # print(len(dataset.codes))


if __name__ == '__main__':
    main()
