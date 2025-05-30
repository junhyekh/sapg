#!/usr/bin/env python3

from typing import Optional
from dataclasses import dataclass
from omegaconf import OmegaConf
from pkm.util.config import with_zen_cli

import trimesh
import coacd

import numpy as np
import os
import glob


from pathlib import Path

@dataclass
class Config:
    # input file
    raw: Optional[str] = None
    # output file
    out: Optional[str] = None
    max_concavity: float = 0.05
    max_chull_count: int = 16
    resolution: int = 512
    mcts_max_depth: int = 4
    mcts_iterations: int = 256
    mcts_nodes: int = 32


# @with_zen_cli
def main(cfg: Config = Config()):
    if cfg.raw is None:
        raise ValueError(F'wrong input file: raw={cfg.raw}')
    if cfg.out is None:
        raise ValueError(F'wrong output file: out={cfg.out}')
    mesh = trimesh.load(
        cfg.raw,
        file_type='obj',
        force='mesh'
    )
    imesh = coacd.Mesh()
    imesh.vertices = mesh.vertices
    imesh.indices = mesh.faces
    coacd.run_coacd
    parts = coacd.run_coacd(
        imesh,
        threshold=cfg.max_concavity,  # max concavity
        max_convex_hull=cfg.max_chull_count,
        resolution=cfg.resolution,
        mcts_max_depth=cfg.mcts_max_depth,
        mcts_iterations=cfg.mcts_iterations,
        mcts_nodes=cfg.mcts_nodes
    )
    try:
        # (old API)
        mesh_parts = [
            trimesh.Trimesh(np.array(p[0]),
                            np.array(p[1]).reshape((-1, 3)))
            for p in parts
        ]
    except TypeError:
        # (new API)
        mesh_parts = [
            trimesh.Trimesh(np.array(p.vertices),
                            np.array(p.indices).reshape((-1, 3)))
            for p in parts
        ]
    # build trimesh scene & output
    scene = trimesh.Scene()
    for p in mesh_parts:
        scene.add_geometry(p)
    scene.export(cfg.out)

def outer_main():
    # Base directory
    base_dir = '/input/isaac_data'
    gso_dir = os.path.join(base_dir, 'GSO')

    # Find all vis.obj files in any subdirectory except GSO
    vis_obj_paths = []
    for root, dirs, files in os.walk(base_dir):
        # Skip GSO directory
        if 'GSO' in dirs:
            dirs.remove('GSO')
        
        if 'vis.obj' in files:
            vis_obj_paths.append(os.path.join(root, 'vis.obj'))

    if not vis_obj_paths:
        raise FileNotFoundError(f'No vis.obj files found in {base_dir} subdirectories')

    # all object in GSO
    gso_obj_paths = []
    for root, dirs, files in os.walk(gso_dir):
       gso_obj_paths.append(Path(root).stem)

    # Process each vis.obj file
    # print(vis_obj_paths)
    for vis_obj_path in vis_obj_paths:
        if Path(vis_obj_path).parent.stem not in gso_obj_paths:
            print(f'Skipping {vis_obj_path} (not in GSO)')
            continue
    #     # Create col.obj path in the same directory as vis.obj
        col_obj_path = os.path.join(os.path.dirname(vis_obj_path), 'col.obj')
        print(f'Processing {vis_obj_path} -> {col_obj_path}')
        
        # Run coacd
        cfg = Config(raw=str(vis_obj_path), out=str(col_obj_path))
        main(cfg)


if __name__ == '__main__':
    outer_main()
