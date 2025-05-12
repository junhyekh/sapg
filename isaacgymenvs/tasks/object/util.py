import trimesh
import numpy as np
import trimesh.poses
import pickle
from pathlib import Path
from typing import Tuple, Union

from scipy.spatial.transform import Rotation as R

def sample_stable_poses(hull: trimesh.Trimesh,
                        # xy_bound:
                        height: float = 0.4,
                        num_samples: int = 16,
                        multiplier: int = 8):
    xfms, probs = trimesh.poses.compute_stable_poses(
        hull, n_samples=(num_samples * multiplier))
    xfms = xfms[np.argsort(probs)[-num_samples:]]
    rot_mat = xfms[:, :3, :3]
    # x[..., :2] =
    x = xfms[:, :3, 3]
    x[..., 2] += height
    q = R.from_matrix(rot_mat).as_quat()
    pose = np.concatenate([x, q], axis=-1)
    return pose

def scene_to_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    if len(scene.graph.nodes_geometry) == 1:
        # Take cheaper option if possible.
        node_name = scene.graph.nodes_geometry[0]
        (transform, geometry_name) = scene.graph[node_name]
        mesh = scene.geometry[geometry_name]
        if not (transform == np.eye(4)).all():
            mesh.apply_transform(transform)
    else:
        # Default = dump
        mesh = scene.dump(concatenate=True)
    return mesh


def load_mesh(mesh_file: str,
              as_mesh: bool = False,
              **kwds) -> Union[trimesh.Trimesh,
                               Tuple[np.ndarray, np.ndarray]]:
    # [1] Load Mesh.
    mesh = trimesh.load(mesh_file,
                        # force='mesh',
                        skip_texture=True,
                        skip_materials=True,
                        **kwds)

    # [2] Ensure single geometry.
    if isinstance(mesh, trimesh.Scene):
        mesh = scene_to_mesh(mesh)
    if as_mesh:
        return mesh

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return (verts, faces)

def load_pkl(s: str):
    if not Path(s).is_file():
        return None
    with open(s, 'rb') as fp:
        return pickle.load(fp)


def load_npy(s: str):
    if not Path(s).is_file():
        return None
    try:
        return np.load(s)
    except FileNotFoundError:
        return None


def load_glb(s: str):
    if not Path(s).is_file():
        return None
    try:
        return load_mesh(s, as_mesh=True)
    except ValueError:
        return None