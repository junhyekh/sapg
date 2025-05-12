import trimesh
import numpy as np
import trimesh.poses

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


