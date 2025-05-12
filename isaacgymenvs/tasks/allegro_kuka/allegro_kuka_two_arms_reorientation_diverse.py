import os
import math
from typing import List
import numpy as np
import torch
from isaacgym import gymapi
from torch import Tensor
import tempfile
from tqdm.auto import tqdm

from isaacgymenvs.utils.torch_jit_utils import to_torch, torch_rand_float
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import DofParameters, populate_dof_properties

from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_two_arms import AllegroKukaTwoArmsBase
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import tolerance_curriculum, tolerance_successes_objective
from isaacgymenvs.tasks.object.mesh_object_set import MeshObjectSet
from isaacgymenvs.utils.torch_jit_utils import *

class AllegroKukaTwoArmsReorientationDiverse(AllegroKukaTwoArmsBase):
    """
    This task is similar to AllegroKukaTwoArmsReorientation, 
    but it uses a diverse set of objects and obstacles.
    """

    """
    TODO:
    - [ ] Add a diverse set of objects. (JH)
        - [ ] Add a diverse set of objects.
        - [ ] Update keypoint offsets.
        - [ ] Update object asset files.
        - [ ] Connect pcd / Unicorn embedding as an input.
    - [ ] Add a diverse set of obstacles. (DW)
        - [ ] Add a diverse set to the scene
        - [ ] Connect scene information (positions, sizes) to the input.
    Is it necessary to have a diverse set of initial conditions?
    - [ ] Add a diverse set of initial conditions.
    - [ ] Add a diverse set of goal conditions.
    """


    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.goal_object_indices = []
        self.goal_assets = []

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

    def _object_keypoint_offsets(self):
        #TODO keypoint offsets should be different for different objects
        return [
            [1, 1, 1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")

        object_asset_root = asset_root
        tmp_assets_dir = tempfile.TemporaryDirectory()
        self.object_asset_files, self.object_asset_scales = self._main_object_assets_and_scales(
            object_asset_root, tmp_assets_dir.name,
        )

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading asset {self.hand_arm_asset_file} from {asset_root}")
        allegro_kuka_asset = self.gym.load_asset(self.sim, asset_root, self.hand_arm_asset_file, asset_options)
        print(f"Loaded asset {allegro_kuka_asset}")

        num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(allegro_kuka_asset)
        num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(allegro_kuka_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(allegro_kuka_asset)
        assert (
            self.num_hand_arm_dofs == num_hand_arm_dofs
        ), f"Number of DOFs in asset {allegro_kuka_asset} is {num_hand_arm_dofs}, but {self.num_hand_arm_dofs} was expected"

        max_agg_bodies = all_arms_bodies = num_hand_arm_bodies * self.num_arms
        max_agg_shapes = all_arms_shapes = num_hand_arm_shapes * self.num_arms

        allegro_rigid_body_names = [
            self.gym.get_asset_rigid_body_name(allegro_kuka_asset, i) for i in range(num_hand_arm_bodies)
        ]
        print(f"Allegro num rigid bodies: {num_hand_arm_bodies}")
        print(f"Allegro rigid bodies: {allegro_rigid_body_names}")

        # allegro_actuated_dof_names = [self.gym.get_asset_actuator_joint_name(allegro_asset, i) for i in range(self.num_allegro_dofs)]
        # self.allegro_actuated_dof_indices = [self.gym.find_asset_dof_index(allegro_asset, name) for name in allegro_actuated_dof_names]

        hand_arm_dof_props = self.gym.get_asset_dof_properties(allegro_kuka_asset)

        arm_hand_dof_lower_limits = []
        arm_hand_dof_upper_limits = []

        for arm_idx in range(self.num_arms):
            for i in range(self.num_hand_arm_dofs):
                arm_hand_dof_lower_limits.append(hand_arm_dof_props["lower"][i])
                arm_hand_dof_upper_limits.append(hand_arm_dof_props["upper"][i])

        # self.allegro_actuated_dof_indices = to_torch(self.allegro_actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(arm_hand_dof_lower_limits, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(arm_hand_dof_upper_limits, device=self.device)

        arm_poses = [gymapi.Transform() for _ in range(self.num_arms)]
        arm_x_ofs, arm_y_ofs = self.arm_x_ofs, self.arm_y_ofs
        for arm_idx, arm_pose in enumerate(arm_poses):
            x_ofs = arm_x_ofs * (-1 if arm_idx == 0 else 1)
            arm_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(x_ofs, arm_y_ofs, 0)

            # arm_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            if arm_idx == 0:
                # rotate 1st arm 90 degrees to the left
                arm_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi / 2)
            else:
                # rotate 2nd arm 90 degrees to the right
                arm_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), -math.pi / 2)

        object_assets, object_rb_count, object_shapes_count = self._load_main_object_asset()

        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = False
        table_asset_options.fix_base_link = True
        
        # TODO: Load table asset with obstacles
        # First define table asset
        table_tmp_assets_dir = tempfile.TemporaryDirectory()
        table_asset = self._define_table_asset(asset_root, 
                                                table_tmp_assets_dir.name, 
                                                num_table_assets=1) # FIXME: num_table_assets should be a variable
        # Load table assets
        table_assets, table_rb_count, table_shapes_count = self._load_table_assets(asset_root,
                                                                                    table_asset,
                                                                                    table_asset_options)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = 0.0
        # table_pose_dy, table_pose_dz = -0.8, 0.38
        table_pose_dy, table_pose_dz = 0.0, 0.38
        table_pose.p.y = arm_y_ofs + table_pose_dy
        table_pose.p.z = table_pose_dz

        # table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        # # table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)
        # max_agg_bodies += table_rb_count
        # max_agg_shapes += table_shapes_count

        # load auxiliary objects
        self._load_additional_assets(object_asset_root, arm_y_ofs)
        # max_agg_bodies += additional_rb
        # max_agg_shapes += additional_shapes

        # set up object and goal positions
        self.object_start_pose = self._object_start_pose(arm_y_ofs, table_pose_dy, table_pose_dz)

        self.envs = []

        object_init_state = []
        object_scales = []
        object_keypoint_offsets = []
        
        self.rigid_body_name_to_idx = {}

        allegro_palm_handle = self.gym.find_asset_rigid_body_index(allegro_kuka_asset, "iiwa7_link_7")
        fingertip_handles = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in self.allegro_fingertips
        ]

        self.allegro_palm_handles = []
        self.allegro_fingertip_handles = []
        for arm_idx in range(self.num_arms):
            self.allegro_palm_handles.append(allegro_palm_handle + arm_idx * num_hand_arm_bodies)
            self.allegro_fingertip_handles.extend([h + arm_idx * num_hand_arm_bodies for h in fingertip_handles])

        # does this rely on the fact that objects are added right after the arms in terms of create_actor()?
        # FIXME: this is a hack, we are not sure this is correct
        self.object_rb_handles = list(range(all_arms_bodies, all_arms_bodies + object_rb_count[0]))

        self.arm_indices = torch.empty([self.num_envs, self.num_arms], dtype=torch.long, device=self.device)
        self.object_indices = torch.empty(self.num_envs, dtype=torch.long, device=self.device)

        assert self.num_envs >= 1
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            agg_bodies = max_agg_bodies
            agg_shapes = max_agg_shapes
            
            # add _object
            object_asset_idx = i % len(object_assets)
            object_asset = object_assets[object_asset_idx]
            agg_bodies += object_rb_count[object_asset_idx]
            agg_shapes += object_shapes_count[object_asset_idx]

            # add table
            table_asset_idx = i % len(table_assets)
            table_asset = table_assets[table_asset_idx]
            agg_bodies += table_rb_count[table_asset_idx]
            agg_shapes += table_shapes_count[table_asset_idx]

            # add auxiliary objects
            agg_bodies += object_rb_count[object_asset_idx]
            agg_shapes += object_shapes_count[object_asset_idx]
            
            self.gym.begin_aggregate(env_ptr, agg_bodies, agg_shapes, True)

            # add arms
            for arm_idx in range(self.num_arms):
                arm = self.gym.create_actor(env_ptr, allegro_kuka_asset, arm_poses[arm_idx], f"arm{arm_idx}", i, -1, 0)
                for name in self.gym.get_actor_rigid_body_names(env_ptr, arm):
                    self.rigid_body_name_to_idx["allegro" + ("/" if arm_idx == 0 else f"{arm_idx}/") + name] = self.gym.find_actor_rigid_body_index(env_ptr, arm, name, gymapi.DOMAIN_ENV)
                populate_dof_properties(hand_arm_dof_props, self.dof_params, self.num_arm_dofs, self.num_hand_dofs)

                self.gym.set_actor_dof_properties(env_ptr, arm, hand_arm_dof_props)
                allegro_hand_idx = self.gym.get_actor_index(env_ptr, arm, gymapi.DOMAIN_SIM)

                self.arm_indices[i, arm_idx] = allegro_hand_idx

            # add object
            obj_pose = self.object_start_pose
            object_handle = self.gym.create_actor(env_ptr, object_asset, obj_pose, "object", i, 0, 0)
            pos, rot = obj_pose.p, obj_pose.r
            object_init_state.append([pos.x, pos.y, pos.z, rot.x, rot.y, rot.z, rot.w, 0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            
            for name in self.gym.get_actor_rigid_body_names(env_ptr, object_handle):
                self.rigid_body_name_to_idx["object/" + name] = self.gym.find_actor_rigid_body_index(env_ptr, object_handle, name, gymapi.DOMAIN_ENV)
            self.object_indices[i] = object_idx

            object_scale = self.object_asset_scales[object_asset_idx]
            object_scales.append(object_scale)
            # object_offsets = []
            # for keypoint in self.keypoints_offsets:
            #     keypoint = copy(keypoint)
            #     for coord_idx in range(3):
            #         keypoint[coord_idx] *= object_scale[coord_idx] * self.object_base_size * self.keypoint_scale / 2
            #     object_offsets.append(keypoint)

            object_keypoint_offsets.append(self._bbox[object_asset_idx])

            # table object
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table_object", i, 0, 0)
            _table_object_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, table_handle):
                self.rigid_body_name_to_idx["table/" + name] = self.gym.find_actor_rigid_body_index(env_ptr, table_handle, name, gymapi.DOMAIN_ENV)

            # task-specific objects (i.e. goal object for reorientation task)
            self._create_additional_objects(env_ptr, env_idx=i, object_asset_idx=object_asset_idx)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        # noinspection PyUnboundLocalVariable
        object_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[0], object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )   
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.allegro_fingertip_handles = to_torch(self.allegro_fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.object_scales = to_torch(object_scales, dtype=torch.float, device=self.device)
        self.object_keypoint_offsets = to_torch(object_keypoint_offsets, dtype=torch.float, device=self.device)

        self._after_envs_created()

        try:
            # by this point we don't need the temporary folder for procedurally generated assets
            tmp_assets_dir.cleanup()
        except Exception:
            pass

    """
    Asset functions
    """

    def _main_object_assets_and_scales(self, 
                                        object_asset_root,
                                        tmp_assets_dir):
        """
        Load main object assets and scales.
        """

        self._object_set = MeshObjectSet(MeshObjectSet.Config(
            **(self.cfg["env"]["mesh_object_set"])
        ))

        obj_keys = list(self._object_set.keys()) * self.cfg["env"]["num_object_repeats"]
        obj_keys = np.random.permutation(obj_keys)
        
        scale = np.random.uniform(self.cfg["env"]["object_scale_range"][0],
                                  self.cfg["env"]["object_scale_range"][1],
                                  size=len(obj_keys))
        object_asset_files = [self._object_set.urdf(key) for key in obj_keys]
        radius = [self._object_set.radius(key) for key in obj_keys]
        rel_scale = [s/r for s, r in zip(scale, radius)]
        bbox = [self._object_set.bbox(key) for key in obj_keys]
        bbox = np.asarray(bbox, dtype=np.float32)
        bbox = bbox * np.array(rel_scale, dtype=np.float32)[:, None, None]
        object_asset_scales = np.max(bbox, axis=-2) - np.min(bbox, axis=-2)
        cloud = np.array([self._object_set.cloud(key) for key in obj_keys])
        self._cloud = cloud[..., :3] * np.array(rel_scale, dtype=np.float32)[:, None, None]
        self._bbox = bbox[..., np.array([0,1,-2,-1]), :]
        print("-"*30)
        print(f"bbox: {self._bbox.shape}")
        print(f"bbox_pre: {bbox.shape}")
        print("-"*30)
        return object_asset_files, object_asset_scales


    def _load_main_object_asset(self):
        """
        Load main object asset.
        """
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.fix_base_link = False
        object_asset_options.override_com = True
        object_asset_options.override_inertia = True
        object_asset_options.density = 200.0
        object_asset_options.thickness = 0.001
        object_asset_options.convex_decomposition_from_submeshes = True

        object_assets = []
        object_rb_count = []
        object_shapes_count = []
        for object_asset_file in tqdm(self.object_asset_files,
                                      desc='create_object_assets'):
            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)
            object_asset_ = self.gym.load_asset(self.sim, object_asset_dir, object_asset_fname, object_asset_options)
            object_assets.append(object_asset_)
            object_rb_count.append(self.gym.get_asset_rigid_body_count(object_asset_))
            object_shapes_count.append(self.gym.get_asset_rigid_shape_count(object_asset_))
        return object_assets, object_rb_count, object_shapes_count
        
        
    def _define_table_asset(self, table_asset_root, tmp_assets_dir,
                           num_table_assets):
        """
        Define table asset.
        Define obstacle in each table asset.
        Convert it to urdf.
        return table assets, obstacle_informations
        """
        # table_path = os.path.join(table_asset_root, self.asset_files_dict["table"]).resolve()
        table_assets = [self.asset_files_dict["table"]] * num_table_assets

        return table_assets

    def _load_table_assets(self,
                            asset_root,
                            table_assets,
                            table_asset_options):
        """
        Load table assets.
        """
        table_assets = [self.gym.load_asset(self.sim, asset_root,
                                            table_asset, table_asset_options)
                                            for table_asset in table_assets]
        table_rb_count = [self.gym.get_asset_rigid_body_count(table_asset) 
                          for table_asset in table_assets]
        table_shapes_count = [self.gym.get_asset_rigid_shape_count(table_asset) 
                                for table_asset in table_assets]
        return table_assets, table_rb_count, table_shapes_count



    def _load_additional_assets(self, object_asset_root, arm_pose):
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        self.goal_assets = []
        for object_asset_file in self.object_asset_files:
            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)

            goal_asset_ = self.gym.load_asset(self.sim, 
                                                object_asset_dir, 
                                                object_asset_fname, 
                                                object_asset_options)
            self.goal_assets.append(goal_asset_)

    def _create_additional_objects(self, env_ptr, env_idx, object_asset_idx):
        self.goal_displacement = gymapi.Vec3(-0.35, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device
        )
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = self.object_start_pose.p + self.goal_displacement
        goal_start_pose.p.z -= 0.04

        goal_asset = self.goal_assets[object_asset_idx]
        goal_handle = self.gym.create_actor(
            env_ptr, goal_asset, goal_start_pose, "goal_object", env_idx + self.num_envs, 0, 0
        )
        goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
        self.goal_object_indices.append(goal_object_idx)
        for name in self.gym.get_actor_rigid_body_names(env_ptr, goal_handle):
            self.rigid_body_name_to_idx["goal/" + name] = self.gym.find_actor_rigid_body_index(
                env_ptr, goal_handle, name, gymapi.DOMAIN_ENV
            )

        # if self.object_type != "block":
        #     self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

    def _after_envs_created(self):
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def _reset_target(self, env_ids: Tensor, tensor_reset=True) -> None:
        if tensor_reset:
            # sample random target location in some volume
            target_volume_origin = self.target_volume_origin
            target_volume_extent = self.target_volume_extent

            target_volume_min_coord = target_volume_origin + target_volume_extent[:, 0]
            target_volume_max_coord = target_volume_origin + target_volume_extent[:, 1]
            target_volume_size = target_volume_max_coord - target_volume_min_coord

            rand_pos_floats = torch_rand_float(0.0, 1.0, (len(env_ids), 3), device=self.device)
            target_coords = target_volume_min_coord + rand_pos_floats * target_volume_size

            # let the target be close to 1st or 2nd arm, randomly
            left_right_random = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device)
            x_ofs = 0.75
            x_pos = torch.where(
                left_right_random > 0,
                x_ofs * torch.ones_like(left_right_random),
                -x_ofs * torch.ones_like(left_right_random),
            )

            target_coords[:, 0] += x_pos.squeeze(dim=1)

            self.goal_states[env_ids, 0:3] = target_coords
            self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]

            # new_rot = randomize_rotation(
            #     rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
            # )

            # new implementation by Ankur:
            new_rot = self.get_random_quat(env_ids)
            self.goal_states[env_ids, 3:7] = new_rot

            self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
                self.root_state_tensor[self.goal_object_indices[env_ids], 7:13]
            )

        object_indices_to_reset = [self.goal_object_indices[env_ids]]
        self.deferred_set_actor_root_state_tensor_indexed(object_indices_to_reset)

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        return [self.goal_object_indices[env_ids]]

    def _extra_curriculum(self):
        self.success_tolerance, self.last_curriculum_update = tolerance_curriculum(
            self.last_curriculum_update,
            self.frame_since_restart,
            self.tolerance_curriculum_interval,
            self.prev_episode_successes,
            self.success_tolerance,
            self.initial_tolerance,
            self.target_tolerance,
            self.tolerance_curriculum_increment,
        )

    def _true_objective(self) -> Tensor:
        true_objective = tolerance_successes_objective(
            self.success_tolerance, self.initial_tolerance, self.target_tolerance, self.successes
        )
        return true_objective

    def update_rigid_body_state_dict(self, state_ddict, env_idx=-1):
        """
        state_ddict: defaultdict of the form Dict[str, List[Tensor]]
        """
        if False:
            print(self.object_asset_scales[env_idx])
            isaacgym_to_blender_name = {
                'allegro/iiwa7_link_0': 'link_0',
                'allegro/iiwa7_link_1': 'link_1',
                'allegro/iiwa7_link_2': 'link_2',
                'allegro/iiwa7_link_3': 'link_3',
                'allegro/iiwa7_link_4': 'link_4',
                'allegro/iiwa7_link_5': 'link_5',
                'allegro/iiwa7_link_6': 'link_6',
                'allegro/iiwa7_link_7': 'link_7',
                'allegro1/iiwa7_link_0': 'link_0.002',
                'allegro1/iiwa7_link_1': 'link_1.002',
                'allegro1/iiwa7_link_2': 'link_2.002',
                'allegro1/iiwa7_link_3': 'link_3.002',
                'allegro1/iiwa7_link_4': 'link_4.002',
                'allegro1/iiwa7_link_5': 'link_5.002',
                'allegro1/iiwa7_link_6': 'link_6.002',
                'allegro1/iiwa7_link_7': 'link_7.002',
                #'allegro/iiwa7_link_ee': 'link_ee',
                'allegro/allegro_mount': 'allegro_mount',
                'allegro1/allegro_mount': 'allegro_mount.002',
                'allegro/palm_link': 'base_link',
                'allegro1/palm_link': 'base_link.002',
                
                'allegro/index_link_0': 'primary_base',
                'allegro/index_link_1': 'primary_proximal',
                'allegro/index_link_2': 'primary_medial',
                'allegro/index_link_3': 'touch_sensor_base',
                'allegro/middle_link_0': 'primary_base.001',
                'allegro/middle_link_1': 'primary_proximal.001',
                'allegro/middle_link_2': 'primary_medial.001',
                'allegro/middle_link_3': 'touch_sensor_base.001',
                'allegro/ring_link_0': 'primary_base.002',
                'allegro/ring_link_1': 'primary_proximal.002',
                'allegro/ring_link_2': 'primary_medial.002',
                'allegro/ring_link_3': 'touch_sensor_base.002',
                
                'allegro1/index_link_0': 'primary_base.006',
                'allegro1/index_link_1': 'primary_proximal.006',
                'allegro1/index_link_2': 'primary_medial.006',
                'allegro1/index_link_3': 'touch_sensor_base.006',
                'allegro1/middle_link_0': 'primary_base.007',
                'allegro1/middle_link_1': 'primary_proximal.007',
                'allegro1/middle_link_2': 'primary_medial.007',
                'allegro1/middle_link_3': 'touch_sensor_base.007',
                'allegro1/ring_link_0': 'primary_base.008',
                'allegro1/ring_link_1': 'primary_proximal.008',
                'allegro1/ring_link_2': 'primary_medial.008',
                'allegro1/ring_link_3': 'touch_sensor_base.008',
                
                'allegro/thumb_link_0': 'thumb_base',
                'allegro/thumb_link_1': 'thumb_proximal',
                'allegro/thumb_link_2': 'thumb_medial',
                'allegro/thumb_link_3': 'touch_sensor_thumb_base',
                'allegro1/thumb_link_0': 'thumb_base.002',
                'allegro1/thumb_link_1': 'thumb_proximal.002',
                'allegro1/thumb_link_2': 'thumb_medial.002',
                'allegro1/thumb_link_3': 'touch_sensor_thumb_base.002',
                # FIXME(JH): 
                # 'object/object': 'cube_multicolor',
                'table/box': 'cube', # 0.475 0.4 0.3
                # 'goal/object': 'cube_multicolor.001'
                
            }
            
            for key, value in isaacgym_to_blender_name.items():
                rigid_body_idx = self.rigid_body_name_to_idx[key]
                pose = self.rigid_body_states[env_idx, rigid_body_idx,0:7].cpu().numpy()
                import transforms3d
                
                pos = pose[0:3]
                quat = pose[3:7]
                rot = transforms3d.euler.quat2euler([quat[3], quat[0], quat[1], quat[2]])
                
                state_ddict[value].append((pos, rot))
