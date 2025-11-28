"""Gymnasium environment for pixel-based robot arm reaching in PyBullet."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class RewardWeights:
    distance: float = 1.0
    contact_bonus: float = 5.0
    time_penalty: float = 0.01


@dataclass
class CameraConfig:
    width: int = 84
    height: int = 84
    fov: float = 60.0
    near: float = 0.01
    far: float = 2.5
    distance: float = 1.0
    target: Tuple[float, float, float] = (0.5, 0.0, 0.1)
    yaw: float = 90.0
    pitch: float = -70.0
    roll: float = 0.0
    eye_in_hand: bool = False


class RobotArmEnv(gym.Env):
    """Pixel observation reinforcement learning environment for a PyBullet arm.

    The agent controls the end effector using delta position commands. The
    observation contains a stack of rendered grayscale frames and joint angles
    so that policies can infer motion and depth cues.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str = "rgb_array",
        use_gui: bool = False,
        camera: CameraConfig = CameraConfig(),
        reward_weights: RewardWeights = RewardWeights(),
        frame_stack: int = 4,
        frame_skip: int = 4,
        image_grayscale: bool = True,
        max_steps: int = 200,
        workspace_limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (0.3, 0.8),
            (-0.3, 0.3),
            (0.0, 0.5),
        ),
        time_step: float = 1.0 / 240.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.use_gui = use_gui
        self.camera = camera
        self.reward_weights = reward_weights
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.image_grayscale = image_grayscale
        self.max_steps = max_steps
        self.workspace_limits = workspace_limits
        self.time_step = time_step

        self._rng, _ = gym.utils.seeding.np_random(seed)
        self.physics_client = self._connect()
        self._setup_simulation()

        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self.num_joints = 7
        self._down_orientation = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
        self._rest_pose = [
            0.0,
            -math.pi / 4,
            0.0,
            -3 * math.pi / 4,
            0.0,
            math.pi / 2,
            0.0,
        ]
        self.action_space = gym.spaces.Box(
            low=np.array([-0.03, -0.03, -0.03], dtype=np.float32),
            high=np.array([0.03, 0.03, 0.03], dtype=np.float32),
            dtype=np.float32,
        )

        # The stacked observation always follows a channel-first layout.
        if self.image_grayscale:
            image_shape = (self.frame_stack, self.camera.height, self.camera.width)
        else:
            # RGB frames are concatenated along the channel axis: (T*3, H, W)
            image_shape = (
                self.frame_stack * 3,
                self.camera.height,
                self.camera.width,
            )
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=image_shape, dtype=np.uint8
                ),
                "joints": gym.spaces.Box(
                    low=-math.pi, high=math.pi, shape=(self.num_joints,), dtype=np.float32
                ),
            }
        )

        self.current_step = 0
        self.target_uid: Optional[int] = None
        self.robot_uid: Optional[int] = None
        self.table_uid: Optional[int] = None

        self.reset(seed=seed)

    def _connect(self) -> int:
        if self.use_gui:
            return p.connect(p.GUI)
        return p.connect(p.DIRECT)

    def _setup_simulation(self) -> None:
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def _load_scene(self) -> None:
        plane_uid = p.loadURDF("plane.urdf")
        self.table_uid = p.loadURDF("table/table.urdf", [0.5, 0.0, -0.65])
        self.robot_uid = p.loadURDF(
            "kuka_iiwa/model.urdf",
            [0.0, 0.0, 0.0],
            useFixedBase=True,
        )

        for j, angle in enumerate(self._rest_pose):
            p.resetJointState(self.robot_uid, j, angle)

        # Lift the arm above the table with a downward-facing wrist
        ik_pose = p.calculateInverseKinematics(
            self.robot_uid,
            6,
            targetPosition=[0.55, 0.0, 0.35],
            targetOrientation=self._down_orientation,
        )
        for idx, angle in enumerate(ik_pose):
            if idx >= self.num_joints:
                break
            p.resetJointState(self.robot_uid, idx, angle)

        obj_x = self._rng.uniform(*self.workspace_limits[0])
        obj_y = self._rng.uniform(*self.workspace_limits[1])
        obj_z = 0.02
        self.target_uid = p.loadURDF(
            "cube_small.urdf", [obj_x, obj_y, obj_z], globalScaling=1.0
        )

    def _get_end_effector_state(self) -> Tuple[np.ndarray, np.ndarray]:
        state = p.getLinkState(self.robot_uid, 6, computeForwardKinematics=True)
        pos = np.array(state[4])
        orn = np.array(state[5])
        return pos, orn

    def _apply_action(self, action: np.ndarray) -> None:
        pos, _ = self._get_end_effector_state()
        target_pos = pos + action

        min_xyz, max_xyz = zip(*self.workspace_limits)
        target_pos = np.clip(target_pos, min_xyz, max_xyz)

        joint_positions = p.calculateInverseKinematics(
            self.robot_uid,
            6,
            target_pos,
            targetOrientation=self._down_orientation,
            lowerLimits=[-2 * math.pi] * self.num_joints,
            upperLimits=[2 * math.pi] * self.num_joints,
            jointRanges=[4 * math.pi] * self.num_joints,
            restPoses=self._rest_pose,
            maxNumIterations=100,
            residualThreshold=1e-4,
        )

        for idx in range(self.num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_uid,
                jointIndex=idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_positions[idx],
                force=500,
            )

    def _render_camera(self) -> np.ndarray:
        if self.camera.eye_in_hand:
            ee_pos, ee_orn = self._get_end_effector_state()
            rot_matrix = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
            camera_direction = rot_matrix @ np.array([1, 0, 0])
            camera_up = rot_matrix @ np.array([0, 0, 1])
            cam_target = ee_pos + camera_direction * 0.1
            cam_view = p.computeViewMatrix(
                cameraEyePosition=ee_pos,
                cameraTargetPosition=cam_target,
                cameraUpVector=camera_up,
            )
            cam_proj = p.computeProjectionMatrixFOV(
                fov=self.camera.fov,
                aspect=float(self.camera.width) / float(self.camera.height),
                nearVal=self.camera.near,
                farVal=self.camera.far,
            )
        else:
            cam_view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.camera.target,
                distance=self.camera.distance,
                yaw=self.camera.yaw,
                pitch=self.camera.pitch,
                roll=self.camera.roll,
                upAxisIndex=2,
            )
            cam_proj = p.computeProjectionMatrixFOV(
                fov=self.camera.fov,
                aspect=float(self.camera.width) / float(self.camera.height),
                nearVal=self.camera.near,
                farVal=self.camera.far,
            )

        _, _, rgba, _, _ = p.getCameraImage(
            width=self.camera.width,
            height=self.camera.height,
            viewMatrix=cam_view,
            projectionMatrix=cam_proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        rgb = np.reshape(rgba, (self.camera.height, self.camera.width, 4))[:, :, :3]
        if self.image_grayscale:
            gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
            return gray.astype(np.uint8)
        return rgb.astype(np.uint8)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        if len(self.frame_buffer) == 0:
            frame = self._render_camera()
            for _ in range(self.frame_stack):
                self.frame_buffer.append(frame)

        if self.image_grayscale:
            stacked = np.stack(list(self.frame_buffer), axis=0)
        else:
            frames_chw = [np.transpose(f, (2, 0, 1)) for f in self.frame_buffer]
            stacked = np.concatenate(frames_chw, axis=0)
        joint_states = np.array(
            [p.getJointState(self.robot_uid, i)[0] for i in range(self.num_joints)],
            dtype=np.float32,
        )
        return {"image": stacked, "joints": joint_states}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng, _ = gym.utils.seeding.np_random(seed)

        self.current_step = 0
        p.resetSimulation(physicsClientId=self.physics_client)
        self._setup_simulation()
        self._load_scene()
        self.frame_buffer.clear()
        obs = self._get_observation()
        return obs, {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.current_step += 1

        for _ in range(self.frame_skip):
            self._apply_action(action)
            p.stepSimulation()

        frame = self._render_camera()
        self.frame_buffer.append(frame)
        obs = self._get_observation()

        ee_pos, _ = self._get_end_effector_state()
        obj_pos, _ = p.getBasePositionAndOrientation(self.target_uid)
        distance = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        contact = len(p.getContactPoints(self.robot_uid, self.target_uid)) > 0

        reward = (
            -self.reward_weights.distance * distance
            + (self.reward_weights.contact_bonus if contact else 0.0)
            - self.reward_weights.time_penalty
        )

        terminated = contact
        truncated = self.current_step >= self.max_steps
        info = {"distance": distance, "contact": contact}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            raise NotImplementedError
        return self._render_camera()

    def close(self):
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
