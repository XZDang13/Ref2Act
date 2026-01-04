import pickle
from dataclasses import dataclass, field
from importlib import resources as importlib_resources
from pathlib import Path

import numpy as np

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
import pinocchio as pin


# ----------------------------
# Config
# ----------------------------
def _default_assets_root() -> Path:
    try:
        assets_root = Path(importlib_resources.files("Ref2Act") / "assets")
        if assets_root.exists():
            return assets_root
    except Exception:
        pass
    return Path(__file__).resolve().parent / "assets"


@dataclass
class GMR2NPZConfig:
    pkl_file_path: str = ""
    out_npz_path: str = ""

    fps_new: int = 30  # desired sampling rate
    smooth_sigma: float = 1.0

    urdf_path: str = field(
        default_factory=lambda: str(
            _default_assets_root() / "G1" / "g1_23dof_rubber_hand.urdf"
        )
    )
    mesh_dir: str = field(
        default_factory=lambda: str(_default_assets_root() / "G1")
    )
    joint_names: list[str] = field(
        default_factory=lambda:[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
        ]
    )
    body_names: list[str] = field(
        default_factory=lambda:[
            "pelvis",
            "torso_link",
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_hip_yaw_link",
            "right_hip_yaw_link",
            "left_rubber_hand",
            "right_rubber_hand",
            "left_knee_link",
            "right_knee_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]
    )

    verbose: bool = False


# ----------------------------
# Converter
# ----------------------------
class GMR2NPZConverter:
    def __init__(self, cfg: GMR2NPZConfig):
        self.cfg = cfg

        self.joint_names = cfg.joint_names

        self.body_names = cfg.body_names or [
            "pelvis",
            "torso_link",
            "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "left_elbow_link",
            "right_elbow_link",
            "left_hip_yaw_link",
            "right_hip_yaw_link",
            "left_rubber_hand",
            "right_rubber_hand",
            "left_knee_link",
            "right_knee_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]

        self.robot: pin.RobotWrapper|None = None
        self.model = None
        self.data_pk = None
        self._frame_ids: np.ndarray|None = None

    # -------- Quaternion helpers (w,x,y,z) --------
    @staticmethod
    def quaternion_inverse(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        norm_sq = w * w + x * x + y * y + z * z
        if norm_sq < 1e-8:
            norm_sq = 1e-8
        return np.array([w, -x, -y, -z], dtype=q.dtype) / norm_sq

    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z], dtype=q1.dtype)

    @classmethod
    def compute_angular_velocity(cls, q_prev: np.ndarray, q_next: np.ndarray, dt: float, eps: float = 1e-8) -> np.ndarray:
        q_inv = cls.quaternion_inverse(q_prev)
        q_rel = cls.quaternion_multiply(q_inv, q_next)
        norm_q_rel = np.linalg.norm(q_rel)
        if norm_q_rel < eps:
            return np.zeros(3, dtype=np.float32)
        q_rel = q_rel / norm_q_rel

        w = np.clip(q_rel[0], -1.0, 1.0)
        angle = 2.0 * np.arccos(w)
        sin_half = np.sqrt(max(0.0, 1.0 - w * w))
        if sin_half < eps:
            return np.zeros(3, dtype=np.float32)
        axis = q_rel[1:] / sin_half
        return (angle / dt) * axis

    # -------- Pinocchio --------
    def build_pin_robot(self) -> pin.RobotWrapper:
        robot = pin.RobotWrapper.BuildFromURDF(
            self.cfg.urdf_path,
            self.cfg.mesh_dir,
            pin.JointModelFreeFlyer(),
        )
        return robot

    def _ensure_robot(self):
        if self.robot is None:
            self.robot = self.build_pin_robot()
            self.model = self.robot.model
            self.data_pk = self.robot.data

            # cache frame ids for body_names once
            frame_ids = []
            for name in self.body_names:
                fid = self.model.getFrameId(name)
                if fid == len(self.model.frames):
                    raise ValueError(f"Frame '{name}' not found in model.frames. Check URDF/frame names.")
                frame_ids.append(fid)
            self._frame_ids = np.array(frame_ids, dtype=np.int64)

    # -------- Data I/O --------
    def load_pkl(self, pkl_path: str|None = None) -> dict:
        pkl_path = pkl_path or self.cfg.pkl_file_path
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return data

    # -------- Core steps --------
    def resample(self, root_pos: np.ndarray, root_rot_xyzw: np.ndarray, dof_pos: np.ndarray, fps_orig: float):
        """
        root_pos: (N,3)
        root_rot_xyzw: (N,4) in (qx,qy,qz,qw)
        dof_pos: (N,D)
        """
        N_orig = dof_pos.shape[0]
        dt_orig = 1.0 / float(fps_orig)
        t_orig = np.linspace(0.0, (N_orig - 1) * dt_orig, N_orig)

        fps_new = float(self.cfg.fps_new)
        dt_new = 1.0 / fps_new

        # keep your original "insert one frame between frames" behavior
        N_new = 2 * N_orig - 1
        t_new = np.linspace(0.0, (N_orig - 1) * dt_orig, N_new)

        # pos linear
        root_pos_interp = interp1d(t_orig, root_pos, axis=0, kind="linear")(t_new)

        # quat slerp (scipy uses xyzw)
        rotations_orig = R.from_quat(root_rot_xyzw)
        slerp = Slerp(t_orig, rotations_orig)
        rotations_new = slerp(t_new)
        root_quat_interp = rotations_new.as_quat()  # xyzw

        # dof linear
        dof_pos_interp = interp1d(t_orig, dof_pos, axis=0, kind="linear")(t_new)

        root_data = np.hstack([root_pos_interp, root_quat_interp])  # (N_new, 7)
        return root_data, dof_pos_interp, int(N_new), float(self.cfg.fps_new), float(dt_new)

    def compute_dof_vel(self, dof_pos: np.ndarray, dt: float) -> np.ndarray:
        vel = np.zeros_like(dof_pos)
        vel[1:-1] = (dof_pos[2:] - dof_pos[:-2]) / (2.0 * dt)
        vel[0] = (dof_pos[1] - dof_pos[0]) / dt
        vel[-1] = (dof_pos[-1] - dof_pos[-2]) / dt
        vel = gaussian_filter1d(vel, sigma=self.cfg.smooth_sigma, axis=0)
        return vel.astype(np.float32)

    def forward_kinematics(self, root_data: np.ndarray, dof_pos: np.ndarray):
        """
        root_data: (N,7) where quat is (qx,qy,qz,qw)
        dof_pos: (N,D)
        """
        self._ensure_robot()
        model, data_pk = self.model, self.data_pk
        frame_ids = self._frame_ids
        assert frame_ids is not None

        N, D = dof_pos.shape
        B = len(self.body_names)

        nq = model.nq
        if (7 + D) != nq and self.cfg.verbose:
            print(f"[Warn] (7 + D)={7 + D}, but pinocchio nq={nq}. Check joint mapping/order.")

        body_positions = np.zeros((N, B, 3), dtype=np.float32)
        body_rotations_wxyz = np.zeros((N, B, 4), dtype=np.float32)

        q_pin = pin.neutral(model)

        for i in range(N):
            q_pin[0:3] = root_data[i, 0:3]
            q_pin[3:7] = root_data[i, 3:7]          # xyzw (Pinocchio freeflyer expects xyzw in q)
            q_pin[7:7 + D] = dof_pos[i, :]

            pin.forwardKinematics(model, data_pk, q_pin)
            pin.updateFramePlacements(model, data_pk)

            # collect frames
            for j, fid in enumerate(frame_ids):
                tf = data_pk.oMf[int(fid)]
                body_positions[i, j, :] = tf.translation

                quat_xyzw = pin.Quaternion(tf.rotation)  # pin quaternion object has x,y,z,w
                body_rotations_wxyz[i, j, :] = np.array(
                    [quat_xyzw.w, quat_xyzw.x, quat_xyzw.y, quat_xyzw.z],
                    dtype=np.float32,
                )

        return body_positions, body_rotations_wxyz

    def compute_body_vel(self, body_positions: np.ndarray, body_quaternions_wxyz: np.ndarray, dt: float):
        N, B, _ = body_positions.shape

        # linear vel
        lin = np.zeros_like(body_positions)
        lin[1:-1] = (body_positions[2:] - body_positions[:-2]) / (2.0 * dt)
        lin[0] = (body_positions[1] - body_positions[0]) / dt
        lin[-1] = (body_positions[-1] - body_positions[-2]) / dt
        lin = gaussian_filter1d(lin, sigma=self.cfg.smooth_sigma, axis=0).astype(np.float32)

        # angular vel from adjacent quats (wxyz)
        ang = np.zeros((N, B, 3), dtype=np.float32)
        for j in range(B):
            quats = body_quaternions_wxyz[:, j, :]
            avel = np.zeros((N, 3), dtype=np.float32)
            if N > 1:
                avel[0] = self.compute_angular_velocity(quats[0], quats[1], dt)
                avel[-1] = self.compute_angular_velocity(quats[-2], quats[-1], dt)
            for k in range(1, N - 1):
                av1 = self.compute_angular_velocity(quats[k - 1], quats[k], dt)
                av2 = self.compute_angular_velocity(quats[k], quats[k + 1], dt)
                avel[k] = 0.5 * (av1 + av2)
            ang[:, j, :] = gaussian_filter1d(avel, sigma=self.cfg.smooth_sigma, axis=0)

        return lin, ang

    # -------- Orchestration --------
    def convert(self, pkl_path: str|None = None, out_npz_path: str|None = None):
        data = self.load_pkl(pkl_path)

        dof_pos_orig = data["dof_pos"]
        root_pos_orig = data["root_pos"]
        root_rot_orig = data["root_rot"]  # expected (N,4) xyzw
        fps_orig = float(data["fps"])

        if self.cfg.verbose:
            print(f"Loading {pkl_path or self.cfg.pkl_file_path}: {dof_pos_orig.shape[0]} frames @ {fps_orig} fps")

        root_data, dof_pos, N, fps, dt = self.resample(root_pos_orig, root_rot_orig, dof_pos_orig, fps_orig)

        dof_vel = self.compute_dof_vel(dof_pos, dt)
        body_pos, body_quaternions_wxyz = self.forward_kinematics(root_data, dof_pos)
        body_lin, body_ang = self.compute_body_vel(body_pos, body_quaternions_wxyz, dt)

        dof_names = np.array(self.joint_names, dtype=np.str_)
        body_names = np.array(self.body_names, dtype=np.str_)

        out = {
            "fps": int(fps),
            "joint_names": dof_names,
            "body_names": body_names,
            "joint_positions": dof_pos.astype(np.float32),
            "joint_velocities": dof_vel.astype(np.float32),
            "body_positions": body_pos.astype(np.float32),
            "body_quaternions": body_quaternions_wxyz.astype(np.float32),  # (w,x,y,z)
            "body_linear_velocities": body_lin.astype(np.float32),
            "body_angular_velocities": body_ang.astype(np.float32),
        }

        out_path = out_npz_path or self.cfg.out_npz_path
        np.savez(out_path, **out)

        if self.cfg.verbose:
            print(f"Saved NPZ -> {out_path}")
            for k, v in out.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: {v.shape}")
                else:
                    print(f"  {k}: {v}")
