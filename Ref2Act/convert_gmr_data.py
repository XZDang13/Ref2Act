import argparse
import pickle
from pathlib import Path

import numpy as np

import torch
from isaaclab.app import AppLauncher

from .motion_segments import (
    DEFAULT_AIRBORNE_HEIGHT_MARGIN,
    build_contact_segments,
    infer_ground_contact_from_foot_heights,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a GMR pickle motion file into the Ref2Act .npz format."
    )
    parser.add_argument("--input_file", "-f", type=str, required=True, help="Path to a GMR .pkl file.")
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output .npz file. Defaults to the input path with an .npz suffix.",
    )
    parser.add_argument("--height_offset", type=float, default=0.0, help="Offset to root z position.")
    parser.add_argument(
        "--segment-bin-size",
        type=float,
        default=0.3,
        help="Base time-bin size in seconds used to build segment-aware failure bins.",
    )
    parser.add_argument(
        "--airborne-height-threshold",
        type=float,
        default=DEFAULT_AIRBORNE_HEIGHT_MARGIN,
        help="Height above each foot's baseline required for both feet to be treated as airborne.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Remap NumPy 2.x internal module path to NumPy 1.x path
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

class GMRMotionData:
    def __init__(self, file: str, device: torch.device, joint_order: list[str], height_offset: float = 0.0):
        with open(file, "rb") as f:
            motion_data = NumpyCompatUnpickler(f).load()

        self.device = device
        self.joint_order = joint_order
        print(motion_data["fps"])
        self.fps = round(motion_data["fps"])
        self.offset = torch.zeros(3).to(self.device)
        self.offset[-1] += height_offset
        self.root_pos = torch.from_numpy(motion_data["root_pos"]).float().to(self.device) + self.offset
        self.root_rot = torch.from_numpy(motion_data["root_rot"]).float().to(self.device)
        self.root_rot = self.root_rot[:, [3, 0, 1, 2]]
        self.joint_pos = torch.from_numpy(motion_data["dof_pos"]).float().to(self.device)
        self.num_frames = self.joint_pos.size(0)
        self.render_interval = 1
        self.physic_dt = 1 / (self.render_interval * self.fps)
        self.current_step = 0

        self.root_lin_vel = torch.gradient(self.root_pos, spacing=self.physic_dt, dim=0)[0]
        self.root_ang_vel = self._so3_derivative(self.root_rot, self.physic_dt)
        self.joint_vel = torch.gradient(self.joint_pos, spacing=self.physic_dt, dim=0)[0]


    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat

        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega
    
    def get_init_state(self):
        motion = (
            self.root_pos[0 : 0 + 1],
            self.root_rot[0 : 0 + 1],
            self.root_lin_vel[0 : 0 + 1],
            self.root_ang_vel[0 : 0 + 1],
            self.joint_pos[0 : 0 + 1],
            self.joint_vel[0 : 0 + 1],
        )

        return motion
    
    def set_root_height(self, height_offset:float):
        offset = torch.zeros(3).to(self.device)
        offset[-1] += height_offset

        self.root_pos += offset
    
    def get_next_state(self):
        motion = (
            self.root_pos[self.current_step : self.current_step + 1],
            self.root_rot[self.current_step : self.current_step + 1],
            self.root_lin_vel[self.current_step : self.current_step + 1],
            self.root_ang_vel[self.current_step : self.current_step + 1],
            self.joint_pos[self.current_step : self.current_step + 1],
            self.joint_vel[self.current_step : self.current_step + 1],
        )
        self.current_step += 1

        reset_flag = False
        if self.current_step >= self.num_frames:
            self.current_step = 0
            reset_flag = True
        return motion, reset_flag

def extract_feet_height(robot, foot_body_names: list[str]) -> dict[str, float]:
    foot_body_indices = [robot.data.body_names.index(body_name) for body_name in foot_body_names]
    foot_heights = robot.data.body_pos_w[0, foot_body_indices, 2].detach().cpu().tolist()
    return {body_name: float(height) for body_name, height in zip(foot_body_names, foot_heights, strict=True)}


def extract_foot_heights(robot, foot_body_indices: list[int]) -> np.ndarray:
    return robot.data.body_pos_w[0, foot_body_indices, 2].detach().cpu().numpy().copy()


def log_in_air_events(has_ground_contact: np.ndarray, dt: float) -> None:
    ground_contact = np.asarray(has_ground_contact, dtype=bool).reshape(-1)
    in_air = ~ground_contact
    run_start: int | None = None

    for frame_index, is_in_air in enumerate(in_air):
        if is_in_air and run_start is None:
            run_start = frame_index
        elif not is_in_air and run_start is not None:
            run_end = frame_index - 1
            start_time = run_start * dt
            end_time = (run_end + 1) * dt
            print(
                f"[IN AIR]: frames {run_start}-{run_end}, "
                f"time {start_time:.3f}s-{end_time:.3f}s"
            )
            run_start = None

    if run_start is not None:
        run_end = len(in_air) - 1
        start_time = run_start * dt
        end_time = (run_end + 1) * dt
        print(
            f"[IN AIR]: frames {run_start}-{run_end}, "
            f"time {start_time:.3f}s-{end_time:.3f}s"
        )


def run_simulator(
    sim,
    scene,
    motion_data: GMRMotionData,
    simulation_app,
    output_file: Path,
    segment_bin_size: float,
    airborne_height_threshold: float,
):
    robot = scene["robot"]
    joint_indices = robot.find_joints(motion_data.joint_order, preserve_order=True)[0]

    log = {
        "fps": motion_data.fps,
        "joint_names": robot.data.joint_names,
        "body_names": robot.data.body_names,
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    foot_height_frames: list[np.ndarray] = []

    motion = motion_data.get_init_state()
    reference_root_pos, reference_root_rot, reference_root_lin_vel, reference_root_ang_vel, reference_joint_pose, reference_joint_vel = motion
    root_states = robot.data.default_root_state.clone()
    root_states[:, :3] = reference_root_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = reference_root_rot
    root_states[:, 7:10] = reference_root_lin_vel
    root_states[:, 10:] = reference_root_ang_vel
    robot.write_root_state_to_sim(root_states)
    # set joint state
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()

    joint_pos[:, joint_indices] = reference_joint_pose
    joint_vel[:, joint_indices] = reference_joint_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    #sim.step()
    sim.render()  # We don't want physic (sim.step())
    scene.update(sim.get_physics_dt())

    foot_body_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    foot_body_indices = [robot.data.body_names.index(body_name) for body_name in foot_body_names]
    feet_height = extract_feet_height(robot, foot_body_names)
    lowest_foot_height = min(feet_height.values()) - 0.025
    motion_data.set_root_height(-lowest_foot_height)
    print(f"[INFO]: Applied root height offset {-lowest_foot_height:.6f} from lowest foot height {lowest_foot_height:.6f}")

    while simulation_app.is_running():
        motion, reset_flag = motion_data.get_next_state()
        reference_root_pos, reference_root_rot, reference_root_lin_vel, reference_root_ang_vel, reference_joint_pose, reference_joint_vel = motion

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = reference_root_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = reference_root_rot
        root_states[:, 7:10] = reference_root_lin_vel
        root_states[:, 10:] = reference_root_ang_vel
        robot.write_root_state_to_sim(root_states)
        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()

        joint_pos[:, joint_indices] = reference_joint_pose
        joint_vel[:, joint_indices] = reference_joint_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        #sim.step()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())
        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            foot_height_frames.append(extract_foot_heights(robot, foot_body_indices))
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

            if reset_flag and not file_saved:
                file_saved = True
                for k in (
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ):
                    log[k] = np.stack(log[k], axis=0)

                ground_contact = infer_ground_contact_from_foot_heights(
                    np.stack(foot_height_frames, axis=0),
                    airborne_height_margin=airborne_height_threshold,
                )
                log_in_air_events(ground_contact, motion_data.physic_dt)

                segment_start_times, segment_end_times, segment_types = build_contact_segments(
                    has_ground_contact=ground_contact,
                    dt=motion_data.physic_dt,
                    duration=motion_data.physic_dt * motion_data.num_frames,
                    bin_size=segment_bin_size,
                )
                log["segment_start_times"] = segment_start_times
                log["segment_end_times"] = segment_end_times
                log["segment_types"] = segment_types

                np.savez(output_file, **log)
                print("[INFO]: Motion npz file saved to", output_file)
                break

def main(argv: list[str] | None = None):
    args_cli = build_parser().parse_args(argv)
    output_file = Path(args_cli.output_file) if args_cli.output_file else Path(args_cli.input_file).with_suffix(".npz")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene

        from .config.env_cfg import MotionViewerCfg, JOINT_ORDER

        motion_data = GMRMotionData(args_cli.input_file, args_cli.device, JOINT_ORDER, args_cli.height_offset)

        dt = motion_data.physic_dt
        render_interval = motion_data.render_interval

        sim_cfg = sim_utils.SimulationCfg(dt=dt, render_interval=render_interval, device=args_cli.device)
        sim = sim_utils.SimulationContext(sim_cfg)

        scene_cfg = MotionViewerCfg(1, env_spacing=2.0)
        scene = InteractiveScene(scene_cfg)

        sim.reset()
        print("[INFO]: Setup complete...")

        run_simulator(
            sim,
            scene,
            motion_data,
            simulation_app,
            output_file,
            args_cli.segment_bin_size,
            args_cli.airborne_height_threshold,
        )
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)

if __name__ == "__main__":
    main()
