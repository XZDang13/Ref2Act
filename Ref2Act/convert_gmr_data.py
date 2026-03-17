import argparse
import pickle
from pathlib import Path

import numpy as np

import torch
from isaaclab.app import AppLauncher


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

def is_contact(net_contact_forces, body_ids):
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > 10.0

    print(is_contact)

def run_simulator(sim, scene, motion_data: GMRMotionData, simulation_app, output_file: Path):
    robot = scene["robot"]
    contact_sensors = scene["contact_sensor"]
    joint_indices = robot.find_joints(motion_data.joint_order, preserve_order=True)[0]
    contact_tracking_body_indices, _ = contact_sensors.find_bodies([
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ])

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
        #net_force = contact_sensors.data.net_forces_w_history
        #is_contact(net_force, contact_tracking_body_indices)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
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


                np.savez(output_file, **log)
                print("[INFO]: Motion npz file saved to", output_file)

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

        run_simulator(sim, scene, motion_data, simulation_app, output_file)
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
