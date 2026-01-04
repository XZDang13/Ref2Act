import torch
from isaaclab.assets import Articulation
from .buffer import DequeBuffer
from .utils import IndexLike

class ActionProcessor:
    
    def __init__(self, robot: Articulation, action_buffer_length: int=1):
        self.scale: float|torch.Tensor = 0.0
        self.offset: float|torch.Tensor = 0.0
        self.device = robot.data.device
        self.num_env = robot.data.joint_pos.size(0)
        self.action_size = robot.data.joint_pos.size(1)

        self.action_buffer = DequeBuffer(self.num_env, action_buffer_length, (self.action_size, ),
                                         device=self.device)

        self.delays: torch.Tensor|None = None
        self.applied_action = torch.zeros_like(robot.data.default_joint_pos)

    def set_median_scale_offset(
        self,
        robot: Articulation
    ) -> None:
        self.joint_low_limit = robot.data.joint_pos_limits[0, :, 0]
        self.joint_up_limit = robot.data.joint_pos_limits[0, :, 1]
        self.scale = 0.5 * (self.joint_up_limit - self.joint_low_limit)
        self.offset = 0.5 * (self.joint_up_limit + self.joint_low_limit)
    
    def set_robot_default_scale_offset(
        self,
        robot: Articulation,
        scale: float
    ):
        self.joint_low_limit = robot.data.joint_pos_limits[0]
        self.joint_up_limit = robot.data.joint_pos_limits[1]
        self.scale = scale
        self.offset = robot.data.default_joint_pos

    def reset_action_buffer(self, env_ids:IndexLike):
        self.action_buffer.reset(env_ids)

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        
        return action * self.scale + self.offset
    
    def set_random_delays(self, ranges:tuple[int, int]):
        if self.delays is None:
            self.delays = torch.empty(self.num_env, dtype=torch.long)

        self.delays.uniform_(ranges[0], ranges[1])
    
    def pre_process_action(self, action: torch.Tensor):
        self.action_buffer.append(action)

        if self.delays is None:
            self.applied_action = self.action_buffer.latest()
        else:
            self.applied_action = self.action_buffer.get(self.delays)

        self.target_joint_position = self.applied_action * self.scale + self.offset
        self.target_joint_position.clamp_(self.joint_low_limit, self.joint_up_limit)

    