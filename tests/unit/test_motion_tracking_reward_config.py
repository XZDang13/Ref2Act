import importlib
import sys
import types
from enum import Enum

import torch


def _quat_apply(quat, vec):
    return vec.clone()


def _quat_error_magnitude(q1, q2):
    return q1.new_zeros(q1.shape[:-1])


def _identity_configclass(cls):
    return cls


class _Cfg:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        if args:
            self.name = args[0]
        for key, value in kwargs.items():
            setattr(self, key, value)

    def replace(self, **updates):
        params = dict(self.kwargs)
        params.update(updates)
        return type(self)(*self.args, **params)


def _restore_modules(previous_modules, previous_attrs, sentinel):
    import isaaclab

    for module_name, previous_module in previous_modules.items():
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    for attr_name, previous_attr in previous_attrs.items():
        if previous_attr is sentinel:
            if hasattr(isaaclab, attr_name):
                delattr(isaaclab, attr_name)
        else:
            setattr(isaaclab, attr_name, previous_attr)


def _load_motion_tracking_env_module():
    import isaaclab

    sentinel = object()
    previous_modules = {
        "isaaclab.assets": sys.modules.get("isaaclab.assets"),
        "isaaclab.envs": sys.modules.get("isaaclab.envs"),
        "isaaclab.sensors": sys.modules.get("isaaclab.sensors"),
        "isaaclab.sim": sys.modules.get("isaaclab.sim"),
        "isaaclab.utils": sys.modules.get("isaaclab.utils"),
        "isaaclab.utils.math": sys.modules.get("isaaclab.utils.math"),
        "ref2act.envs.motion_tracking.action": sys.modules.get("ref2act.envs.motion_tracking.action"),
        "ref2act.envs.motion_tracking.curriculum": sys.modules.get("ref2act.envs.motion_tracking.curriculum"),
        "ref2act.envs.motion_tracking.observation": sys.modules.get("ref2act.envs.motion_tracking.observation"),
        "ref2act.envs.motion_tracking.rewards": sys.modules.get("ref2act.envs.motion_tracking.rewards"),
        "ref2act.envs.motion_tracking.termination": sys.modules.get("ref2act.envs.motion_tracking.termination"),
        "ref2act.envs.motion_tracking.tracking_quality": sys.modules.get(
            "ref2act.envs.motion_tracking.tracking_quality"
        ),
        "ref2act.envs.motion_tracking.visualization": sys.modules.get("ref2act.envs.motion_tracking.visualization"),
        "ref2act.motion": sys.modules.get("ref2act.motion"),
    }
    previous_attrs = {
        "assets": getattr(isaaclab, "assets", sentinel),
        "envs": getattr(isaaclab, "envs", sentinel),
        "sensors": getattr(isaaclab, "sensors", sentinel),
        "sim": getattr(isaaclab, "sim", sentinel),
        "utils": getattr(isaaclab, "utils", sentinel),
    }

    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.Articulation = type("Articulation", (), {})

    envs_mod = types.ModuleType("isaaclab.envs")
    envs_mod.DirectRLEnv = type("DirectRLEnv", (), {})

    sensors_mod = types.ModuleType("isaaclab.sensors")
    sensors_mod.ContactSensor = type("ContactSensor", (), {})

    sim_mod = types.ModuleType("isaaclab.sim")

    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.quat_apply = _quat_apply
    math_mod.quat_apply_inverse = _quat_apply
    math_mod.quat_error_magnitude = _quat_error_magnitude
    math_mod.quat_inv = lambda q: q
    math_mod.quat_mul = lambda q1, q2: q1
    math_mod.yaw_quat = lambda q: q

    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.math = math_mod

    action_mod = types.ModuleType("ref2act.envs.motion_tracking.action")
    action_mod.ActionProcessor = type("ActionProcessor", (), {})

    curriculum_mod = types.ModuleType("ref2act.envs.motion_tracking.curriculum")
    curriculum_mod.TerminationThresholdCurriculum = type("TerminationThresholdCurriculum", (), {})

    observation_mod = types.ModuleType("ref2act.envs.motion_tracking.observation")
    observation_mod.Observation = type("Observation", (), {})

    termination_mod = types.ModuleType("ref2act.envs.motion_tracking.termination")
    termination_mod.FailureRule = type("FailureRule", (), {})
    termination_mod.Termination = type("Termination", (), {})
    termination_mod.TerminationContext = type("TerminationContext", (), {})
    termination_mod.TerminationSpec = type("TerminationSpec", (), {})

    visualization_mod = types.ModuleType("ref2act.envs.motion_tracking.visualization")
    visualization_mod.ReferenceMotionViewer = type("ReferenceMotionViewer", (), {})

    motion_mod = types.ModuleType("ref2act.motion")
    motion_mod.MotionLib = type("MotionLib", (), {})
    motion_mod.MotionSampler = type("MotionSampler", (), {})
    motion_mod.SamplingStrategy = Enum("SamplingStrategy", {"FailureWeighted": "failure_weighted"})
    motion_mod.SegmentSource = Enum("SegmentSource", {"Anchor": "anchor", "Time": "time"})

    sys.modules["isaaclab.assets"] = assets_mod
    sys.modules["isaaclab.envs"] = envs_mod
    sys.modules["isaaclab.sensors"] = sensors_mod
    sys.modules["isaaclab.sim"] = sim_mod
    sys.modules["isaaclab.utils"] = utils_mod
    sys.modules["isaaclab.utils.math"] = math_mod
    sys.modules["ref2act.envs.motion_tracking.action"] = action_mod
    sys.modules["ref2act.envs.motion_tracking.curriculum"] = curriculum_mod
    sys.modules["ref2act.envs.motion_tracking.observation"] = observation_mod
    sys.modules["ref2act.envs.motion_tracking.termination"] = termination_mod
    sys.modules["ref2act.envs.motion_tracking.visualization"] = visualization_mod
    sys.modules["ref2act.motion"] = motion_mod
    isaaclab.assets = assets_mod
    isaaclab.envs = envs_mod
    isaaclab.sensors = sensors_mod
    isaaclab.sim = sim_mod
    isaaclab.utils = utils_mod

    try:
        sys.modules.pop("ref2act.envs.motion_tracking.rewards", None)
        sys.modules.pop("ref2act.envs.motion_tracking.tracking_quality", None)
        sys.modules.pop("ref2act.envs.motion_tracking.env", None)
        rewards_mod = importlib.import_module("ref2act.envs.motion_tracking.rewards")
        env_mod = importlib.import_module("ref2act.envs.motion_tracking.env")
        return env_mod, rewards_mod
    finally:
        _restore_modules(previous_modules, previous_attrs, sentinel)


def _load_env_cfg_shared_module():
    import isaaclab

    sentinel = object()
    previous_modules = {
        "isaaclab.assets": sys.modules.get("isaaclab.assets"),
        "isaaclab.envs": sys.modules.get("isaaclab.envs"),
        "isaaclab.envs.mdp": sys.modules.get("isaaclab.envs.mdp"),
        "isaaclab.managers": sys.modules.get("isaaclab.managers"),
        "isaaclab.markers": sys.modules.get("isaaclab.markers"),
        "isaaclab.markers.config": sys.modules.get("isaaclab.markers.config"),
        "isaaclab.scene": sys.modules.get("isaaclab.scene"),
        "isaaclab.sensors": sys.modules.get("isaaclab.sensors"),
        "isaaclab.sim": sys.modules.get("isaaclab.sim"),
        "isaaclab.terrains": sys.modules.get("isaaclab.terrains"),
        "isaaclab.utils": sys.modules.get("isaaclab.utils"),
        "isaaclab.utils.math": sys.modules.get("isaaclab.utils.math"),
        "ref2act.envs.motion_tracking.action": sys.modules.get("ref2act.envs.motion_tracking.action"),
        "ref2act.envs.motion_tracking.curriculum": sys.modules.get("ref2act.envs.motion_tracking.curriculum"),
        "ref2act.envs.motion_tracking.observation": sys.modules.get("ref2act.envs.motion_tracking.observation"),
        "ref2act.envs.motion_tracking.randomization": sys.modules.get("ref2act.envs.motion_tracking.randomization"),
        "ref2act.envs.motion_tracking.rewards": sys.modules.get("ref2act.envs.motion_tracking.rewards"),
        "ref2act.envs.motion_tracking.termination": sys.modules.get("ref2act.envs.motion_tracking.termination"),
        "ref2act.envs.motion_tracking.tracking_quality": sys.modules.get(
            "ref2act.envs.motion_tracking.tracking_quality"
        ),
        "ref2act.motion.sampling": sys.modules.get("ref2act.motion.sampling"),
        "ref2act.robots._articulation_shared": sys.modules.get("ref2act.robots._articulation_shared"),
    }
    previous_attrs = {
        "assets": getattr(isaaclab, "assets", sentinel),
        "envs": getattr(isaaclab, "envs", sentinel),
        "managers": getattr(isaaclab, "managers", sentinel),
        "markers": getattr(isaaclab, "markers", sentinel),
        "scene": getattr(isaaclab, "scene", sentinel),
        "sensors": getattr(isaaclab, "sensors", sentinel),
        "sim": getattr(isaaclab, "sim", sentinel),
        "terrains": getattr(isaaclab, "terrains", sentinel),
        "utils": getattr(isaaclab, "utils", sentinel),
    }

    assets_mod = types.ModuleType("isaaclab.assets")
    assets_mod.Articulation = type("Articulation", (), {})
    assets_mod.ArticulationCfg = _Cfg
    assets_mod.AssetBaseCfg = _Cfg

    envs_mod = types.ModuleType("isaaclab.envs")
    envs_mod.DirectRLEnvCfg = type("DirectRLEnvCfg", (), {})

    envs_mdp_mod = types.ModuleType("isaaclab.envs.mdp")
    envs_mdp_mod.randomize_rigid_body_material = lambda *args, **kwargs: None
    envs_mdp_mod.randomize_joint_parameters = lambda *args, **kwargs: None
    envs_mdp_mod.push_by_setting_velocity = lambda *args, **kwargs: None

    managers_mod = types.ModuleType("isaaclab.managers")
    managers_mod.EventTermCfg = _Cfg
    managers_mod.SceneEntityCfg = _Cfg

    markers_mod = types.ModuleType("isaaclab.markers")
    markers_mod.VisualizationMarkersCfg = _Cfg

    markers_config_mod = types.ModuleType("isaaclab.markers.config")
    markers_config_mod.FRAME_MARKER_CFG = _Cfg()

    scene_mod = types.ModuleType("isaaclab.scene")
    scene_mod.InteractiveSceneCfg = _Cfg

    sensors_mod = types.ModuleType("isaaclab.sensors")
    sensors_mod.ContactSensor = type("ContactSensor", (), {})
    sensors_mod.ContactSensorCfg = _Cfg

    sim_mod = types.ModuleType("isaaclab.sim")
    sim_mod.RigidBodyMaterialCfg = _Cfg
    sim_mod.GroundPlaneCfg = _Cfg
    sim_mod.DomeLightCfg = _Cfg
    sim_mod.SimulationCfg = _Cfg

    terrains_mod = types.ModuleType("isaaclab.terrains")
    terrains_mod.TerrainImporterCfg = _Cfg
    terrains_mod.TerrainGeneratorCfg = _Cfg
    terrains_mod.HfSteppingStonesTerrainCfg = _Cfg

    math_mod = types.ModuleType("isaaclab.utils.math")
    math_mod.quat_apply = _quat_apply
    math_mod.quat_error_magnitude = _quat_error_magnitude
    math_mod.quat_inv = lambda q: q
    math_mod.quat_mul = lambda q1, q2: q1
    math_mod.yaw_quat = lambda q: q

    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.configclass = _identity_configclass
    utils_mod.math = math_mod

    action_mod = types.ModuleType("ref2act.envs.motion_tracking.action")
    action_mod.ActionSpec = _Cfg
    action_mod.ActionProcessor = type("ActionProcessor", (), {})

    curriculum_mod = types.ModuleType("ref2act.envs.motion_tracking.curriculum")
    curriculum_mod.TerminationCurriculumCfg = type("TerminationCurriculumCfg", (), {})

    observation_mod = types.ModuleType("ref2act.envs.motion_tracking.observation")
    observation_mod.default_training_observation_spec = lambda add_noise: _Cfg(add_noise=add_noise)

    randomization_mod = types.ModuleType("ref2act.envs.motion_tracking.randomization")
    randomization_mod.randomize_action_latency = lambda *args, **kwargs: None
    randomization_mod.randomize_group_actuator_gains = lambda *args, **kwargs: None
    randomization_mod.randomize_group_body_masses = lambda *args, **kwargs: None
    randomization_mod.randomize_rigid_body_collider_offsets_by_body = lambda *args, **kwargs: None
    randomization_mod.randomize_rigid_body_com_from_default = lambda *args, **kwargs: None

    termination_mod = types.ModuleType("ref2act.envs.motion_tracking.termination")
    termination_mod.default_termination_spec = lambda **kwargs: _Cfg(**kwargs)
    termination_mod.FailureRule = type("FailureRule", (), {})
    termination_mod.TerminationContext = type("TerminationContext", (), {})

    sampling_mod = types.ModuleType("ref2act.motion.sampling")
    sampling_mod.SamplerMod = Enum("SamplerMod", {"Clamp": "clamp"})
    sampling_mod.SamplingStrategy = Enum("SamplingStrategy", {"FailureWeighted": "failure_weighted"})
    sampling_mod.SegmentSource = Enum("SegmentSource", {"Time": "time", "Anchor": "anchor"})

    articulation_shared_mod = types.ModuleType("ref2act.robots._articulation_shared")
    articulation_shared_mod.G1_CFG = _Cfg(prim_path="/World/G1")

    sys.modules["isaaclab.assets"] = assets_mod
    sys.modules["isaaclab.envs"] = envs_mod
    sys.modules["isaaclab.envs.mdp"] = envs_mdp_mod
    sys.modules["isaaclab.managers"] = managers_mod
    sys.modules["isaaclab.markers"] = markers_mod
    sys.modules["isaaclab.markers.config"] = markers_config_mod
    sys.modules["isaaclab.scene"] = scene_mod
    sys.modules["isaaclab.sensors"] = sensors_mod
    sys.modules["isaaclab.sim"] = sim_mod
    sys.modules["isaaclab.terrains"] = terrains_mod
    sys.modules["isaaclab.utils"] = utils_mod
    sys.modules["isaaclab.utils.math"] = math_mod
    sys.modules["ref2act.envs.motion_tracking.action"] = action_mod
    sys.modules["ref2act.envs.motion_tracking.curriculum"] = curriculum_mod
    sys.modules["ref2act.envs.motion_tracking.observation"] = observation_mod
    sys.modules["ref2act.envs.motion_tracking.randomization"] = randomization_mod
    sys.modules["ref2act.envs.motion_tracking.termination"] = termination_mod
    sys.modules["ref2act.motion.sampling"] = sampling_mod
    sys.modules["ref2act.robots._articulation_shared"] = articulation_shared_mod
    isaaclab.assets = assets_mod
    isaaclab.envs = envs_mod
    isaaclab.managers = managers_mod
    isaaclab.markers = markers_mod
    isaaclab.scene = scene_mod
    isaaclab.sensors = sensors_mod
    isaaclab.sim = sim_mod
    isaaclab.terrains = terrains_mod
    isaaclab.utils = utils_mod

    try:
        sys.modules.pop("ref2act.envs.motion_tracking.rewards", None)
        sys.modules.pop("ref2act.envs.motion_tracking.tracking_quality", None)
        sys.modules.pop("ref2act.robots._env_cfg_shared", None)
        rewards_mod = importlib.import_module("ref2act.envs.motion_tracking.rewards")
        shared_mod = importlib.import_module("ref2act.robots._env_cfg_shared")
        return shared_mod, rewards_mod
    finally:
        _restore_modules(previous_modules, previous_attrs, sentinel)


def test_build_reward_spec_autofills_end_effector_indices_alongside_existing_fields() -> None:
    env_mod, rewards_mod = _load_motion_tracking_env_module()

    env = object.__new__(env_mod.MotionTrackingEnv)
    env.anchor_body_index = 3
    env.key_body_indices = [4, 5]
    env.end_effector_body_indices = [6, 7]
    env.step_dt = 0.02
    env.cfg = types.SimpleNamespace(
        rewards=rewards_mod.RewardSpec(
            dt=0.0,
            terms=(
                rewards_mod.KeyPositionRewardTermCfg(),
                rewards_mod.EndEffectorPositionRewardTermCfg(),
                rewards_mod.EndEffectorVelocityRewardTermCfg(),
                rewards_mod.FootSlipPenaltyTermCfg(),
                rewards_mod.SelfCollisionPenaltyTermCfg(),
                rewards_mod.CoMSupportRewardTermCfg(),
            ),
        )
    )

    reward_spec = env_mod.MotionTrackingEnv._build_reward_spec(
        env,
        collision_track_body_indices=(8, 9),
        foot_body_indices=(10, 11),
        foot_contact_body_indices=(12, 13),
    )

    assert reward_spec.terms[0].key_body_indices == (4, 5)
    assert reward_spec.terms[1].end_effector_body_indices == (6, 7)
    assert reward_spec.terms[2].anchor_body_index == 3
    assert reward_spec.terms[2].end_effector_body_indices == (6, 7)
    assert reward_spec.terms[3].foot_body_indices == (10, 11)
    assert reward_spec.terms[3].foot_contact_body_indices == (12, 13)
    assert reward_spec.terms[4].body_indices == (8, 9)
    assert reward_spec.terms[5].foot_body_indices == (10, 11)
    assert reward_spec.terms[5].foot_contact_body_indices == (12, 13)
    assert reward_spec.dt == 0.02


def test_get_dones_preserves_legacy_failure_recording_when_quality_gate_disabled() -> None:
    env_mod, _ = _load_motion_tracking_env_module()
    env = object.__new__(env_mod.MotionTrackingEnv)

    class _Termination:
        terminated_env_ids = torch.tensor([0], dtype=torch.long)

        def get_dones(self, *args):
            return torch.tensor([True, False]), torch.tensor([False, True])

    recorded: list[torch.Tensor] = []
    env.cfg = types.SimpleNamespace(robust_tracking=types.SimpleNamespace(enabled=False))
    env.termination_curriculum = types.SimpleNamespace(apply=lambda step: {}, has_schedules=False)
    env.common_step_counter = 7
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.max_episode_length = torch.full((2,), 10, dtype=torch.long)
    env.robot = object()
    env.reference_motion = object()
    env.termination_model = _Termination()
    env.tracking_quality_gate = None
    env.sampler = types.SimpleNamespace(record_failures=lambda env_ids: recorded.append(env_ids.clone()))
    env.extras = {}

    terminate, time_out = env_mod.MotionTrackingEnv._get_dones(env)

    assert torch.equal(terminate, torch.tensor([True, False]))
    assert torch.equal(time_out, torch.tensor([False, True]))
    assert len(recorded) == 1
    assert torch.equal(recorded[0], torch.tensor([0]))


def test_get_dones_quality_gate_records_only_quality_failure_mask() -> None:
    env_mod, _ = _load_motion_tracking_env_module()
    env = object.__new__(env_mod.MotionTrackingEnv)

    quality = env_mod.TrackingQualityResult(
        state=torch.tensor([0, 2], dtype=torch.long),
        score=torch.tensor([0.5, 2.0], dtype=torch.float32),
        previous_score=torch.tensor([0.5, 1.5], dtype=torch.float32),
        soft_violation_mask=torch.tensor([False, False]),
        recovery_needed_mask=torch.tensor([False, True]),
        hard_tracking_failure_mask=torch.tensor([False, False]),
        record_failure_mask=torch.tensor([False, True]),
        per_rule_errors={"anchor_position_failure": torch.tensor([0.1, 0.4], dtype=torch.float32)},
        per_rule_normalized_errors={"anchor_position_failure": torch.tensor([0.5, 2.0], dtype=torch.float32)},
    )
    recorded: list[torch.Tensor] = []
    tracked: list[torch.Tensor] = []

    class _Termination:
        def build_context(self, *args):
            return object()

        def evaluate_timeouts(self, context):
            return torch.tensor([False, True])

        def track_terminated_env_ids(self, failed):
            tracked.append(failed.clone())

    env.cfg = types.SimpleNamespace(
        robust_tracking=types.SimpleNamespace(
            enabled=True,
            quality_gate=types.SimpleNamespace(enabled=True),
        )
    )
    env.termination_curriculum = types.SimpleNamespace(apply=lambda step: {}, has_schedules=False)
    env.common_step_counter = 7
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.max_episode_length = torch.full((2,), 10, dtype=torch.long)
    env.robot = object()
    env.reference_motion = object()
    env.termination_model = _Termination()
    env.tracking_quality_gate = types.SimpleNamespace(
        cfg=types.SimpleNamespace(log_quality_counts=True, log_per_rule_errors=True),
        evaluate=lambda context: quality,
    )
    env.sampler = types.SimpleNamespace(record_failures=lambda env_ids: recorded.append(env_ids.clone()))
    env.extras = {}

    terminate, time_out = env_mod.MotionTrackingEnv._get_dones(env)

    assert torch.equal(terminate, torch.tensor([False, False]))
    assert torch.equal(time_out, torch.tensor([False, True]))
    assert len(recorded) == 1
    assert torch.equal(recorded[0], torch.tensor([1]))
    assert len(tracked) == 1
    assert torch.equal(tracked[0], torch.tensor([False, False]))
    assert env.extras["sampler/recorded_recovery_failure_count"].item() == 1.0
    assert env.extras["tracking_quality/recovery_needed_rate"].item() == 0.5


def test_rewards_and_dones_reuse_one_tracking_quality_evaluation_per_step() -> None:
    env_mod, _ = _load_motion_tracking_env_module()
    env = object.__new__(env_mod.MotionTrackingEnv)
    quality = env_mod.TrackingQualityResult(
        state=torch.tensor([0, 2], dtype=torch.long),
        score=torch.tensor([0.5, 2.0], dtype=torch.float32),
        previous_score=torch.tensor([0.5, 1.5], dtype=torch.float32),
        soft_violation_mask=torch.tensor([False, False]),
        recovery_needed_mask=torch.tensor([False, True]),
        hard_tracking_failure_mask=torch.tensor([False, False]),
        record_failure_mask=torch.tensor([False, True]),
        per_rule_errors={},
        per_rule_normalized_errors={},
    )
    evaluate_calls: list[object] = []
    recorded: list[torch.Tensor] = []
    reward_quality: list[object] = []

    class _Termination:
        def build_context(self, *args):
            return object()

        def evaluate_timeouts(self, context):
            return torch.tensor([False, False])

        def track_terminated_env_ids(self, failed):
            return torch.nonzero(failed, as_tuple=False).squeeze(-1)

    env.cfg = types.SimpleNamespace(
        robust_tracking=types.SimpleNamespace(
            enabled=True,
            quality_gate=types.SimpleNamespace(enabled=True),
        )
    )
    env.termination_curriculum = types.SimpleNamespace(apply=lambda step: {}, has_schedules=False)
    env.common_step_counter = 7
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.max_episode_length = torch.full((2,), 10, dtype=torch.long)
    env.robot = object()
    env.reference_motion = object()
    env.contact_sensor = object()
    env.action_processer = object()
    env.termination_model = _Termination()
    env.tracking_quality_gate = types.SimpleNamespace(
        cfg=types.SimpleNamespace(log_quality_counts=True, log_per_rule_errors=True),
        evaluate=lambda context: evaluate_calls.append(context) or quality,
    )
    env.tracking_quality_output = None
    env._tracking_quality_cache_step = None
    env.reward_model = types.SimpleNamespace(
        get_task_reward=lambda *args, tracking_quality=None: reward_quality.append(tracking_quality)
        or torch.tensor([1.0, 1.0], dtype=torch.float32)
    )
    env.sampler = types.SimpleNamespace(record_failures=lambda env_ids: recorded.append(env_ids.clone()))
    env.extras = {}

    reward = env_mod.MotionTrackingEnv._get_rewards(env)
    terminate, _ = env_mod.MotionTrackingEnv._get_dones(env)

    assert torch.equal(reward, torch.tensor([1.0, 1.0], dtype=torch.float32))
    assert torch.equal(terminate, torch.tensor([False, False]))
    assert len(evaluate_calls) == 1
    assert reward_quality == [quality]
    assert len(recorded) == 1
    assert torch.equal(recorded[0], torch.tensor([1]))


def test_g1_reward_spec_uses_robust_defaults_with_com_terms() -> None:
    shared_mod, rewards_mod = _load_env_cfg_shared_module()

    g1_terms = shared_mod.G1MotionTrackingEnvCfg.rewards.terms

    assert [term.type for term in g1_terms] == [
        term.type for term in rewards_mod.robust_tracking_reward_spec(dt=0.0, include_com_terms=True).terms
    ]
    assert rewards_mod.CoMPositionRewardTerm.type_name in [term.type for term in g1_terms]
    assert rewards_mod.CoMVelocityRewardTerm.type_name in [term.type for term in g1_terms]
    assert rewards_mod.CoMSupportRewardTerm.type_name in [term.type for term in g1_terms]
    assert shared_mod.G1MotionTrackingEnvCfg.robust_tracking.enabled is True
    assert shared_mod.G1MotionTrackingEnvCfg.robust_tracking.quality_gate.enabled is True
