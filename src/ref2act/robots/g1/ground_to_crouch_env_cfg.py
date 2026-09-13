from isaaclab.utils.configclass import configclass
from .stand_up_env_cfg import G1StandUpEnvCfg
from ref2act.envs.stand_up.crouch_defaults import default_config

@configclass
class G1GroundToCrouchEnvCfg(G1StandUpEnvCfg):
    """Supine to autonomous bilateral crouch; no standing target."""
    task: dict = default_config()
