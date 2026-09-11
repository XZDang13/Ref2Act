"""Frozen V7/V8/V9 transfer reward formulas for training/replay compatibility.

New reward design belongs in standup_stages.py. Keep the legacy formulas intact.
"""
import torch
from .recovery_reward import smooth_height_gate


def righting_quality(upright_projection, cfg):
    """Signed feedback below horizontal, with a broad lean-tolerant plateau."""
    return ((upright_projection.clamp(-1, 1) + 1) / (1 + cfg["righting_target_projection"])).clamp(0, 1)


def independent_height_deficits(root_score, shoulder_score, *, step_dt, cfg, upright_projection=None, squat_quality=None, root_to_shoulder_ratio=None):
    """Keep upper-body progress visible even while the pelvis stays low."""
    # Optional discovery objective: once the torso is off the floor, permit
    # forward lean without paying for shoulder height all the way to standing.
    # Final body-height/success objectives continue to use the full raw height.
    upper_credit = (shoulder_score / cfg.get("upper_preparation_full", 1.)).clamp(0, 1)
    pelvis_credit = root_score.clamp(0, 1)
    if cfg.get("bilateral_squat", False):
        if squat_quality is None:
            raise ValueError("Bilateral pelvis height requires squat support quality.")
        if cfg.get("upper_relative_before_support", False):
            if root_to_shoulder_ratio is None:
                raise ValueError("Relative torso preparation requires the target height ratio.")
            relative_upper = (shoulder_score - root_score * root_to_shoulder_ratio).clamp(0, 1)
            upper_credit = relative_upper + squat_quality * (upper_credit - relative_upper)
        # V9 uses zero free pelvis credit; torso righting still guides sitting.
        preparation = pelvis_credit.clamp_max(cfg["pelvis_preparation_full"])
        pelvis_credit = preparation + squat_quality * (pelvis_credit - preparation)
    if cfg.get("righting_scale", 0) > 0:
        if upright_projection is None:
            raise ValueError("Righting-aware pelvis credit requires the signed upright projection.")
        pelvis_credit = pelvis_credit * righting_quality(upright_projection, cfg)
    return {
        "pelvis": -step_dt * cfg["pelvis_height_scale"] * (1 - pelvis_credit),
        "upper": -step_dt * cfg["upper_height_scale"] * (1 - upper_credit),
    }


def shank_preparation_gate(shoulder_score, cfg, root_score=None):
    if "shank_gate_start" not in cfg:
        return torch.ones_like(shoulder_score)
    if cfg.get("shank_gate_source", "shoulder") == "pelvis":
        if root_score is None:
            raise ValueError("Pelvis-gated shank preparation requires root_score.")
        shoulder_score = root_score
    return smooth_height_gate(shoulder_score, start=cfg["shank_gate_start"], full=cfg["shank_gate_full"])


def support_transfer_rewards(quality, load, previous_peak, body_score, *, step_dt, cfg, shoulder_score=None, upright_projection=None, root_score=None, contact_load=None):
    """Preparation deficits -> load transfer -> supported rise, all bounded.

    Prepared sitting still incurs task deficits; a negative reward baseline
    alone does not prevent a seated local optimum.
    Load progress pays only for episode-best load, never for repeated stomping.
    Unlike the old potential pilot, these task rewards need no bootstrap change.
    """
    from .standup_support import prepared_support_load, squat_support_quality, supported_pelvis_lift_quality
    load, geometry_gates = prepared_support_load(quality, load, cfg)
    squat = squat_support_quality(quality, cfg) if cfg.get("bilateral_squat", False) else None
    height = body_score.clamp(0, 1)
    rise = smooth_height_gate(height, start=cfg["rise_start"], full=cfg["rise_full"])
    preparation_weight = 1 - load * rise * (squat if squat is not None else 1.)
    rewards = {
        f"{name}_deficit": -step_dt * cfg[f"{name}_scale"] * preparation_weight * (1 - quality[name])
        for name in ("feet_low", "feet_flat", "shank_upright")
    }
    if "shank_gate_start" in cfg:
        if shoulder_score is None:
            raise ValueError("Staged shank preparation requires the independent shoulder score.")
        gate = shank_preparation_gate(shoulder_score, cfg, root_score)
        # Gate the credit, not the cost: rising must not switch on a new penalty.
        rewards["shank_upright_deficit"] = -step_dt * cfg["shank_upright_scale"] * preparation_weight * (1 - gate * quality["shank_upright"])
    rewards["load_deficit"] = -step_dt * cfg["load_scale"] * (1 - load)
    if cfg.get("contact_scale", 0) > 0:
        if contact_load is None:
            raise ValueError("Contact exploration requires persistent ground load.")
        # A small separate credit for trying ground contact; never used for success.
        rewards["contact_deficit"] = -step_dt * cfg["contact_scale"] * (1 - contact_load.clamp(0, 1))
    rise_deficit = 1 - load * height if cfg.get("supported_rise_mode", "legacy") == "joint" else load * (1 - height)
    if squat is not None:
        rise_deficit = 1 - squat * load * height
    rewards["supported_rise_deficit"] = -step_dt * cfg["supported_rise_scale"] * rise_deficit
    for name in ("com_preparation", "bilateral_load", "foot_placement", "sole_plant"):
        if cfg.get(f"{name}_scale", 0) > 0:
            credit = quality[name]
            if name == "sole_plant" and "sole_plant_hip_floor" in cfg:
                # Couple each sole to its own geometry: let the other foot
                # reposition without losing credit for an already placed foot.
                floor = cfg["sole_plant_hip_floor"]
                per_foot = quality["sole_plant_per_foot"] * (
                    floor + (1 - floor) * (quality["hip_geometry_per_foot"]
                        if geometry_gates is None else geometry_gates))
                leading = per_foot.amax(-1) if cfg.get("allow_first_foot_preparation", False) else per_foot.amin(-1)
                credit = .5 * (per_foot.mean(-1) + leading)
            if name == "bilateral_load" and geometry_gates is not None:
                credit = credit * geometry_gates.amin(-1)
            if name == "bilateral_load" and squat is not None:
                credit = squat
            rewards[f"{name}_deficit"] = -step_dt * cfg[f"{name}_scale"] * (1 - credit)
    if cfg.get("pelvis_lift_scale", 0) > 0:
        if root_score is None or upright_projection is None:
            raise ValueError("Supported pelvis lift requires root score and signed upright projection.")
        lift = supported_pelvis_lift_quality(root_score, load, upright_projection, cfg=cfg,
                                              hip_quality=quality.get("hip_geometry"), squat_quality=squat)
        rewards["pelvis_lift_deficit"] = -step_dt * cfg["pelvis_lift_scale"] * (1 - lift)
    rewards["load_progress"] = cfg["load_progress_scale"] * (load - previous_peak).clamp_min(0)
    if cfg.get("righting_scale", 0) > 0:
        if upright_projection is None:
            raise ValueError("Early righting requires the signed upright projection.")
        rewards["righting_deficit"] = -step_dt * cfg["righting_scale"] * (1 - righting_quality(upright_projection, cfg))
    if cfg.get("com_scale", 0) > 0:
        # Joint support/alignment deficit, NOT gate * distance_penalty: losing
        # contact removes alignment credit instead of switching off a cost.
        alignment = quality["com_alignment"]
        if geometry_gates is not None:
            alignment = alignment * geometry_gates.amin(-1)
        rewards["com_deficit"] = -step_dt * cfg["com_scale"] * (1 - alignment)
    return rewards, torch.maximum(previous_peak, load), preparation_weight
