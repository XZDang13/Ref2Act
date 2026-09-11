"""Three-stage stand-up objectives, independent of simulation and legacy rewards.

Credits are cumulative milestones, not mutually exclusive negative penalties.
Soft phase weights describe transitions; they do not introduce hidden phase state.
Only the ground-contact reader carries persistence, and contact loss revokes credit.
"""
from dataclasses import dataclass
import math
import torch
from .recovery_reward import smooth_height_gate


@dataclass(frozen=True)
class StageInputs:
    root_height: torch.Tensor
    shoulder_height: torch.Tensor
    root_target: float
    shoulder_target: float
    upright: torch.Tensor
    foot_distance: torch.Tensor
    sole_plant: torch.Tensor
    usable_load: torch.Tensor
    persistent_contact: torch.Tensor
    filtered_load: torch.Tensor
    com_distance: torch.Tensor
    linear_speed: torch.Tensor
    angular_speed: torch.Tensor
    joint_speed: torch.Tensor
    torso_forward_lean: torch.Tensor | None = None  # radians; positive forward
    assistance_ratio: torch.Tensor | None = None  # actual upward force / mg
    filtered_ground_fraction: torch.Tensor | None = None
    sole_heading_xy: torch.Tensor | None = None
    com_margin: torch.Tensor | None = None
    torso_upright: torch.Tensor | None = None
    foot_low: torch.Tensor | None = None
    foot_flat: torch.Tensor | None = None
    foot_clearance: torch.Tensor | None = None


@dataclass(frozen=True)
class StageRewards:
    rewards: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]
    assistance_gate: torch.Tensor


def validate_stages(cfg):
    if 'foot_stance_v5' in cfg:
        if cfg['foot_stance_v5'] is not True or not cfg.get('foot_ground_retraction_v4',False) or 'stance' not in cfg:
            raise ValueError('foot_stance_v5 requires ground retraction and stance geometry')
        cfg = {k:v for k,v in cfg.items() if k!='foot_stance_v5'}
    if 'foot_ground_retraction_v4' in cfg:
        if cfg['foot_ground_retraction_v4'] is not True or any(cfg.get(k,False) for k in ('foot_retraction_v3','foot_discovery_v2')):
            raise ValueError('foot_ground_retraction_v4 replaces previous foot discovery flags')
        cfg = {k:v for k,v in cfg.items() if k!='foot_ground_retraction_v4'}
    if 'foot_retraction_v3' in cfg:
        if cfg['foot_retraction_v3'] is not True or cfg.get('foot_discovery_v2',False):
            raise ValueError('foot_retraction_v3 must be true and replaces foot_discovery_v2')
        cfg = {k:v for k,v in cfg.items() if k!='foot_retraction_v3'}
    if 'rise_height_v3' in cfg:
        if cfg['rise_height_v3'] is not True or not cfg.get('rise_hold_v2',False):
            raise ValueError('rise_height_v3 requires rise_hold_v2 support sensing')
        cfg = {k:v for k,v in cfg.items() if k!='rise_height_v3'}
    if 'foot_discovery_v2' in cfg:
        if cfg['foot_discovery_v2'] is not True:
            raise ValueError('foot_discovery_v2 must be true when specified')
        cfg = {k:v for k,v in cfg.items() if k!='foot_discovery_v2'}
    if 'rise_hold_v2' in cfg:
        if cfg['rise_hold_v2'] is not True or 'rise_transfer' not in cfg:
            raise ValueError('rise_hold_v2 requires rise transfer')
        cfg = {k:v for k,v in cfg.items() if k!='rise_hold_v2'}
    if 'stance' in cfg:
        if 'rise_transfer' not in cfg or not cfg.get('hands',{}).get('enabled',False):
            raise ValueError('Stance progress requires rise transfer and hand sensing')
        c = cfg['stance']
        keys = {'fraction','clearance_full','clearance_sigma','side_full','side_sigma',
                'width_full','width_zero','width_sigma'}
        if set(c)!=keys or any(not math.isfinite(float(v)) or v<=0 for v in c.values()):
            raise ValueError('Invalid bilateral stance settings')
        if not (c['fraction']<1 and 2*c['side_full']<c['width_full']<c['width_zero']):
            raise ValueError('Invalid stance reward fraction or width range')
        cfg = {k:v for k,v in cfg.items() if k!='stance'}
    if 'rise_transfer' in cfg:
        c = cfg['rise_transfer']
        if c.get('unload_curve','smoothstep') not in ('smoothstep','rational'):
            raise ValueError('Invalid hand unload curve')
        c = {k:v for k,v in c.items() if k!='unload_curve'}
        expected = {'unload_fraction', 'lean_fraction', 'hand_load_full', 'hand_load_zero',
                    'lean_start_deg', 'lean_full_deg', 'lean_upper_deg', 'lean_zero_deg',
                    'fade_start', 'fade_full', 'minimum_ground_weight'}
        if set(c) != expected or any(not math.isfinite(float(v)) or v < 0 for v in c.values()):
            raise ValueError('Invalid rise transfer settings')
        if not (0 < c['unload_fraction'] and 0 < c['lean_fraction']
                and c['unload_fraction'] + c['lean_fraction'] < .4):
            raise ValueError('Rise transfer must preserve height and balance credit')
        if not (0 <= c['hand_load_full'] < c['hand_load_zero'] <= 1
                and 0 <= c['lean_start_deg'] < c['lean_full_deg'] <= c['lean_upper_deg'] < c['lean_zero_deg'] < 90
                and 0 <= c['fade_start'] < c['fade_full'] < cfg['standing_start']):
            raise ValueError('Invalid rise transfer intervals')
        if not cfg.get('hands', {}).get('enabled', False):
            raise ValueError('Rise transfer requires ground-only hand sensing')
        if not 0 < c['minimum_ground_weight'] <= 1:
            raise ValueError('Invalid minimum ground weight')
        cfg = {k:v for k,v in cfg.items() if k != 'rise_transfer'}
    if 'hands' in cfg:
        from .standup_hands import validate_hands
        validate_hands(cfg['hands'])
        cfg = {k:v for k,v in cfg.items() if k != 'hands'}
    expected = {'distance_full', 'distance_start', 'distance_zero', 'plant_start', 'plant_full',
        'touch_start', 'touch_full', 'load_full', 'torso_start', 'torso_full',
        'righting_projection', 'standing_start', 'standing_full', 'upright_start', 'upright_full',
        'com_sigma', 'linear_speed_scale', 'angular_speed_scale', 'joint_speed_scale',
        'gate_assistance', 'preparation_righting_fraction', 'rise_torso_start', 'rise_torso_full',
        'rise_upright_start', 'rise_upright_full', 'weights'}
    if set(cfg) != expected:
        raise ValueError(f"Stage configuration keys differ: {set(cfg) ^ expected}")
    for key in expected - {'weights', 'gate_assistance'}:
        if not math.isfinite(float(cfg[key])) or cfg[key] < 0:
            raise ValueError(f"Invalid stage setting {key}")
    for key in ('distance_zero','load_full','righting_projection','com_sigma',
                'linear_speed_scale','angular_speed_scale','joint_speed_scale'):
        if cfg[key] <= 0:
            raise ValueError(f"Stage setting {key} must be positive")
    if not cfg['distance_full'] < cfg['distance_start'] < cfg['distance_zero']:
        raise ValueError('Require distance_full < distance_start < distance_zero')
    for prefix in ('plant','touch','torso','standing','upright','rise_torso','rise_upright'):
        if not 0 <= cfg[prefix+'_start'] < cfg[prefix+'_full'] <= 1:
            raise ValueError(f'Invalid stage interval: {prefix}')
    if not cfg['touch_full'] < cfg['load_full'] <= 1 or cfg['righting_projection'] > 1:
        raise ValueError('Touch readiness must precede full load transfer')
    if type(cfg['gate_assistance']) is not bool:
        raise ValueError('gate_assistance must be boolean')
    if not 0 < cfg['preparation_righting_fraction'] < 1:
        raise ValueError('Both righting and feet need nonzero preparation credit')
    if cfg['rise_torso_full'] > cfg['torso_full']:
        raise ValueError('Rise must not require completing torso preparation')
    if set(cfg['weights']) != {'preparation','rise','stand'} or any(
        not math.isfinite(float(v)) or v <= 0 for v in cfg['weights'].values()):
        raise ValueError('Specify three positive stage reward weights')


def _gate(value, start, full):
    return smooth_height_gate((value-start)/(full-start), start=0., full=1.)


def _both_score(value):
    # Either foot may improve first; a single good foot never means completion.
    return .5 * (value.mean(-1) + value.amin(-1))


def hand_unload_quality(hand_load, cfg):
    if cfg.get('unload_curve') == 'rational':
        excess = (hand_load-cfg['hand_load_full']).clamp_min(0)
        return 1/(1+(excess/(cfg['hand_load_zero']-cfg['hand_load_full'])).square())
    return _gate(-hand_load,-cfg['hand_load_zero'],-cfg['hand_load_full'])


def signed_torso_lean(quaternion):
    """Sagittal lean from a normalized torso world quaternion (x,y,z,w).

    Positive means the torso's local forward axis points downwards. World
    yaw does not affect this angle; lateral tilt alone is not forward lean.
    """
    x,y,z,w = quaternion.unbind(-1)
    return torch.atan2(2*(w*y-x*z),1-2*(x*x+y*y))


def stage_rewards(x: StageInputs, cfg, *, step_dt: float, hand_state=None, reward_form="deficit") -> StageRewards:
    relative_torso = ((x.shoulder_height-x.root_height)/x.shoulder_target).clamp(0,1)
    orientation = ((x.upright.clamp(-1,1)+1)/(1+cfg['righting_projection'])).clamp(0,1)
    torso = _gate(relative_torso, cfg['torso_start'], cfg['torso_full'])
    # A broad physical readiness check, not completion of sitting upright.
    body_ready = torch.minimum(
        _gate(relative_torso, cfg['rise_torso_start'], cfg['rise_torso_full']),
        _gate(x.upright, cfg['rise_upright_start'], cfg['rise_upright_full']))
    righting = .5*(orientation+torso)

    distance = x.foot_distance.clamp_min(0)
    near = _gate(-distance, -cfg['distance_start'], -cfg['distance_full'])
    # Constant improvement per metre within the discovery range. Independent
    # of sole height/contact: moving a lifted foot inward still earns credit.
    geometry = ((cfg['distance_zero']-distance)/(cfg['distance_zero']-cfg['distance_full'])).clamp(0,1)
    plant = x.sole_plant.clamp(0,1)
    planted = _gate(plant, cfg['plant_start'], cfg['plant_full'])
    touch = _gate(x.usable_load, cfg['touch_start'], cfg['touch_full'])
    contact = x.persistent_contact.to(plant.dtype)
    feet = .5*_both_score(geometry) + .3*_both_score(plant) + .2*_both_score(touch*contact)
    discovery_diagnostics = {}
    if cfg.get('foot_discovery_v2',False):
        if x.foot_low is None or x.foot_flat is None:
            raise ValueError('foot_discovery_v2 requires per-foot height and signed orientation')
        low = _both_score(x.foot_low.clamp(0,1))
        flat = _both_score(x.foot_flat.clamp(0,1))
        feet = .4*_both_score(geometry) + .3*low + .2*flat + .1*_both_score(touch*contact)
        discovery_diagnostics = dict(foot_lowering_credit=low,foot_orientation_credit=flat)
    if cfg.get('foot_retraction_v3',False):
        # Equal marginal credit for either foot, including the first moving foot.
        # Planting only earns completion credit inside the approach region.
        retract = geometry.mean(-1)
        finish = (near*plant).mean(-1)
        feet = .85*retract + .15*finish
        discovery_diagnostics.update(foot_retraction_credit=retract,
            foot_placement_gate=near.mean(-1),foot_near_plant_credit=finish)
    if cfg.get('foot_ground_retraction_v4',False):
        if x.foot_clearance is None or x.foot_flat is None:
            raise ValueError('Ground retraction requires measured sole clearance and orientation')
        # A low ground-level approach region, not a vertical column above the hip.
        h=x.foot_clearance.clamp_min(0)
        approach_distance=torch.sqrt(distance.square()+(1.5*(h-.12).clamp_min(0)).square())
        retract=((cfg['distance_zero']-approach_distance)/(cfg['distance_zero']-cfg['distance_full'])).clamp(0,1)
        lower=(1-(h-.03).clamp_min(0)/.6).clamp(0,1)
        orient=x.foot_flat.clamp(0,1)
        finish=.7*orient+.3*plant
        feet=(.6*retract+.3*near*lower+.1*near*finish).mean(-1)
        discovery_diagnostics.update(foot_retraction_credit=retract.mean(-1),
            foot_approach_distance=approach_distance.mean(-1),
            foot_placement_gate=near.mean(-1),foot_lowering_credit=(near*lower).mean(-1),
            foot_near_plant_credit=(near*plant).mean(-1))
    placement_ready = torch.minimum(near, planted).amin(-1)*contact.all(-1)
    support_ready = torch.minimum(torch.minimum(near, planted),touch).amin(-1)*contact.all(-1)
    stance_diagnostics = {}
    if 'stance' in cfg:
        if x.sole_heading_xy is None:
            raise ValueError('Bilateral stance requires measured sole geometry')
        from .standup_stance import stance_geometry
        s = stance_geometry(x.sole_heading_xy,cfg['stance'])
        f = cfg['stance']['fraction']
        if cfg.get('foot_ground_retraction_v4',False):
            finish=(1-f)*finish+f*s['quality'][:,None]
            per_foot=.6*retract+.3*near*lower+.1*near*finish
            if cfg.get('foot_stance_v5',False):
                from .standup_stance import preparation_stance_factors
                side_factor,separation_factor=preparation_stance_factors(x.sole_heading_xy,s,cfg['stance'])
                per_foot=per_foot*side_factor*separation_factor[:,None]
                discovery_diagnostics.update(foot_own_side_factor=side_factor.mean(-1),
                    foot_separation_factor=separation_factor,
                    foot_valid_retraction_credit=(retract*side_factor*separation_factor[:,None]).mean(-1))
            feet=per_foot.mean(-1)
        elif cfg.get('foot_retraction_v3',False):
            # Stance geometry cannot reward a pair of distant, flat feet.
            finish = (near*((1-f)*plant+f*s['quality'][:,None])).mean(-1)
            feet = .85*retract + .15*finish
        else:
            feet = (1-f)*feet + f*s['quality']
        support_ready *= s['ready']
        placement_ready *= s['ready']
        stance_diagnostics = {f'stance_{k}':v for k,v in s.items()}
    ready = body_ready*support_ready

    height = torch.minimum(x.root_height/x.root_target,x.shoulder_height/x.shoulder_target).clamp(0,1)
    load = (x.filtered_load/cfg['load_full']).clamp(0,1)
    com = 1/(1+(x.com_distance.clamp_min(0)/cfg['com_sigma']).square())
    # No required full load or CoM alignment before lifting; improve together.
    rise = ready*(.2*load+.6*height+.2*com)
    transfer_diagnostics = {}
    if 'rise_transfer' in cfg:
        if hand_state is None or x.torso_forward_lean is None:
            raise ValueError('Rise transfer requires hand loads and signed torso lean')
        c = cfg['rise_transfer']
        # Filter memory alone cannot retain load credit when feet unload.
        if x.assistance_ratio is None or x.filtered_ground_fraction is None:
            raise ValueError('Rise transfer requires actual assistance and normalized load history')
        remaining = (1-x.assistance_ratio.clamp_min(0)).clamp_min(c['minimum_ground_weight'])
        ground_fraction = x.usable_load.clamp_min(0).sum(-1)/remaining
        # Filter normalized samples, not old raw forces divided by today's
        # smaller remaining weight: assistance onset must not magnify history.
        live_load = torch.minimum(x.filtered_ground_fraction, ground_fraction)
        load = (live_load/cfg['load_full']).clamp(0,1)
        hand_load = hand_state['load'].clamp_min(0).sum(-1)/remaining
        unload = hand_unload_quality(hand_load,c)
        takeover = load*unload
        angle = torch.rad2deg(x.torso_forward_lean)
        lean = _gate(angle,c['lean_start_deg'],c['lean_full_deg'])*(1-_gate(
            angle,c['lean_upper_deg'],c['lean_zero_deg']))
        # As hips rise, replace the temporary posture target with completion
        # credit. Removing a target must not penalize successful extension.
        fade = _gate(x.root_height/x.root_target,c['fade_start'],c['fade_full'])
        lean_credit = fade+(1-fade)*lean
        u,l = c['unload_fraction'],c['lean_fraction']
        budget_remaining = .4-u-l
        rise = ready*(.6*height + .5*budget_remaining*(load+com) + u*takeover + l*lean_credit)
        transfer_diagnostics = dict(torso_forward_lean_deg=angle, rise_load_credit=load,
            remaining_ground_weight=remaining, foot_load_target_mg=cfg['load_full']*remaining,
            foot_ground_fraction=ground_fraction, hand_ground_fraction=hand_load,
            hand_unload_quality=unload, foot_takeover_credit=ready*takeover,
            lean_quality=lean, lean_fade=fade, lean_guidance_credit=ready*lean_credit)
    standing = _gate(height,cfg['standing_start'],cfg['standing_full'])*_gate(
        x.upright,cfg['upright_start'],cfg['upright_full'])
    motion = 1/(1+(x.linear_speed/cfg['linear_speed_scale']).square()
        +(x.angular_speed/cfg['angular_speed_scale']).square()
        +(x.joint_speed/cfg['joint_speed_scale']).square())
    stand = ready*standing*load*motion
    fraction = cfg['preparation_righting_fraction']
    preparation = fraction*righting + (1-fraction)*feet
    hand_diagnostics = {}
    if cfg.get('hands', {}).get('enabled', False):
        if hand_state is None:
            raise ValueError('Hand rewards require measured hand support state')
        h = cfg['hands']
        # Either hand can help. Modest approach credit precedes real support.
        help_credit = .15*hand_state['approach'].amax(-1) + .85*hand_state['support'].amax(-1)*righting
        # Foot takeover preserves earned support credit when hands unload.
        support_credit = torch.maximum(help_credit, ready)
        preparation = (1-h['preparation_fraction'])*preparation + h['preparation_fraction']*support_credit
        release = hand_state['release'].amin(-1)
        stand *= (1-h['release_fraction']) + h['release_fraction']*release
        hand_diagnostics = dict(hand_help=help_credit, hand_support_credit=support_credit, hand_release=release)
    if cfg.get('rise_hold_v2',False):
        from .rise_hold import rise_hold_credit
        rise,stand,new_diagnostics=rise_hold_credit(
            x,cfg,ready,load,takeover,hand_state['load'].clamp_min(0).sum(-1))
        transfer_diagnostics.update(new_diagnostics)
    credits = dict(preparation=preparation, rise=rise, stand=stand)
    if reward_form not in ('deficit','positive'):
        raise ValueError('Unknown stage reward form')
    rewards = {k: step_dt*cfg['weights'][k]*(v if reward_form == 'positive' else v-1)
               for k,v in credits.items()}
    diagnostics = {f'credit_{k}':v for k,v in credits.items()}
    diagnostics.update(discovery_diagnostics)
    diagnostics.update(hand_diagnostics)
    diagnostics.update(transfer_diagnostics)
    diagnostics.update(stance_diagnostics)
    diagnostics.update(foot_distance_credit=_both_score(geometry), foot_plant_credit=_both_score(plant),
        credit_righting=righting, credit_feet=feet, body_ready=body_ready, support_ready=support_ready,
        placement_ready=placement_ready, loaded_rise=ready, com_quality=com,
        phase_preparation=1-ready,
        phase_rise=ready*(1-standing), phase_stand=ready*standing)
    if cfg.get('rise_hold_v2',False):
        diagnostics['phase_stand']=new_diagnostics['stand_posture_credit']
        diagnostics['phase_rise']=(new_diagnostics.get('rise_support_valid',ready)-diagnostics['phase_stand']).clamp_min(0)
    # Placement/contact, not large load, unlocks pull force: no load-force deadlock.
    return StageRewards(rewards,diagnostics,placement_ready)


def preparation_config(support):
    """Adapt shared contact/geometry measurements without importing old rewards.

    The existing support reader uses nonzero scales to enable measurements.
    These internal switches are never reward weights in the staged design.
    """
    return {**support['preparation'], 'foot_placement_scale':1., 'sole_plant_scale':1.,
        'bilateral_load_scale':0., 'com_scale':0., 'com_preparation_scale':0.,
        'bilateral_squat':True, 'placement_reference':'hips'}
