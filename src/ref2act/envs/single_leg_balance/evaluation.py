"""Robustness summaries; failures before a scheduled push remain failures."""
import math

DEFAULT_STRENGTHS=(0.,.1,.2,.3,.4,.5)
DEFAULT_DIRECTIONS=tuple(i*math.pi/4 for i in range(8))


def summarize(rows,threshold=.8,strengths=DEFAULT_STRENGTHS,directions=DEFAULT_DIRECTIONS):
    if not 0<threshold<=1:raise ValueError('Survival threshold must be in (0,1]')
    if not rows:raise ValueError('No completed evaluation episodes')
    cells=[]
    for dv in strengths:
        for theta in directions:
            selected=[r for r in rows if abs(r['delta_v']-dv)<1e-5 and abs(r['direction']-theta)<1e-5]
            if not selected:raise ValueError(f'Missing evaluation cell {dv}, {theta}')
            n=len(selected);recovered=[r['recovery_time_s'] for r in selected if r['recovery_time_s']>=0]
            cells.append(dict(delta_v=dv,direction=theta,episodes=n,
                              survival_rate=sum(r['survived'] for r in selected)/n,
                              # Time-to-fall is right censored for surviving episodes.
                              restricted_mean_survival_s=sum(r['duration_s'] for r in selected)/n,
                              touchdown_episode_rate=sum(r['touchdowns']>0 for r in selected)/n,
                              recovered_fraction=len(recovered)/n,
                              mean_recovery_time_s=sum(recovered)/len(recovered) if recovered else None))
    curve=[];maximum=None;passing=True
    for dv in sorted(strengths):
        group=[c for c in cells if c['delta_v']==dv]
        worst=min(c['survival_rate'] for c in group)
        # Report the contiguous tested range; isolated successes at larger pushes
        # do not enlarge the recoverable range after a lower strength has failed.
        passing=passing and worst>=threshold
        if passing:maximum=dv
        curve.append(dict(delta_v=dv,mean_direction_survival_rate=sum(c['survival_rate'] for c in group)/len(group),
                          worst_direction_survival_rate=worst))
    return dict(cells=cells,curve=curve,survival_threshold=threshold,
                maximum_recoverable_delta_v_m_s=maximum,
                maximum_definition='Largest contiguous tested strength passing the threshold in every direction; null if zero fails. Delta-v is impulse divided by robot mass.')
