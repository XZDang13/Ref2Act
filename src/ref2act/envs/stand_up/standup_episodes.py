"""Per-environment deadlines and conservative no-progress sampling cutoffs."""
import math
import torch


def validate_episode_sampling(c, settling_s):
    if set(c)!={'enabled','initial_timeout_range_s','timeout_range_s','grace_s','stagnation_s','stagnation_jitter_s','progress_epsilon'} or type(c['enabled']) is not bool:
        raise ValueError('Invalid episode_sampling configuration')
    for key in ('initial_timeout_range_s','timeout_range_s'):
        v=c[key]
        if len(v)!=2 or not all(math.isfinite(float(x)) for x in v) or not settling_s < v[0] <= v[1]:
            raise ValueError(f'Invalid {key}')
    for key in ('grace_s','stagnation_s','progress_epsilon'):
        if not math.isfinite(float(c[key])) or c[key]<=0:raise ValueError(f'Invalid {key}')
    if not math.isfinite(float(c['stagnation_jitter_s'])) or not 0<=c['stagnation_jitter_s']<c['stagnation_s']:
        raise ValueError('Invalid stagnation jitter')
    if c['progress_epsilon']>=1:raise ValueError('Progress epsilon must be below 1')


class EpisodeSampling:
    def __init__(self, n, device, dt, cfg, feature_count=10):
        self.cfg=cfg;self.dt=dt
        self.deadline=torch.zeros(n,dtype=torch.long,device=device)
        self.initial=torch.ones(n,dtype=torch.bool,device=device)
        self.seen=torch.zeros_like(self.initial)
        self.idle=torch.zeros_like(self.deadline)
        self.idle_limit=torch.zeros_like(self.deadline)
        self.best=torch.zeros(n,feature_count,device=device)

    def reset(self, ids, *, initial=False):
        first=~self.seen[ids] | initial
        a,b=self.cfg['initial_timeout_range_s'];c,d=self.cfg['timeout_range_s']
        u=torch.rand(len(ids),device=self.deadline.device)
        duration=torch.where(first,a+(b-a)*u,c+(d-c)*u)
        self.deadline[ids]=torch.ceil(duration/self.dt).long()
        jitter=self.cfg['stagnation_jitter_s']
        window=self.cfg['stagnation_s']+(2*torch.rand(len(ids),device=self.deadline.device)-1)*jitter
        self.idle_limit[ids]=torch.ceil(window/self.dt).long()
        self.initial[ids]=first;self.seen[ids]=True
        self.idle[ids]=0;self.best[ids]=0

    def stagnated(self, features, active_age, active, stable):
        features=features.clamp(0,1)
        improved=features > self.best+self.cfg['progress_epsilon']
        self.best.copy_(torch.where(improved & active[:,None],features,self.best))
        grace=active_age*self.dt < self.cfg['grace_s']
        clear=improved.any(-1)|grace|~active|stable
        self.idle.copy_(torch.where(clear,0,self.idle+1))
        return active & ~stable & (self.idle >= self.idle_limit)
