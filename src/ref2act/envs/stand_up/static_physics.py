"""Environment-local PhysX parameter cache with write-through invalidation.

Only model parameters are cached. World poses, velocities, contact forces and
simulation timestamps retain their normal Isaac Lab update cadence.
"""
from functools import wraps


class StaticPhysicsCache:
    """Cache a single articulation view, preserving its native Torch/Warp types.

    Installed on the view instance so articulation data and wrench composers
    share the cache. All public mass/COM/inertia writers route through these
    three PhysX setters. Direct USD/backend edits must call ``invalidate``;
    changing the articulation view requires closing and installing a new cache.
    Returned buffers follow the backend's borrowed-buffer convention: callers
    modifying model parameters must finish with the matching setter.
    """
    GETTERS = ('get_masses', 'get_coms')
    SETTERS = ('set_masses', 'set_coms', 'set_inertias')

    def __init__(self, view, *, on_invalidate=None, enabled=True):
        if getattr(view, '_standup_static_cache', None) is not None:
            raise RuntimeError('Articulation already has a static parameter cache')
        self.view = view
        self.on_invalidate = on_invalidate
        self._enabled = bool(enabled)
        self.values = {}
        self.originals = {name: getattr(view, name) for name in self.GETTERS+self.SETTERS}
        self.closed = False
        for name in self.GETTERS:
            setattr(view, name, self._getter(name))
        for name in self.SETTERS:
            setattr(view, name, self._setter(name))
        view._standup_static_cache = self

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        if bool(value) != self._enabled:
            self.invalidate()
            self._enabled = bool(value)

    def _getter(self, name):
        original = self.originals[name]
        @wraps(original)
        def read(*args, **kwargs):
            if not self.enabled or args or kwargs:
                return original(*args, **kwargs)
            if name not in self.values:
                value = original()
                # PhysX reuses getter buffers; own a snapshot of the parameters.
                import torch
                if isinstance(value, torch.Tensor):
                    value = value.clone()
                else:
                    import warp as wp
                    value = wp.clone(value)
                self.values[name] = value
            return self.values[name]
        return read

    def _setter(self, name):
        original = self.originals[name]
        @wraps(original)
        def write(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            finally:
                # Invalidate even after a failed/partially applied write.
                self.invalidate()
        return write

    def invalidate(self):
        self.values.clear()
        if self.on_invalidate is not None:
            self.on_invalidate()

    def close(self):
        if self.closed:
            return
        self.invalidate()
        for name, original in self.originals.items():
            setattr(self.view, name, original)
        del self.view._standup_static_cache
        self.closed = True
