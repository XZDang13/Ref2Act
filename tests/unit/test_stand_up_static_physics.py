from types import SimpleNamespace
import pytest
import torch
from ref2act.envs.stand_up.static_physics import StaticPhysicsCache


class View:
    def __init__(self):
        self.masses = torch.ones(3,2)
        self.coms = torch.zeros(3,2,7)
        self.mass_reads = self.com_reads = 0
    def get_masses(self):
        self.mass_reads += 1
        return self.masses.clone()
    def get_coms(self):
        self.com_reads += 1
        return self.coms.clone()
    def set_masses(self, value, indices):
        self.masses[indices] = value[indices]
    def set_coms(self, value, indices):
        self.coms[indices] = value[indices]
    def set_inertias(self, value, indices):
        if value is None:
            raise ValueError('invalid inertia')


def test_reads_cached_and_partial_writes_invalidate():
    view=View();invalidations=[]
    cache=StaticPhysicsCache(view,on_invalidate=lambda:invalidations.append(1))
    a=view.get_masses();b=view.get_coms()
    assert view.get_masses() is a and view.get_coms() is b
    assert view.mass_reads==view.com_reads==1
    new_mass=a.clone();new_mass[1]=3
    view.set_masses(new_mass,indices=[1])
    torch.testing.assert_close(view.get_masses(),new_mass,rtol=0,atol=0)
    assert view.get_coms() is not b
    new_com=b.clone();new_com[2,:,0]=.02
    view.set_coms(new_com,indices=[2])
    torch.testing.assert_close(view.get_coms(),new_com,rtol=0,atol=0)
    assert len(invalidations)==2
    with pytest.raises(ValueError): view.set_inertias(None,indices=[0])
    assert cache.values=={} and len(invalidations)==3
    cache.close();cache.close()
    assert not hasattr(view,'_standup_static_cache')
    assert view.get_masses() is not view.get_masses()


def test_switch_bypass_and_external_edit_refresh():
    view=View();cache=StaticPhysicsCache(view)
    view.get_masses()
    cache.enabled=False
    assert view.get_masses() is not view.get_masses()
    view.masses[0]=7
    cache.enabled=True
    torch.testing.assert_close(view.get_masses(),view.masses)
    view.masses[1]=9
    cache.invalidate()
    torch.testing.assert_close(view.get_masses(),view.masses)
    with pytest.raises(RuntimeError): StaticPhysicsCache(view)
    cache.close()


def test_warp_type_is_preserved():
    import warp as wp
    view=View()
    source=wp.ones((3,2),dtype=wp.float32,device='cpu')
    view.get_masses=lambda:source
    cache=StaticPhysicsCache(view)
    cached=view.get_masses()
    assert isinstance(cached,wp.array) and cached is not source
    torch.testing.assert_close(wp.to_torch(cached),wp.to_torch(source))
    cache.close()
