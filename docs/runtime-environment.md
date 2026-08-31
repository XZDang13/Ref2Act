# Runtime Environment

## Verified stack

Ref2Act is installed into the existing `isaaclab` environment:

```text
Python       3.12.13
Isaac Lab    3.0.0b2
Isaac Sim    6.0.0.1
PyTorch      2.11.0+cu128
torchvision  0.26.0+cu128
Ref2Act      0.2.1 (editable)
```

## Reproduce

```bash
conda activate isaaclab
python -m pip install --no-deps -e /home/xdang/Desktop/Ref2Act
```

## NVIDIA EULA

Isaac Sim checks the NVIDIA Omniverse EULA on first use. Accept it interactively before unattended runs. After acceptance, automated commands may pass `OMNI_KIT_ACCEPT_EULA=YES`.

## Headless launch

The verified unattended headless launch form on this workstation is:

```bash
env -u DISPLAY -u XAUTHORITY \
  OMNI_KIT_ACCEPT_EULA=YES \
  python tests/integration/isaac_locomotion_headless_smoke.py
```

This workaround is only for headless jobs. GUI launches should retain `DISPLAY` and `XAUTHORITY`.

## Verification

```bash
python -m pip check
pytest -q tests/unit tests/integration/test_g1_mjcf.py tests/integration/test_g1_usd.py
env -u DISPLAY -u XAUTHORITY \
  OMNI_KIT_ACCEPT_EULA=YES \
  python tests/integration/isaac_locomotion_headless_smoke.py --num-envs 2 --steps 10
```
