"""
SimWorld smoke test (Python client side).

This script verifies:
- SimWorld can be imported
- UnrealCV client can *attempt* to connect (with a timeout)
- Basic API calls are wired (spawn humanoid, fetch one camera frame) once UE server is running

Note: You must run the SimWorld UE server executable separately.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim != 3 or img_bgr.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={getattr(img_bgr, 'shape', None)}")
    return img_bgr[:, :, ::-1].copy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=os.environ.get("SIMWORLD_IP", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIMWORLD_PORT", "9000")))
    parser.add_argument("--timeout_s", type=float, default=float(os.environ.get("SIMWORLD_CONNECT_TIMEOUT_S", "5")))
    parser.add_argument("--out", default="benchmarks/simworld/out_smoke_rgb.npy")
    args = parser.parse_args()

    # Import locally so that "pip install" / editable install issues surface here.
    from simworld.communicator.communicator import Communicator
    from simworld.communicator.unrealcv import UnrealCV
    from simworld.agent.humanoid import Humanoid
    from simworld.utils.vector import Vector

    try:
        ucv = UnrealCV(ip=args.ip, port=args.port, resolution=(640, 360), connect_timeout_s=args.timeout_s)
    except TimeoutError as e:
        print(f"[smoke_test] Could not connect to UE UnrealCV server: {e}")
        print("[smoke_test] This is expected if the UE server executable is not running yet.")
        return 2

    comm = Communicator(ucv)

    # (Re)spawn a humanoid at origin; camera id is assigned by the Humanoid class.
    agent = Humanoid(position=Vector(0, 0), direction=Vector(1, 0))
    comm.spawn_agent(agent, name=None, model_path="/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C", type="humanoid")

    img_bgr = comm.get_camera_observation(agent.camera_id, "lit", mode="direct")
    img_rgb = _bgr_to_rgb(img_bgr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array (for programmatic use)
    np.save(out_path, img_rgb)
    
    # Save as PNG image (for human viewing)
    png_path = out_path.with_suffix('.png')
    Image.fromarray(img_rgb.astype(np.uint8)).save(png_path)
    
    print(f"[smoke_test] OK. Saved RGB frame:")
    print(f"  - NumPy: {out_path} (shape={img_rgb.shape}, dtype={img_rgb.dtype})")
    print(f"  - PNG:   {png_path} (viewable image)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


