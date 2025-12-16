#!/usr/bin/env python3
"""
Curate a small Habitat R2R VLN episode split (gzipped JSON) from an existing split.

Input format (Habitat VLN R2R):
{
  "episodes": [ { ... VLNEpisode fields ... } ],
  "instruction_vocab": { "word_list": [...] }
}

This script selects:
- N unique scenes (scans)
- For each selected scene, one trajectory_id that has at least K unique instructions
- Exactly K episodes for that (scene, trajectory_id), with unique instruction_text

Example:
python scripts/dataset_converters/curate_r2r_episode_split.py \
  --input data/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz \
  --output data/vln_ce/raw_data/r2r/test/test_scene8_instr3.json.gz \
  --num-scenes 8 \
  --num-instr-per-scene 3
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _scan_from_scene_id(scene_id: str) -> str:
    # Typical: "mp3d/<scan>/<scan>.glb"
    parts = scene_id.split("/")
    if len(parts) >= 2:
        return parts[1]
    return scene_id


def _load_gz_json(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _write_gz_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def curate(
    source: Dict[str, Any],
    *,
    num_scenes: int,
    num_instr_per_scene: int,
    seed: int,
    scene_pick: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = source["episodes"]

    # Group: scan -> trajectory_id -> episodes
    by_scan_traj: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ep in episodes:
        scan = _scan_from_scene_id(ep["scene_id"])
        by_scan_traj[scan][str(ep["trajectory_id"])].append(ep)

    scan_counts = {scan: sum(len(v) for v in trajs.values()) for scan, trajs in by_scan_traj.items()}

    scans = list(by_scan_traj.keys())
    if scene_pick == "largest":
        scans.sort(key=lambda s: (-scan_counts[s], s))
    elif scene_pick == "random":
        rng = random.Random(seed)
        rng.shuffle(scans)
    else:
        raise ValueError(f"Unknown --scene-pick={scene_pick}")

    selected_scans = scans[:num_scenes]
    if len(selected_scans) < num_scenes:
        raise ValueError(
            f"Requested {num_scenes} scenes, but only found {len(selected_scans)}"
        )

    curated_episodes: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {"seed": seed, "scene_pick": scene_pick, "scenes": []}

    for scan in selected_scans:
        trajs = by_scan_traj[scan]

        # Pick a trajectory that has >=K unique instructions.
        # Deterministic order: by trajectory_id (numeric if possible).
        def _traj_sort_key(tid: str):
            try:
                return (0, int(tid))
            except Exception:
                return (1, tid)

        chosen_traj = None
        chosen_eps: List[Dict[str, Any]] = []

        for traj_id in sorted(trajs.keys(), key=_traj_sort_key):
            eps = trajs[traj_id]
            # Keep stable order: episode_id
            eps_sorted = sorted(eps, key=lambda e: int(e["episode_id"]) if str(e["episode_id"]).isdigit() else str(e["episode_id"]))
            seen_text = set()
            picked = []
            for ep in eps_sorted:
                text = ep.get("instruction", {}).get("instruction_text", "")
                if not text or text in seen_text:
                    continue
                seen_text.add(text)
                picked.append(ep)
                if len(picked) >= num_instr_per_scene:
                    break
            if len(picked) >= num_instr_per_scene:
                chosen_traj = traj_id
                chosen_eps = picked
                break

        if chosen_traj is None:
            raise ValueError(
                f"Scan {scan} has no trajectory with {num_instr_per_scene} unique instructions"
            )

        curated_episodes.extend(chosen_eps)
        manifest["scenes"].append(
            {
                "scan": scan,
                "trajectory_id": chosen_traj,
                "num_selected": len(chosen_eps),
                "episode_ids": [ep["episode_id"] for ep in chosen_eps],
            }
        )

    curated = {
        "episodes": curated_episodes,
        # Keep the original vocab so instruction_tokens remain valid.
        "instruction_vocab": source.get("instruction_vocab", {}),
    }
    return curated, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to source split .json.gz")
    ap.add_argument("--output", required=True, help="Path to output curated .json.gz")
    ap.add_argument("--num-scenes", type=int, default=8)
    ap.add_argument("--num-instr-per-scene", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--scene-pick",
        choices=["largest", "random"],
        default="largest",
        help="How to choose scenes from the source split",
    )
    args = ap.parse_args()

    source = _load_gz_json(args.input)
    if "episodes" not in source:
        raise ValueError("Input JSON must contain key 'episodes'")

    curated, manifest = curate(
        source,
        num_scenes=args.num_scenes,
        num_instr_per_scene=args.num_instr_per_scene,
        seed=args.seed,
        scene_pick=args.scene_pick,
    )
    _write_gz_json(args.output, curated)

    print(f"Wrote: {args.output}")
    print(f"Episodes: {len(curated['episodes'])} (expected {args.num_scenes * args.num_instr_per_scene})")
    print("Selected scenes:")
    for s in manifest["scenes"]:
        print(f"  - {s['scan']} traj={s['trajectory_id']} episodes={s['episode_ids']}")


if __name__ == "__main__":
    main()


