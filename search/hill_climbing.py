"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from tqdm import trange

from envs.highway_env_utils import run_episode


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Track crash, min distance, average closest distance, and some metrics to reward chaos.
    """
    crash_count = 0
    min_distance = float("inf")
    closest_distances = []  # Track closest car distance at each frame
    speeds = []
    lane_changes = 0
    prev_lane = None
    prev_pos = None
    position_deltas = []

    for frame in time_series:
        crash_count |= int(frame["crashed"])
        
        ego = frame.get("ego")
        if ego is None:
            continue
        
        ego_pos = np.array(ego["pos"])
        ego_speed = ego.get("speed", 0)
        ego_lane = ego.get("lane_id", None)
        
        speeds.append(ego_speed)
        
        # Track lane changes
        if prev_lane is not None and ego_lane is not None and ego_lane != prev_lane:
            lane_changes += 1
        prev_lane = ego_lane
        
        # Track position changes (to measure crazy movements)
        if prev_pos is not None:
            delta = np.linalg.norm(ego_pos - prev_pos)
            position_deltas.append(delta)
        prev_pos = ego_pos
        
        # Track min distance and closest at this frame
        frame_min_dist = float("inf")
        for other in frame["others"]:
            other_pos = np.array(other["pos"])
            distance = np.linalg.norm(ego_pos - other_pos)
            min_distance = min(min_distance, distance)
            frame_min_dist = min(frame_min_dist, distance)
        
        if frame_min_dist != float("inf"):
            closest_distances.append(frame_min_dist)

    # Compute variance and averages (indicators for chaotic behavior)
    speed_variance = float(np.var(speeds)) if speeds else 0.0
    avg_position_change = float(np.mean(position_deltas)) if position_deltas else 0.0
    avg_closest_distance = float(np.mean(closest_distances)) if closest_distances else 100.0

    return {
        "crash_count": crash_count,
        "min_distance": float(min_distance),
        "avg_closest_distance": avg_closest_distance,
        "speed_variance": speed_variance,
        "lane_changes": lane_changes,
        "avg_position_change": avg_position_change,
    }


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Fitness based on crash patterns I have seen in known cases:
    
    The car seems to crash often during lane changes, it bases its next moves on a local environment
    During the switch, it is too late to steer back in case of any other car being in its path
    The fitness below tries to reward such behaviors.
    
    We clip some metrics and add them with different weights, this is done through iterative tuning
    The current setup seems to work quite well, well enough so to say.
    """
    if objectives["crash_count"] == 1:
        return -1.0
    
    # Clip abnormal values to focus on crash-like scenarios
    speed_var = objectives.get("speed_variance", 0)
    lane_changes = objectives.get("lane_changes", 0)
    avg_pos = min(objectives.get("avg_position_change", 0), 8.0)
    avg_closest = objectives.get("avg_closest_distance", 100)
    
    # Target low lane changes (crashes have 4-12, not 40-60)
    # Penalize both too many and too few
    lane_penalty = abs(lane_changes - 8) * 1.0  # Target ~8 lane changes, seems to be a sweet spot for crashes
    
    # Minimize the following fitness score, in this way we maximize chaos in the runs
    chaos_score = (
        -speed_var * 2.0               # High speeds
        - avg_pos * 5.0                # Chaotic movements
        + avg_closest * 6.0            # Close proximity to other cars
        + lane_penalty                 # Penalization factor for lane changes
    )
    
    # We do not use min_distance, it does not seems to have much correlation with crashes in practice
    # Noticed that it's often around 4-7 for crashes, but the other metrics capture in a more sophisticated way
    
    return chaos_score


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Aggressive mutation - often changes multiple parameters to escape local optima.
    """
    new_cfg = copy.deepcopy(cfg)
    
    # 60% chance to mutate multiple random parameters (more aggressive exploration)
    if rng.random() < 0.60:
        num_params = rng.choice([2, 3])  # Change 2-3 params
        params_to_mutate = rng.choice(list(param_spec.keys()), size=num_params, replace=False)
    else:
        # 40% chance to mutate a single random parameter
        params_to_mutate = [rng.choice(list(param_spec.keys()))]
    
    for param in params_to_mutate:
        spec = param_spec[param]
        cur = cfg.get(param, (spec["min"] + spec["max"]) / 2)
        
        if param == "vehicles_count":
            # Wide range: -10 to +10
            delta = rng.integers(-10, 11)
            new_cfg[param] = int(np.clip(cur + delta, spec["min"], spec["max"]))
        
        elif param in ["initial_spacing", "ego_spacing"]:
            # Wide range for spacing
            delta = rng.uniform(-1.5, 1.5)
            new_cfg[param] = float(np.clip(cur + delta, spec["min"], spec["max"]))
        
        elif param == "lanes_count":
            # Wide range
            delta = rng.choice([-3, -2, -1, 0, 1, 2, 3])
            new_cfg[param] = int(np.clip(cur + delta, spec["min"], spec["max"]))
        
        elif param == "initial_lane_id":
            lanes = new_cfg.get("lanes_count", cfg.get("lanes_count", 4))
            new_cfg[param] = int(rng.integers(0, lanes))
        
        # All other integer and float variables
        else:
            if spec["type"] == "int":
                new_cfg[param] = int(np.clip(cur + rng.choice([-3, -2, -1, 0, 1, 2, 3]), spec["min"], spec["max"]))
            else:
                new_cfg[param] = float(np.clip(cur + rng.uniform(-1.0, 1.0), spec["min"], spec["max"]))
    
    # Fix lane_id constraint
    lanes = new_cfg.get("lanes_count", cfg.get("lanes_count", 4))
    new_cfg["initial_lane_id"] = int(np.clip(new_cfg.get("initial_lane_id", 0), 0, lanes - 1))

    return new_cfg


# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def generate_and_select_best_neighbor(
    current_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator,
    neighbors_per_iter: int,
    env_id: str,
    policy,
    defaults: Dict[str, Any],
    seed_base: int
) -> Tuple[Dict[str, Any], Dict[str, Any], float, int]:
    """
    Generate neighbors and select the best one.
    
    Returns:
        (best_neighbor_cfg, best_neighbor_obj, best_neighbor_fit, best_neighbor_seed)
    """
    best_neighbor_fit = float("inf")
    best_neighbor_cfg = None
    best_neighbor_obj = None
    best_neighbor_seed = seed_base
    
    for _ in range(neighbors_per_iter):
        neighbor = mutate_config(current_cfg, param_spec, rng)
        # Use varying seeds to explore more crash scenarios
        # Note: this is of course deterministic when setting seed in the main function
        nbr_seed = int(rng.integers(1e9))
        crashed, nbr_ts = run_episode(env_id, neighbor, policy, defaults, nbr_seed)
        nbr_obj = compute_objectives_from_time_series(nbr_ts)
        if crashed:
            nbr_obj["crash_count"] = 1
        nbr_fit = compute_fitness(nbr_obj)
        
        if nbr_fit < best_neighbor_fit:
            best_neighbor_fit = nbr_fit
            best_neighbor_cfg = neighbor
            best_neighbor_obj = nbr_obj
            best_neighbor_seed = nbr_seed
    
    return best_neighbor_cfg, best_neighbor_obj, best_neighbor_fit, best_neighbor_seed


def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    rng = np.random.default_rng(seed)

    print(f"Running Hill Climbing for {iterations} iterations...")
    
    # Initialize the config - bias toward crash-prone zones based on observed crashes
    current_cfg = dict(base_cfg)
    if "vehicles_count" not in current_cfg:
        # Crashes found: 32-42 vehicles
        current_cfg["vehicles_count"] = rng.integers(30, 45)
    if "initial_spacing" not in current_cfg:
        # Crashes found: 2.46-4.1 spacing (wider than expected!)
        current_cfg["initial_spacing"] = rng.uniform(2.0, 4.5)
    if "ego_spacing" not in current_cfg:
        # Crashes found: 2.04-2.5
        current_cfg["ego_spacing"] = rng.uniform(1.8, 3.0)
    if "lanes_count" not in current_cfg:
        # Crashes found: 3-6 lanes (not just high)
        current_cfg["lanes_count"] = rng.integers(3, 7)
    if "initial_lane_id" not in current_cfg:
        # Crashes found: lane_id=2 (middle lane) - bias toward middle
        lanes = current_cfg["lanes_count"]
        current_cfg["initial_lane_id"] = rng.integers(max(1, lanes//2 - 1), min(lanes-1, lanes//2 + 2))

    # Evaluate initial solution
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    if crashed:
        obj["crash_count"] = 1
    cur_fit = compute_fitness(obj)

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base

    print(f"Initial fitness: {best_fit:.4f} (crash={best_obj['crash_count']}, min_dist={best_obj['min_distance']:.2f})")
    print(f"  DEBUG - speed_var={best_obj.get('speed_variance', 0):.4f}, lane_changes={best_obj.get('lane_changes', 0)}, avg_pos_change={best_obj.get('avg_position_change', 0):.4f}, avg_closest={best_obj.get('avg_closest_distance', 0):.2f}")
    
    history = [best_fit]
    iterations_without_improvement = 0
    restart_fitness = cur_fit  # Track fitness at the last restart for local optimization

    for i in trange(iterations, desc="Hill Climbing"):
        if best_obj["crash_count"] == 1:
            print(f"\n💥 Crash found at iteration {i}! Stopping early.")
            break
        
        # Restart if we get stuck (do not improve fitness) for 5 iterations - jump to random config
        if iterations_without_improvement >= 5:
            print(f"\n🔄 RESTART at iter {i} - jumping to a random config!")
            current_cfg = {param: rng.uniform(spec["min"], spec["max"]) if spec["type"] == "float" 
                          else rng.integers(spec["min"], spec["max"]+1)
                          for param, spec in param_spec.items()}
            current_cfg["initial_lane_id"] = int(np.clip(current_cfg["initial_lane_id"], 0, current_cfg["lanes_count"]-1))
            
            seed_base = int(rng.integers(1e9))
            crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
            obj = compute_objectives_from_time_series(ts)
            if crashed:
                obj["crash_count"] = 1
            cur_fit = compute_fitness(obj)
            restart_fitness = cur_fit  # Reset restart baseline
            iterations_without_improvement = 0
        
        # Generate and evaluate neighbors; keep the best
        nbr_cfg, nbr_obj, nbr_fit, nbr_seed = generate_and_select_best_neighbor(
            current_cfg, param_spec, rng, neighbors_per_iter,
            env_id, policy, defaults, seed_base
        )
        
        # Accept if better than current
        if nbr_fit < cur_fit:
            current_cfg = nbr_cfg
            cur_fit = nbr_fit
            
            # Check if we improved from restart point
            if nbr_fit < restart_fitness:
                iterations_without_improvement = 0  # Reset only if better than restart
            
            if nbr_fit < best_fit:
                best_cfg = nbr_cfg
                best_obj = nbr_obj
                best_fit = nbr_fit
                best_seed_base = nbr_seed
                print(f"\n✨ Improved! Iter {i}: fitness={best_fit:.4f} (crash={best_obj['crash_count']}, min_dist={best_obj['min_distance']:.2f})")
                print(f"  DEBUG - speed_var={best_obj.get('speed_variance', 0):.4f}, lane_changes={best_obj.get('lane_changes', 0)}, avg_pos_change={best_obj.get('avg_position_change', 0):.4f}, avg_closest={best_obj.get('avg_closest_distance', 0):.2f}")
        else:
            iterations_without_improvement += 1
        
        history.append(best_fit)
    
    print(f"\n✅ Hill Climbing Complete!")
    print(f"   Final best fitness: {best_fit:.4f}")
    print(f"   Crash found: {best_obj['crash_count'] == 1}")
    print(f"   Min distance: {best_obj['min_distance']:.2f}")
    
    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": best_seed_base,
        "history": history
    }