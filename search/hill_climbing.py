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
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """
    crash_count = 0
    min_distance = float("inf")

    for frame in time_series:
        # If a crash occurs in any frame, mark the scenario as crashing
        crash_count |= int(frame["crashed"])

        # Get ego vehicle position
        ego_pos = np.array(frame["ego"]["pos"])

        # Compute distance to every other vehicle in the frame
        for other in frame["others"]:
            other_pos = np.array(other["pos"])
            distance = np.linalg.norm(ego_pos - other_pos)
            # Keep the smallest distance observed so far
            min_distance = min(min_distance, distance)

    return {
        "crash_count": crash_count,
        "min_distance": float(min_distance),
    }


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """
    if objectives["crash_count"] == 1:
        return -1.0
    return objectives["min_distance"]


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """
    new_cfg = copy.deepcopy(cfg)

    # === Multi-Strategy Mutation ===
    # We use three strategies with different exploration/exploitation tradeoffs:
    #   1. Single-param fine-tune (40%): small local perturbation
    #   2. Multi-param exploration (35%): mutate several params at once
    #   3. Danger-biased mutation (25%): bias toward crash-prone configurations

    strategy = rng.random()

    if strategy < 0.40:
        # --- Strategy 1: Single-parameter fine-tune ---
        # Pick one random parameter and apply a small perturbation
        param = rng.choice(list(param_spec.keys()))
        spec = param_spec[param]

        if spec["type"] == "int":
            # Small integer step: ±1 or ±2
            delta = rng.choice([-2, -1, 1, 2])
            new_val = cfg.get(param, spec["min"]) + delta
            new_cfg[param] = int(np.clip(new_val, spec["min"], spec["max"]))
        else:
            # Gaussian perturbation with 10% of range as sigma
            sigma = (spec["max"] - spec["min"]) * 0.10
            new_val = cfg.get(param, spec["min"]) + rng.normal(0, sigma)
            new_cfg[param] = float(np.clip(new_val, spec["min"], spec["max"]))

    elif strategy < 0.75:
        # --- Strategy 2: Multi-parameter exploration ---
        # Each parameter has 40% chance of being mutated
        mutation_prob = 0.40
        mutated_any = False

        for param, spec in param_spec.items():
            if rng.random() < mutation_prob:
                mutated_any = True
                if spec["type"] == "int":
                    delta = rng.integers(-3, 4)  # [-3, 3]
                    new_val = cfg.get(param, spec["min"]) + delta
                    new_cfg[param] = int(np.clip(new_val, spec["min"], spec["max"]))
                else:
                    sigma = (spec["max"] - spec["min"]) * 0.15
                    new_val = cfg.get(param, spec["min"]) + rng.normal(0, sigma)
                    new_cfg[param] = float(np.clip(new_val, spec["min"], spec["max"]))

        # Guarantee at least one mutation
        if not mutated_any:
            param = rng.choice(list(param_spec.keys()))
            spec = param_spec[param]
            if spec["type"] == "int":
                new_cfg[param] = int(rng.integers(spec["min"], spec["max"] + 1))
            else:
                new_cfg[param] = float(rng.uniform(spec["min"], spec["max"]))

    else:
        # --- Strategy 3: Danger-biased mutation ---
        # Domain knowledge: crashes are more likely with:
        #   - More vehicles (higher density)
        #   - Smaller spacing (less reaction time)
        #   - Middle lanes (more neighbors)

        param = rng.choice(list(param_spec.keys()))
        spec = param_spec[param]
        current_val = cfg.get(param, spec["min"])

        if param == "vehicles_count":
            # Bias toward more vehicles (70% increase, 30% decrease)
            if rng.random() < 0.70:
                delta = rng.integers(1, 8)
            else:
                delta = -rng.integers(1, 4)
            new_cfg[param] = int(np.clip(current_val + delta, spec["min"], spec["max"]))

        elif param in ["initial_spacing", "ego_spacing"]:
            # Bias toward smaller spacing (tighter traffic)
            if rng.random() < 0.70:
                # Decrease spacing
                delta = -abs(rng.normal(0, 0.4))
            else:
                delta = rng.normal(0, 0.3)
            new_cfg[param] = float(np.clip(current_val + delta, spec["min"], spec["max"]))

        elif param == "lanes_count":
            # More lanes = more traffic variety, slight bias upward
            delta = rng.choice([-1, 1, 1, 2])
            new_cfg[param] = int(np.clip(current_val + delta, spec["min"], spec["max"]))

        elif param == "initial_lane_id":
            # Bias toward middle lanes (more neighbors = more danger)
            lanes = new_cfg.get("lanes_count", cfg.get("lanes_count", 3))
            middle = (lanes - 1) / 2
            # Move toward middle with 60% prob
            if rng.random() < 0.60 and current_val != int(middle):
                new_cfg[param] = int(current_val + np.sign(middle - current_val))
            else:
                new_cfg[param] = int(np.clip(current_val + rng.choice([-1, 1]), 0, lanes - 1))

    # === Constraint Repair ===
    # Ensure initial_lane_id is valid for the current lanes_count
    lanes = new_cfg.get("lanes_count", cfg.get("lanes_count", 3))
    if "initial_lane_id" in new_cfg:
        new_cfg["initial_lane_id"] = min(new_cfg["initial_lane_id"], lanes - 1)
        new_cfg["initial_lane_id"] = max(new_cfg["initial_lane_id"], 0)

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
        # Use the same seed for all evaluations for fair comparison
        crashed, nbr_ts = run_episode(env_id, neighbor, policy, defaults, seed_base)
        nbr_obj = compute_objectives_from_time_series(nbr_ts)
        nbr_fit = compute_fitness(nbr_obj)
        
        if nbr_fit < best_neighbor_fit:
            best_neighbor_fit = nbr_fit
            best_neighbor_cfg = neighbor
            best_neighbor_obj = nbr_obj
    
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
    
    # TODO (students): choose initialization (base_cfg or random scenario)
    current_cfg = dict(base_cfg)

    # Evaluate initial solution (seed_base used for reproducibility)
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    best_cfg = copy.deepcopy(current_cfg)
    best_obj = dict(obj)
    best_fit = float(cur_fit)
    best_seed_base = seed_base

    print(f"Initial fitness: {best_fit:.4f} (crash={best_obj['crash_count']}, min_dist={best_obj['min_distance']:.2f})")
    
    history = [best_fit]

    for i in trange(iterations, desc="Hill Climbing"):
        if best_obj["crash_count"] == 1:
            print(f"\n💥 Crash found at iteration {i}! Stopping early.")
            break
        
        # Get a neighbor
        nbr_cfg, nbr_obj, nbr_fit, nbr_seed = generate_and_select_best_neighbor(
            current_cfg, param_spec, rng, neighbors_per_iter,
            env_id, policy, defaults, seed_base
        )
        
        if nbr_fit < cur_fit:
            current_cfg = nbr_cfg
            cur_fit = nbr_fit
            
            if nbr_fit < best_fit:
                best_cfg = nbr_cfg
                best_obj = nbr_obj
                best_fit = nbr_fit
                best_seed_base = nbr_seed
                print(f"\n✨ Improved! Iter {i}: fitness={best_fit:.4f} (crash={best_obj['crash_count']}, min_dist={best_obj['min_distance']:.2f})")
        
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