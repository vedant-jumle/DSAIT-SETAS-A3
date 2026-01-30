"""
Quick script to run Random Search with a specific seed
Usage: python run_random_search.py [seed] [n_scenarios]
"""

import sys
import time
import json
from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch

def main():
    # Default values
    seed = 42
    n_scenarios = 50

    # Parse command line arguments
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_scenarios = int(sys.argv[2])

    print(f"{'='*60}")
    print(f"RANDOM SEARCH - SEED {seed}, n_scenarios={n_scenarios}")
    print(f"{'='*60}")

    # Setup
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)

    start = time.time()
    crashes = search.run_search(n_scenarios=n_scenarios, n_eval=1, seed=seed)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Seed: {seed}")
    print(f"Scenarios evaluated: {n_scenarios}")
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Crashes found: {len(crashes)}")

    if crashes:
        print(f"\nCrash configurations:")
        for i, crash in enumerate(crashes):
            cfg = crash['cfg']
            print(f"\n  Crash {i+1} (scenario seed={crash['seed']}):")
            print(f"    vehicles_count: {cfg.get('vehicles_count')}")
            print(f"    lanes_count: {cfg.get('lanes_count')}")
            print(f"    initial_spacing: {cfg.get('initial_spacing'):.2f}")
            print(f"    ego_spacing: {cfg.get('ego_spacing'):.2f}")
            print(f"    initial_lane_id: {cfg.get('initial_lane_id')}")

    # Save results
    results = {
        'method': 'Random Search',
        'seed': seed,
        'n_scenarios': n_scenarios,
        'runtime_seconds': round(elapsed, 2),
        'crashes_found': len(crashes),
        'crash_configs': crashes
    }

    filename = f'rs_results_seed_{seed}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    main()
