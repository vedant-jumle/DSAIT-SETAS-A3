"""
Experiment script for Assignment 3 - HC vs Random Search comparison
Run with: python run_experiments.py
"""

import time
import json
from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env, record_video_episode
from search.hill_climbing import hill_climb, compute_objectives_from_time_series
from search.random_search import RandomSearch

def run_hill_climbing_experiment(env_id, policy, defaults, seed, iterations=25, neighbors_per_iter=6):
    """Run Hill Climbing with timing and metrics."""
    print(f"\n{'='*60}")
    print(f"HILL CLIMBING - SEED {seed}")
    print(f"{'='*60}")

    start = time.time()
    result = hill_climb(
        env_id=env_id,
        base_cfg=base_cfg,
        param_spec=param_spec,
        policy=policy,
        defaults=defaults,
        seed=seed,
        iterations=iterations,
        neighbors_per_iter=neighbors_per_iter
    )
    elapsed = time.time() - start

    # Calculate evaluations: 1 initial + (iterations * neighbors_per_iter)
    # But if crashed early, use actual iterations from history
    actual_iterations = len(result['history']) - 1
    evaluations = 1 + (actual_iterations * neighbors_per_iter)

    metrics = {
        'method': 'Hill Climbing',
        'seed': seed,
        'runtime_seconds': round(elapsed, 2),
        'iterations': actual_iterations,
        'evaluations': evaluations,
        'crash_found': result['best_objectives']['crash_count'] == 1,
        'min_distance': round(result['best_objectives']['min_distance'], 4),
        'best_config': result['best_cfg'],
        'best_objectives': result['best_objectives'],
        'best_fitness': result['best_fitness'],
        'best_seed_base': result['best_seed_base']
    }

    print(f"\n--- HC SEED {seed} RESULTS ---")
    print(f"Runtime: {metrics['runtime_seconds']} seconds")
    print(f"Iterations to crash/end: {metrics['iterations']}")
    print(f"Total evaluations: {metrics['evaluations']}")
    print(f"Crash found: {metrics['crash_found']}")
    print(f"Min distance: {metrics['min_distance']}")
    print(f"Best config: {metrics['best_config']}")

    return metrics, result

def run_random_search_experiment(env_id, policy, defaults, seed, n_scenarios=50):
    """Run Random Search with timing and metrics."""
    print(f"\n{'='*60}")
    print(f"RANDOM SEARCH - SEED {seed}, n_scenarios={n_scenarios}")
    print(f"{'='*60}")

    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)

    start = time.time()
    crashes = search.run_search(n_scenarios=n_scenarios, n_eval=1, seed=seed)
    elapsed = time.time() - start

    metrics = {
        'method': 'Random Search',
        'seed': seed,
        'n_scenarios': n_scenarios,
        'runtime_seconds': round(elapsed, 2),
        'evaluations': n_scenarios,
        'crashes_found': len(crashes),
        'crash_configs': crashes
    }

    print(f"\n--- RS SEED {seed} RESULTS ---")
    print(f"Runtime: {metrics['runtime_seconds']} seconds")
    print(f"Evaluations: {metrics['evaluations']}")
    print(f"Crashes found: {metrics['crashes_found']}")
    if crashes:
        for i, crash in enumerate(crashes):
            print(f"  Crash {i+1}: config={crash['cfg']}, seed={crash['seed']}")

    return metrics, crashes

def main():
    print("="*60)
    print("ASSIGNMENT 3 - EXPERIMENTAL EVALUATION")
    print("Hill Climbing vs Random Search Comparison")
    print("="*60)

    # Setup
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    all_results = {
        'hill_climbing': [],
        'random_search': []
    }

    # =====================
    # HILL CLIMBING EXPERIMENTS
    # =====================
    hc_seeds = [532, 1362]  # Seeds that produce crashes

    for seed in hc_seeds:
        metrics, result = run_hill_climbing_experiment(
            env_id, policy, defaults, seed,
            iterations=25, neighbors_per_iter=6
        )
        all_results['hill_climbing'].append(metrics)

        # Record video if crash found
        if metrics['crash_found']:
            print(f"Recording video for HC crash (seed {seed})...")
            record_video_episode(
                env_id,
                metrics['best_config'],
                policy,
                defaults,
                metrics['best_seed_base'],
                out_dir=f"videos/hc_seed_{seed}"
            )

    # =====================
    # RANDOM SEARCH EXPERIMENTS
    # =====================
    rs_seed = 12288
    rs_n_scenarios = 50

    metrics, crashes = run_random_search_experiment(
        env_id, policy, defaults, rs_seed, n_scenarios=rs_n_scenarios
    )
    all_results['random_search'].append(metrics)

    # =====================
    # SUMMARY
    # =====================
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)

    print("\n### Hill Climbing Results ###")
    print(f"{'Seed':<10} {'Runtime(s)':<12} {'Iterations':<12} {'Evaluations':<12} {'Crash':<8} {'Min Dist':<10}")
    print("-"*70)
    for hc in all_results['hill_climbing']:
        print(f"{hc['seed']:<10} {hc['runtime_seconds']:<12} {hc['iterations']:<12} {hc['evaluations']:<12} {str(hc['crash_found']):<8} {hc['min_distance']:<10}")

    print("\n### Random Search Results ###")
    print(f"{'Seed':<10} {'Runtime(s)':<12} {'Scenarios':<12} {'Crashes':<10}")
    print("-"*50)
    for rs in all_results['random_search']:
        print(f"{rs['seed']:<10} {rs['runtime_seconds']:<12} {rs['n_scenarios']:<12} {rs['crashes_found']:<10}")

    print("\n### Crash Configurations ###")
    print("\nHill Climbing crashes:")
    for hc in all_results['hill_climbing']:
        if hc['crash_found']:
            cfg = hc['best_config']
            print(f"  Seed {hc['seed']}: vehicles={cfg.get('vehicles_count')}, lanes={cfg.get('lanes_count')}, "
                  f"initial_spacing={cfg.get('initial_spacing'):.2f}, ego_spacing={cfg.get('ego_spacing'):.2f}, "
                  f"initial_lane={cfg.get('initial_lane_id')}")

    print("\nRandom Search crashes:")
    for rs in all_results['random_search']:
        for i, crash in enumerate(rs['crash_configs']):
            cfg = crash['cfg']
            print(f"  Crash {i+1}: vehicles={cfg.get('vehicles_count')}, lanes={cfg.get('lanes_count')}, "
                  f"initial_spacing={cfg.get('initial_spacing'):.2f}, ego_spacing={cfg.get('ego_spacing'):.2f}, "
                  f"initial_lane={cfg.get('initial_lane_id')}")

    # Save results to JSON
    with open('experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nResults saved to experiment_results.json")

if __name__ == "__main__":
    main()
