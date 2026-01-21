import winsound
from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.hill_climbing import hill_climb
from search.random_search import RandomSearch
from tkinter import messagebox
import tkinter as tk

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    # search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    # crashes = search.run_search(n_scenarios=50, seed=12288)

    # print(f"✅ Found {len(crashes)} crashes.")
    # if crashes:
    #    print(crashes)
    
    
    # Uncomment following block to run the Hill Climbing search with some printing included.
    result = hill_climb(
        env_id=env_id,
        base_cfg=base_cfg,
        param_spec=param_spec,
        policy=policy,
        defaults=defaults,
        seed=1362,
        iterations=25,
        neighbors_per_iter=6
    )
    
    # Seeds to use:
    # 532 -> Contiuously increase fitness and crash on iteration 11
    # 1362 -> Get stuck in local optimum once at iteration 6, then restarts and crashes at iteration 8

    print(f"\n✅ Hill Climbing Results:")
    print(f"   Best fitness: {result['best_fitness']}")
    print(f"   Crash found: {result['best_objectives']['crash_count'] == 1}")
    print(f"   Min distance: {result['best_objectives']['min_distance']:.2f}")
    print(f"   Best config: {result['best_cfg']}")

if __name__ == "__main__":
    main()