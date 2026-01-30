# How to Run & Reproduce Results

This document provides a concise guide for running the Hill Climber and Random Search algorithms, reproducing results, and understanding special setup requirements for this project.

---

## 1. Running the Hill Climber

To run the Hill Climbing search algorithm:

```bash
python main.py
```

- By default, `main.py` is set up to run the Hill Climber (see the call to `hill_climb` in the script).
- The script will print the best fitness, whether a crash was found, the minimum distance, and the best scenario configuration found.

---

## 2. Running Random Search (Baseline)

To run the Random Search baseline:

1. Open `main.py`.
2. Uncomment the block that instantiates and runs `RandomSearch`.
3. Comment out or remove the `hill_climb` block if needed.
4. Run:

```bash
python main.py
```

---

## 3. Reproducing Results

- Results reported in the paper (e.g., crash scenarios, fitness progression) can be reproduced by running `main.py` with the same seeds and parameters as described in the report or in the `fitness_progression.json` and `experiment_results.json` files.
- To reproduce a run with a specific seed, edit the `seed` parameter in the `hill_climb` call in `main.py` (e.g., `seed=532`).
- Fitness progression and crash configurations are logged in the provided JSON files for reference.
- Random Search results can be reproduced by running the baseline as described above.

---

## 4. Special Setup Steps

- Ensure all dependencies are installed (see `requirements.txt`).
- The project requires Python 3.8+ and works best in a virtual environment.
- Pre-trained agent files must be present in the `agents/` directory (these are required for evaluation).
- Videos of crash scenarios will be saved in the `videos/` directory after running the search algorithms.
- You can modify the number of iterations, neighbors per iteration, or seeds in `main.py` to run your own experiments.

---

For further details, see the main `README.md` or the report.
