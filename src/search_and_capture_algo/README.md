# Search and Capture (MILP)

Simple multi-robot search planning project using Gurobi MILP.

## Main files

- `code/main.py` - solver entry point
- `code/helper.py` - MILP variables/constraints + plotting
- `code/searchers.py` - searcher setup
- `code/target.py` - target motion model

## Setup

1. Install Gurobi and activate your license.
2. Install dependencies:

   `pip install -r requirements.txt`

## Run

`python3 code/main.py`

Frames are saved as `results/path_t=<t>.png`.

## Change configuration (simple)

Open `code/main.py`, class `Mespp`, method `__init__`, then edit:

- `grid_side` (grid size)
- `self.HORIZON` (planning steps)
- `target_start_vertex`
- `Target(..., motion="uniform" or "stationary")`
- `Searchers(..., M=..., initial_positions=np.array([...]))`

Run again:

`python3 code/main.py`

## Optional tools

- `python3 benchmark_configs.py`
- `python3 visualize_interactive.py`
- `python3 animate_results.py`
