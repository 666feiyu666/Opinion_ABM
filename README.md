# Opinion Leader Influence in Agent-Based Online Opinion Dynamics

This repository contains the code and documentation for an agent-based model of online opinion dynamics. The model is named the **Opinion Leader Influence Model (OLIM)**. It studies how opinion leaders, content creation, platform-mediated exposure, opinion updating, and adaptive follower networks jointly shape polarization, opinion extremity, homophily, and echo-chamber-like structures.

## Project Information

- **Project title:** Opinion Leader Influence in Agent-Based Online Opinion Dynamics
- **Author:** Feiyu Tang
- **Supervisor / Advisor:** Pavel Loskot
- **Institution:** ZJUI
- **Academic year:** 2026
- **Status:** Senior thesis / undergraduate thesis project

## What This Repository Contains

The repository includes:

- A Python implementation of OLIM.
- Simulation modules for origination, content creation, diffusion, opinion updating, network adaptation, and round closure.
- Experiment scripts for baseline runs, leader-effects comparisons, no-leader controls, and robustness checks.
- Metrics and visualization utilities for analyzing opinion distributions, content balance, exposure, homophily, modularity, and network structure.
- ODD model documentation for replication-oriented model description.

## Thesis and Documentation

- A compiled ODD PDF is available as `ODD_Design.pdf`.
- The final thesis PDF should be placed at `[path/to/final_thesis.pdf]` when available.


## Model Overview

OLIM represents users as agents in a directed follower network. Each user has a latent continuous opinion, an expressed discrete stance, confidence, activity, follower count, and an opinion-leader indicator. Users may create posts with a stance and style, receive posts through follower-based and out-of-network exposure, update opinions after exposure, and revise follow ties toward encountered creators.

Opinion leaders are not a separate agent type. They are ordinary users with structural visibility advantages that affect posting origination, diffusion probability, exposure weight, response coefficients, and creator evaluation.

## Repository Structure

```text
.
├── config.py
├── initialization.py
├── main.py
├── metrics.py
├── utils.py
├── visualization.py
├── modules/
│   ├── origination.py
│   ├── content_creation.py
│   ├── diffusion.py
│   ├── opinion_update.py
│   ├── network_update.py
│   └── round_closure.py
├── experiments/
│   ├── run_baseline.py
│   └── run_leader_effects.py
├── notebooks/
├── Opinion_Dynamic_ABM_design/
├── Senior_Thesis/
└── ODD_Design.pdf
```

## Running the Model

Run the default simulation:

```bash
python main.py
```

Run the baseline experiment:

```bash
python experiments/run_baseline.py
```

Run the leader-effects experiment matrix:

```bash
python experiments/run_leader_effects.py --profile main --scenario core
```

For a smaller smoke test:

```bash
python experiments/run_leader_effects.py --profile trial --scenario core --max-runs 4
```

## Notes

This model is abstract and theory-driven. It is not calibrated to a single empirical platform. Parameter values operationalize modeling assumptions about online visibility, content circulation, expression, opinion updating, and adaptive network change.

## Citation

If citing this project, use:
