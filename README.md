# VR Guardian

**Safe Difficulty: Human-State-Aware Adaptive Control for Industrial Virtual-Reality Training**

VR Guardian is a reproducible research prototype for industrial VR training. It jointly models **human-state risk** and **operational/task risk**, then chooses adaptations while penalizing experience disruption and **learning loss** from making training too easy.

## Core question

> What adaptation protects the trainee while preserving the difficulty the trainee still needs to master?

## Policies

- Fixed Training
- Safety-First
- Comfort-First
- VR Guardian

## Scenarios

Forklift, crane, electrical substation, mining, oil & gas, and factory training.

## Run locally

```bash
python -m pip install -r requirements.txt
pytest -q
python scripts/run_all.py
streamlit run app.py
```

## Reproducibility outputs

The workflow generates and stores:

- `results/benchmark.csv`
- `results/policy_summary.csv`
- `results/scenario_policy_summary.csv`
- `results/summary.json`
- `results/REPORT.md`
- publication-ready SVG figures in `results/figures/`

GitHub Actions tests Python 3.10, 3.11, and 3.12, uploads the result bundle as workflow artifacts, and commits the generated reproducibility results from Python 3.12 back to `main`.

## Research scope

All numerical outputs are synthetic software-validation results. They do not establish human safety, medical efficacy, or industrial deployment performance.

## License

Apache License 2.0  
Copyright (C) 2026 Mohammad Amir Khusru Akhtar
