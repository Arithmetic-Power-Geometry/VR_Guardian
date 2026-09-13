from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math

@dataclass(frozen=True)
class Telemetry:
    head_rotation: float
    head_accel: float
    task_errors: float
    hesitation: float
    frame_drop: float
    hazard_proximity: float
    task_difficulty: float
    workload: float

@dataclass(frozen=True)
class Action:
    key: str
    name: str
    human_relief: float
    operational_relief: float
    disruption: float
    learning_loss: float

@dataclass(frozen=True)
class Weights:
    alpha_operational: float = 1.15
    beta_disruption: float = 0.70
    gamma_learning: float = 1.10

SCENARIO_PROFILES = {
    'forklift': {'human': 1.00, 'operational': 1.10},
    'crane': {'human': 1.05, 'operational': 1.15},
    'substation': {'human': 0.95, 'operational': 1.25},
    'mining': {'human': 1.10, 'operational': 1.20},
    'oil_gas': {'human': 1.05, 'operational': 1.30},
    'factory': {'human': 0.95, 'operational': 1.05},
}

ACTIONS: Dict[str, Action] = {
    'none': Action('none','No intervention',0.00,0.00,0.00,0.00),
    'stabilize': Action('stabilize','Stabilize scene',0.18,0.02,0.05,0.03),
    'reduce_rotation': Action('reduce_rotation','Reduce rotation speed',0.24,0.03,0.07,0.05),
    'dynamic_fov': Action('dynamic_fov','Dynamic FOV',0.29,0.00,0.11,0.08),
    'simplify': Action('simplify','Simplify visual clutter',0.16,0.06,0.08,0.10),
    'slow_scenario': Action('slow_scenario','Slow scenario pace',0.20,0.17,0.10,0.18),
    'micro_pause': Action('micro_pause','Micro-pause',0.32,0.10,0.16,0.16),
    'teleport': Action('teleport','Switch locomotion mode',0.36,0.04,0.18,0.15),
    'stop': Action('stop','Stop session',0.60,0.55,0.35,0.50),
}

def _sigmoid(z: float) -> float:
    return 1.0/(1.0+math.exp(-z))

def estimate_states(t: Telemetry, scenario: str='forklift') -> Tuple[float,float]:
    p = SCENARIO_PROFILES.get(scenario, SCENARIO_PROFILES['forklift'])
    h_raw = (
        1.05*t.head_rotation + 0.85*t.head_accel + 1.10*t.task_errors +
        0.90*t.hesitation + 0.70*t.frame_drop + 0.85*t.workload - 2.95
    ) * p['human']
    o_raw = (
        1.35*t.hazard_proximity + 1.00*t.task_errors + 0.85*t.hesitation +
        1.10*t.task_difficulty + 0.55*t.workload - 2.65
    ) * p['operational']
    return _sigmoid(h_raw), _sigmoid(o_raw)

def score_action(h: float, o: float, a: Action, w: Weights) -> float:
    h_next = max(0.0, h-a.human_relief)
    o_next = max(0.0, o-a.operational_relief)
    return h_next + w.alpha_operational*o_next + w.beta_disruption*a.disruption + w.gamma_learning*a.learning_loss

def choose_guardian_action(t: Telemetry, scenario: str='forklift', w: Weights = Weights(), safe_h: float = 0.48, safe_o: float = 0.52):
    h,o = estimate_states(t, scenario)
    if h < safe_h and o < safe_o:
        return ACTIONS['none'], h, o
    if h > 0.82 and o > 0.82:
        return ACTIONS['stop'], h, o
    candidates = [a for a in ACTIONS.values() if a.key != 'stop']
    ranked = sorted(candidates, key=lambda a: score_action(h,o,a,w))
    return ranked[0], h, o

def choose_policy_action(policy: str, t: Telemetry, scenario: str='forklift', w: Weights=Weights()):
    h,o = estimate_states(t, scenario)
    if policy == 'Fixed Training':
        return ACTIONS['none'], h, o
    if policy == 'Safety-First':
        if h >= 0.48 or o >= 0.52:
            return ACTIONS['stop'], h, o
        return ACTIONS['none'], h, o
    if policy == 'Comfort-First':
        if h >= 0.48:
            return ACTIONS['micro_pause'], h, o
        return ACTIONS['none'], h, o
    return choose_guardian_action(t, scenario, w)

def apply_action(h: float, o: float, a: Action):
    return max(0.0,h-a.human_relief), max(0.0,o-a.operational_relief)
