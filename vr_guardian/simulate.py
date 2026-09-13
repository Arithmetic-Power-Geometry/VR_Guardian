from __future__ import annotations
import random
import pandas as pd
from .core import Telemetry, choose_policy_action, apply_action, Weights

SCENARIOS = ['forklift','crane','substation','mining','oil_gas','factory']
POLICIES = ['Fixed Training','Safety-First','Comfort-First','VR Guardian']

def _telemetry(rng, trend):
    return Telemetry(
        head_rotation=min(1,max(0,0.30+0.45*trend+rng.gauss(0,0.12))),
        head_accel=min(1,max(0,0.25+0.35*trend+rng.gauss(0,0.12))),
        task_errors=min(1,max(0,0.18+0.35*trend+rng.gauss(0,0.10))),
        hesitation=min(1,max(0,0.20+0.38*trend+rng.gauss(0,0.11))),
        frame_drop=min(1,max(0,0.10+0.18*trend+rng.gauss(0,0.08))),
        hazard_proximity=min(1,max(0,0.28+0.45*trend+rng.gauss(0,0.10))),
        task_difficulty=min(1,max(0,0.35+0.35*trend+rng.gauss(0,0.08))),
        workload=min(1,max(0,0.25+0.42*trend+rng.gauss(0,0.10))),
    )

def generate_base_trajectory(seed: int=7, steps: int=120):
    rng = random.Random(seed)
    return [_telemetry(rng, t/max(1,steps-1)) for t in range(steps)]

def generate_session(seed: int=7, steps: int=120, scenario: str='forklift', policy: str='VR Guardian', weights: Weights=Weights()):
    traj = generate_base_trajectory(seed, steps)
    rows=[]
    for t,x in enumerate(traj):
        a,h,o = choose_policy_action(policy,x,scenario,weights)
        h2,o2 = apply_action(h,o,a)
        rows.append({
            'step':t,'scenario':scenario,'policy':policy,
            'human_risk':h,'operational_risk':o,
            'post_human_risk':h2,'post_operational_risk':o2,
            'action':a.name,'action_key':a.key,
            'disruption':a.disruption,'learning_loss':a.learning_loss,
            'joint_risk':0.5*(h2+o2)
        })
    return pd.DataFrame(rows)

def benchmark(seed: int=11, sessions_per_scenario: int=40, steps: int=120):
    out=[]
    k=0
    for s in SCENARIOS:
        for i in range(sessions_per_scenario):
            session_seed = seed + 1000*k
            for p in POLICIES:
                df=generate_session(session_seed,steps,s,p)
                out.append({
                    'scenario':s,'session':i,'policy':p,
                    'mean_human_risk':df.post_human_risk.mean(),
                    'mean_operational_risk':df.post_operational_risk.mean(),
                    'mean_joint_risk':df.joint_risk.mean(),
                    'mean_disruption':df.disruption.mean(),
                    'mean_learning_loss':df.learning_loss.mean(),
                    'intervention_rate':(df.action_key!='none').mean(),
                    'stop_rate':(df.action_key=='stop').mean(),
                })
            k += 1
    return pd.DataFrame(out)
