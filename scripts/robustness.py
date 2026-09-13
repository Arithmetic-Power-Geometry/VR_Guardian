from pathlib import Path
import sys, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from vr_guardian.core import Weights, ACTIONS, estimate_states
from vr_guardian.simulate import SCENARIOS, generate_base_trajectory

def choose_custom(t, scenario, w, action_scale=1.0, use_learning=True):
    h,o=estimate_states(t,scenario)
    if h < 0.48 and o < 0.52: return ACTIONS['none'],h,o
    if h > 0.82 and o > 0.82: return ACTIONS['stop'],h,o
    gamma=w.gamma_learning if use_learning else 0.0
    ww=Weights(w.alpha_operational,w.beta_disruption,gamma)
    candidates=[a for a in ACTIONS.values() if a.key!='stop']
    def score(a):
        h2=max(0,h-action_scale*a.human_relief); o2=max(0,o-action_scale*a.operational_relief)
        return h2+ww.alpha_operational*o2+ww.beta_disruption*a.disruption+ww.gamma_learning*a.learning_loss
    return min(candidates,key=score),h,o

def evaluate(label,w=Weights(),action_scale=1.0,use_learning=True,seed=11,sessions=40,steps=120):
    rows=[]; k=0
    for s in SCENARIOS:
        for i in range(sessions):
            vals=[]
            for t in generate_base_trajectory(seed+1000*k,steps):
                a,h,o=choose_custom(t,s,w,action_scale,use_learning)
                h2=max(0,h-action_scale*a.human_relief); o2=max(0,o-action_scale*a.operational_relief)
                vals.append((.5*(h2+o2),a.disruption,a.learning_loss,a.key!='none',a.key=='stop'))
            arr=np.array(vals,float)
            rows.append({'experiment':label,'scenario':s,'session':i,'mean_joint_risk':arr[:,0].mean(),'mean_disruption':arr[:,1].mean(),'mean_learning_loss':arr[:,2].mean(),'intervention_rate':arr[:,3].mean(),'stop_rate':arr[:,4].mean()})
            k+=1
    return pd.DataFrame(rows)

def main():
    runs=[]
    for a in [0.75,1.15,1.55]:
        for b in [0.35,0.70,1.05]:
            for g in [0.55,1.10,1.65]: runs.append(evaluate(f'w_a{a}_b{b}_g{g}',Weights(a,b,g)))
    for scale in [0.70,0.85,1.00,1.15,1.30]: runs.append(evaluate(f'action_scale_{scale:.2f}',Weights(),scale,True))
    runs.append(evaluate('ablation_learning_on',Weights(),1.0,True)); runs.append(evaluate('ablation_learning_off',Weights(),1.0,False))
    all_df=pd.concat(runs,ignore_index=True); out=ROOT/'results'/'robustness'; out.mkdir(parents=True,exist_ok=True)
    all_df.to_csv(out/'robustness_sessions.csv',index=False); summ=all_df.groupby('experiment').mean(numeric_only=True).reset_index(); summ.to_csv(out/'robustness_summary.csv',index=False)
    weight=summ[summ.experiment.str.startswith('w_')]; effects=summ[summ.experiment.str.startswith('action_scale_')]; abl=summ[summ.experiment.str.startswith('ablation_')]
    report={'weight_grid_runs':int(len(weight)),'weight_joint_risk_range':[float(weight.mean_joint_risk.min()),float(weight.mean_joint_risk.max())],'weight_disruption_range':[float(weight.mean_disruption.min()),float(weight.mean_disruption.max())],'weight_learning_loss_range':[float(weight.mean_learning_loss.min()),float(weight.mean_learning_loss.max())],'action_effect_joint_risk_range':[float(effects.mean_joint_risk.min()),float(effects.mean_joint_risk.max())],'action_effect_disruption_range':[float(effects.mean_disruption.min()),float(effects.mean_disruption.max())],'ablation':abl[['experiment','mean_joint_risk','mean_disruption','mean_learning_loss','intervention_rate','stop_rate']].to_dict(orient='records')}
    (out/'robustness_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
