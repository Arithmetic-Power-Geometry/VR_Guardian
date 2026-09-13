from pathlib import Path
import sys, json
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from vr_guardian.simulate import benchmark

out = ROOT/'results'
out.mkdir(exist_ok=True)
figdir = out/'figures'
figdir.mkdir(exist_ok=True)

df = benchmark(seed=11, sessions_per_scenario=40, steps=120)
df.to_csv(out/'benchmark.csv', index=False)
summary = df.groupby('policy').mean(numeric_only=True).reset_index()
summary.to_csv(out/'policy_summary.csv', index=False)
scenario_summary = df.groupby(['scenario','policy']).mean(numeric_only=True).reset_index()
scenario_summary.to_csv(out/'scenario_policy_summary.csv', index=False)

report = {
    'base_sessions': int(df[['scenario','session']].drop_duplicates().shape[0]),
    'policy_evaluations': int(len(df)),
    'policies': sorted(df.policy.unique().tolist()),
    'scenarios': sorted(df.scenario.unique().tolist()),
    'policy_summary': summary.to_dict(orient='records')
}
(out/'summary.json').write_text(json.dumps(report,indent=2))
(out/'REPORT.md').write_text('# VR Guardian Reproducibility Report\n\n' + summary.to_markdown(index=False))

for metric,title,filename in [
    ('mean_joint_risk','Mean joint risk','fig1_joint_risk.svg'),
    ('mean_learning_loss','Mean learning loss','fig2_learning_loss.svg'),
    ('stop_rate','Mean stop rate','fig3_stop_rate.svg'),
    ('mean_disruption','Mean disruption','fig4_disruption.svg')
]:
    plt.figure(figsize=(7,4.2))
    plt.bar(summary['policy'],summary[metric])
    plt.ylabel(metric.replace('_',' ').title())
    plt.title(title)
    plt.xticks(rotation=15,ha='right')
    plt.tight_layout()
    plt.savefig(figdir/filename)
    plt.close()

print(summary.to_string(index=False))
