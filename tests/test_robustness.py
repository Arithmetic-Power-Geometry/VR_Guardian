from scripts.robustness import evaluate
from vr_guardian.core import Weights

def test_robustness_evaluate_small():
    df=evaluate('tiny',Weights(),1.0,True,seed=1,sessions=1,steps=10)
    assert len(df)==6
    assert df.mean_joint_risk.between(0,1).all()

def test_learning_ablation_runs():
    a=evaluate('on',Weights(),1.0,True,seed=2,sessions=1,steps=20)
    b=evaluate('off',Weights(),1.0,False,seed=2,sessions=1,steps=20)
    assert len(a)==len(b)==6
