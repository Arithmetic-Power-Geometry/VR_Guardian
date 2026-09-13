from vr_guardian.core import Telemetry, estimate_states, choose_guardian_action, choose_policy_action, ACTIONS, score_action, Weights

def mild(): return Telemetry(.1,.1,.05,.05,.05,.05,.1,.1)
def severe(): return Telemetry(.95,.95,.95,.95,.8,.95,.95,.95)

def test_state_bounds_all_scenarios():
    for s in ['forklift','crane','substation','mining','oil_gas','factory']:
        h,o=estimate_states(mild(),s)
        assert 0<=h<=1 and 0<=o<=1

def test_guardian_no_action_when_safe():
    a,h,o=choose_guardian_action(mild(),'forklift')
    assert a.key=='none'

def test_extreme_joint_risk_stops():
    a,h,o=choose_guardian_action(severe(),'oil_gas')
    assert a.key=='stop'

def test_safety_first_uses_strongest():
    a,h,o=choose_policy_action('Safety-First', severe(), 'crane')
    assert a.key=='stop'

def test_all_scores_finite():
    h,o=estimate_states(severe(),'mining')
    for a in ACTIONS.values():
        assert score_action(h,o,a,Weights())>=0
