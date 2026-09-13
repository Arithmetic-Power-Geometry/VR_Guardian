from vr_guardian.simulate import generate_session, benchmark, POLICIES, SCENARIOS

def test_session_length():
    assert len(generate_session(1,40,'crane','VR Guardian'))==40

def test_session_columns():
    df=generate_session(1,20)
    for c in ['human_risk','operational_risk','post_human_risk','post_operational_risk','action','policy']:
        assert c in df.columns

def test_action_never_increases_risk():
    df=generate_session(2,80)
    assert (df.post_human_risk <= df.human_risk + 1e-12).all()
    assert (df.post_operational_risk <= df.operational_risk + 1e-12).all()

def test_benchmark_all_scenarios_and_policies():
    df=benchmark(3,2,20)
    assert df.scenario.nunique()==len(SCENARIOS)
    assert df.policy.nunique()==len(POLICIES)
    assert len(df)==len(SCENARIOS)*2*len(POLICIES)

def test_guardian_stop_rate_below_safety_first():
    df=benchmark(5,4,50)
    means=df.groupby('policy').stop_rate.mean()
    assert means['VR Guardian'] < means['Safety-First']
