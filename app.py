import streamlit as st
import plotly.express as px
from vr_guardian.core import Telemetry, Weights, choose_guardian_action, apply_action
from vr_guardian.simulate import generate_session, benchmark, POLICIES, SCENARIOS

st.set_page_config(page_title='VR Guardian', page_icon='🛡️', layout='wide')
st.markdown(
    """
    <style>
    .block-container {padding-top:1rem; max-width:1450px;}
    .hero {padding:1.4rem 1.6rem; border-radius:22px; border:1px solid rgba(128,128,128,.25);
           background:linear-gradient(135deg,rgba(50,80,120,.15),rgba(30,30,30,.02));}
    .small {opacity:.8;}
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="hero"><h1>🛡️ VR Guardian</h1><h3>Safe Difficulty for Industrial Virtual-Reality Training</h3><p class="small">Protect the trainee without unnecessarily erasing the difficulty the trainee needs to master.</p></div>', unsafe_allow_html=True)

live, sim, bench, about = st.tabs(['Live Controller','Policy Simulator','Benchmark','About'])

with live:
    st.subheader('Live human-state / operational-risk controller')
    scenario = st.selectbox('Industrial scenario', SCENARIOS)
    c = st.columns(4)
    vals = [
        c[0].slider('Head rotation stress',0.0,1.0,0.55,0.01),
        c[1].slider('Head acceleration',0.0,1.0,0.45,0.01),
        c[2].slider('Task errors',0.0,1.0,0.40,0.01),
        c[3].slider('Hesitation',0.0,1.0,0.45,0.01),
    ]
    c = st.columns(4)
    vals += [
        c[0].slider('Frame degradation',0.0,1.0,0.20,0.01),
        c[1].slider('Hazard proximity',0.0,1.0,0.55,0.01),
        c[2].slider('Training difficulty',0.0,1.0,0.65,0.01),
        c[3].slider('Workload',0.0,1.0,0.60,0.01),
    ]
    c = st.columns(3)
    w = Weights(
        c[0].slider('Operational-risk weight α',0.1,2.5,1.15,0.05),
        c[1].slider('Disruption weight β',0.1,2.0,0.70,0.05),
        c[2].slider('Learning-loss weight γ',0.1,2.5,1.10,0.05)
    )
    t = Telemetry(*vals)
    a,h,o = choose_guardian_action(t,scenario,w)
    h2,o2 = apply_action(h,o,a)
    m = st.columns(5)
    m[0].metric('Human risk',f'{h:.3f}',f'{h2-h:.3f}')
    m[1].metric('Operational risk',f'{o:.3f}',f'{o2-o:.3f}')
    m[2].metric('Action',a.name)
    m[3].metric('Disruption',f'{a.disruption:.3f}')
    m[4].metric('Learning loss',f'{a.learning_loss:.3f}')
    st.info(f'VR Guardian recommendation: **{a.name}**')

with sim:
    st.subheader('Policy simulator')
    c = st.columns(4)
    scenario = c[0].selectbox('Scenario', SCENARIOS, key='simscenario')
    policy = c[1].selectbox('Policy', POLICIES)
    seed = c[2].number_input('Seed',1,999999,7)
    steps = c[3].slider('Steps',30,300,120)
    df = generate_session(int(seed),int(steps),scenario,policy)
    st.plotly_chart(px.line(df,x='step',y=['human_risk','post_human_risk'],title='Human-state risk'),use_container_width=True)
    st.plotly_chart(px.line(df,x='step',y=['operational_risk','post_operational_risk'],title='Operational/task risk'),use_container_width=True)
    counts=df['action'].value_counts().rename_axis('action').reset_index(name='count')
    st.plotly_chart(px.bar(counts,x='action',y='count',title='Controller actions'),use_container_width=True)
    st.download_button('Download session CSV',df.to_csv(index=False).encode(),'vr_guardian_session.csv','text/csv')

with bench:
    st.subheader('Cross-scenario four-policy benchmark')
    if st.button('Run benchmark',type='primary'):
        b = benchmark()
        s = b.groupby('policy').mean(numeric_only=True).reset_index()
        st.dataframe(s,use_container_width=True)
        st.plotly_chart(px.scatter(s,x='mean_learning_loss',y='mean_joint_risk',size='intervention_rate',hover_name='policy',title='Risk-learning-loss trade-off'),use_container_width=True)
        st.plotly_chart(px.bar(s,x='policy',y='stop_rate',title='Stop rate by policy'),use_container_width=True)
        st.download_button('Download benchmark CSV',b.to_csv(index=False).encode(),'vr_guardian_benchmark.csv','text/csv')

with about:
    st.markdown('''### Research scope

VR Guardian is a reproducible research prototype for industrial VR training. It compares **Fixed Training**, **Safety-First**, **Comfort-First**, and **VR Guardian** policies across forklift, crane, electrical substation, mining, oil & gas, and factory scenarios.

The included numerical results are **synthetic software-validation outputs only**. They do not establish human safety, medical efficacy, or industrial deployment performance.
''')
