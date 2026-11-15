"""DiagnoseLab demo: watch the oracle, play an episode yourself, or browse the validation results.

    uv run --extra demo streamlit run demo/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plant_view import draw  # noqa: E402

from diagnoselab.oracle.oracle import run_oracle  # noqa: E402
from diagnoselab.scoring.reward import normalization, score_episode  # noqa: E402
from diagnoselab.server.episode import Episode  # noqa: E402
from diagnoselab.server.tools import REPAIR_KIND, ToolRejected, public_topology  # noqa: E402
from diagnoselab.sim.instance import make_instance  # noqa: E402

FAULT_FOR_KIND = {"pipe": "fouling", "valve": "stuck_valve", "pump": "pump_wear", "junction": "leak"}
REPAIR_FOR = {"fouling": "clean_pipe", "stuck_valve": "free_valve", "pump_wear": "replace_pump", "leak": "fix_leak",
              "sensor_bias": "recalibrate_sensor", "sensor_stuck": "recalibrate_sensor"}
TYPICAL = {"fouling": 1.0, "stuck_valve": 0.5, "pump_wear": 0.2, "leak": 0.05, "sensor_bias": 6.0}

st.set_page_config(page_title="DiagnoseLab demo", layout="wide")
st.markdown("""<style>
h1, h2, h3 { color: #1b4073; } .stMetric label { color: #b86300 !important; }
div[data-testid="stMetricValue"] { color: #1b4073; }
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def instance_view(seed: int, tier: str) -> tuple[dict, list[dict]]:
    inst = make_instance(seed, tier)
    truth = [{"component": f.component, "type": f.type, "magnitude": f.magnitude} for f in inst.faults]
    return public_topology(inst.spec), truth


@st.cache_data(show_spinner="The oracle is diagnosing the plant…")
def oracle_run(seed: int, tier: str) -> dict:
    inst = make_instance(seed, tier)
    ep = Episode(inst)
    calls: list[dict] = []

    def call(tool: str, args: dict) -> dict:
        res = ep.call(tool, args)
        if tool in ("read_sensor", "run_experiment"):
            calls.append({"tool": tool, "args": args, "result": res})
        return res

    out = run_oracle(call)
    score = score_episode(inst, ep._state)
    score |= normalization(score["reward"], seed, tier)
    return {"out": out, "calls": calls, "score": score}


def truth_map(truth: list[dict]) -> dict[str, str]:
    return {f["component"]: f["type"] for f in truth}


def score_panel(score: dict) -> None:
    c = st.columns(5)
    c[0].metric("Diagnosis D", f"{score['D']:.2f}")
    c[1].metric("Restoration S", f"{score['S']:.2f}")
    c[2].metric("Budget spent", f"{score['spent']:.0f} / 40")
    c[3].metric("Raw reward R", f"{score.get('reward_raw', score['reward']):.3f}")
    c[4].metric("Normalised", f"{score['score_norm']:.2f}" if score.get("normalized") else "n/a")


def compare_table(truth: list[dict], diagnosis: list[dict]) -> pd.DataFrame:
    sub = {(d["component"], d["type"]): d.get("magnitude") for d in diagnosis}
    rows = [{"component": f["component"], "type": f["type"], "true magnitude": f["magnitude"],
             "submitted": sub.get((f["component"], f["type"]), "missed")} for f in truth]
    rows += [{"component": c, "type": t, "true magnitude": None, "submitted": f"{m} (false positive)"}
             for (c, t), m in sub.items() if (c, t) not in {(f["component"], f["type"]) for f in truth}]
    return pd.DataFrame(rows)


# --- sidebar -------------------------------------------------------------------------------------------------
st.sidebar.title("DiagnoseLab")
st.sidebar.caption("Hidden-fault diagnosis under a probe budget")
mode = st.sidebar.radio("Mode", ["Watch the oracle", "Play yourself", "Results"])
tier = st.sidebar.selectbox("Tier", ["easy", "medium", "hard"], index=1)
seed = int(st.sidebar.number_input("Seed (dev 0-199)", min_value=0, max_value=199, value=3, step=1))
reveal = st.sidebar.toggle("Reveal the hidden truth", value=False)
topo, truth = instance_view(seed, tier)

# --- watch the oracle ----------------------------------------------------------------------------------------
if mode == "Watch the oracle":
    st.title("Watch the oracle")
    st.write("The reference oracle plays through the same tools as an agent. It picks each probe by information gained "
             "per unit of cost and fits faults with the free digital twin.")
    run = oracle_run(seed, tier)
    trace = run["out"]["trace"]
    step = st.slider("Probe", 0, len(trace), len(trace), help="0 = before the first probe")
    probed = set()
    for c in run["calls"][:step]:
        probed |= ({c["args"]["sensor_id"]} if c["tool"] == "read_sensor" else set(c["result"]["readings"]))
    current = trace[step - 1]["support"] if step else []
    diagnosed = {(s.split(":")[1] if s.startswith("bias:") else s.split(":")[0]):
                 ("sensor_bias" if s.startswith("bias:") else s.split(":")[1]) for s in current}
    st.pyplot(draw(topo, truth_map(truth) if reveal else None, diagnosed if reveal else None, probed), clear_figure=True)
    with st.container():
        st.subheader("Probes so far")
        rows = [{"#": i + 1, "probe": t["probe"], "spent": t["spent"], "current hypothesis": ", ".join(t["support"]) or "-"}
                for i, t in enumerate(trace[:step])]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=360)
        st.caption("Orange rings and dots: sensors read so far. The hypothesis is the oracle's fitted fault set.")
    if step == len(trace):
        st.subheader("Submission and score")
        score_panel(run["score"])
        if reveal:
            st.dataframe(compare_table(truth, run["out"]["submission"]["diagnosis"]), hide_index=True, use_container_width=True)
        else:
            st.info("Turn on *Reveal the hidden truth* to compare the diagnosis with the hidden faults.")

# --- play yourself -------------------------------------------------------------------------------------------
elif mode == "Play yourself":
    st.title("Play yourself")
    key = f"{tier}:{seed}"
    if st.session_state.get("key") != key or st.sidebar.button("Restart episode"):
        st.session_state.update(key=key, ep=Episode(make_instance(seed, tier)), log=[], done=False)
    ep: Episode = st.session_state.ep
    spent = ep._state["spent"]
    st.progress(min(spent / 40, 1.0), text=f"Budget spent: {spent} / 40")
    sensors = [s["id"] for s in topo["sensors"]]
    probed = {sid for e in st.session_state.log for sid in e.get("readings", {})}
    left, right = st.columns([3, 2])
    with left:
        st.pyplot(draw(topo, truth_map(truth) if reveal and st.session_state.done else None, None, probed), clear_figure=True)
    with right:
        tab_read, tab_exp, tab_sim = st.tabs(["Read a sensor (1)", "Run an experiment (4-8)", "Simulate (free)"])
        with tab_read:
            sid = st.selectbox("Sensor", sensors)
            if st.button("Read", disabled=st.session_state.done):
                try:
                    r = ep.call("read_sensor", {"sensor_id": sid})
                    st.session_state.log.append({"action": f"read {sid}", "readings": {sid: r["value"]}})
                except ToolRejected as e:
                    st.error(e.message)
                st.rerun()
        with tab_exp:
            kind = st.selectbox("Experiment", ["speed_step", "valve_step", "isolate_branch", "pump_off_static"])
            args: dict = {}
            if kind == "speed_step":
                args["s"] = st.select_slider("Pump speed", [0.7, 0.75, 0.8, 0.85, 0.9, 1.1, 1.15, 1.2], value=1.2)
            elif kind == "valve_step":
                args["valve"] = st.selectbox("Valve", [l["id"] for l in topo["links"] if l["kind"] == "valve"])
                args["theta"] = st.slider("Commanded opening", 0.1, 0.95, 0.3, 0.05)
            elif kind == "isolate_branch":
                args["branch"] = st.selectbox("Branch", [b["id"] for b in topo["branches"]])
            if st.button("Run experiment", disabled=st.session_state.done):
                try:
                    r = ep.call("run_experiment", {"type": kind, "args": args})
                    st.session_state.log.append({"action": f"{kind} {args}", "readings": r["readings"]})
                except ToolRejected as e:
                    st.error(e.message)
                st.rerun()
        with tab_sim:
            comps = [(l["id"], FAULT_FOR_KIND[l["kind"]]) for l in topo["links"]] + [(j["id"], "leak") for j in topo["junctions"]]
            pick = st.selectbox("Hypothesis: one fault", [f"{c} · {t}" for c, t in comps])
            comp, typ = pick.split(" · ")
            mag = st.number_input("Magnitude", value=TYPICAL[typ], min_value=0.0, format="%.3f")
            if st.button("Predict readings at nominal"):
                pred = ep.call("simulate", {"overrides": [{"component": comp, "type": typ, "magnitude": mag}]})["readings"]
                base = ep.call("simulate", {"overrides": []})["readings"]
                st.dataframe(pd.DataFrame([{"sensor": s, "healthy": base[s], "with fault": pred[s],
                                            "shift (sigma)": round((pred[s] - base[s]) / next(x["sigma"] for x in topo["sensors"] if x["id"] == s), 1)}
                                           for s in pred]), hide_index=True, use_container_width=True, height=280)
    st.subheader("Measurements")
    if st.session_state.log:
        st.dataframe(pd.DataFrame([{"#": i + 1, "action": e["action"],
                                    "readings": ", ".join(f"{k}={v:.3f}" for k, v in e["readings"].items())}
                                   for i, e in enumerate(st.session_state.log)]), hide_index=True, use_container_width=True)
    else:
        st.caption("No measurements yet. Reading the manual, the topology and simulating are free; measurements cost budget.")
    st.subheader("Submit")
    options = [f"{c} · {t}" for c, t in ([(l["id"], FAULT_FOR_KIND[l["kind"]]) for l in topo["links"]] +
                                          [(j["id"], "leak") for j in topo["junctions"]] +
                                          [(s, "sensor_bias") for s in sensors] + [(s, "sensor_stuck") for s in sensors])]
    chosen = st.multiselect("Your diagnosis (up to 10 faults)", options, max_selections=10, disabled=st.session_state.done)
    diagnosis = []
    for opt in chosen:
        comp, typ = opt.split(" · ")
        entry = {"component": comp, "type": typ}
        if typ != "sensor_stuck":
            entry["magnitude"] = st.number_input(f"Magnitude of {opt}", value=TYPICAL[typ], min_value=0.0, format="%.3f",
                                                 key=f"m-{opt}", disabled=st.session_state.done)
        diagnosis.append(entry)
    if st.button("Submit diagnosis and repair plan", type="primary", disabled=st.session_state.done):
        repairs = [{"action": REPAIR_FOR[d["type"]], "target": d["component"]} for d in diagnosis
                   if not d["type"].startswith("sensor_")]
        try:
            ep.call("submit", {"diagnosis": diagnosis, "repairs": repairs})
            st.session_state.done = True
        except ToolRejected as e:
            st.error(e.message)
        st.rerun()
    if st.session_state.done:
        score = score_episode(make_instance(seed, tier), ep._state)
        score |= normalization(score["reward"], seed, tier)
        score_panel(score)
        st.dataframe(compare_table(truth, ep._state["submission"]["diagnosis"]), hide_index=True, use_container_width=True)
        oracle = oracle_run(seed, tier)["score"]
        st.caption(f"The oracle scored D {oracle['D']:.2f} with {oracle['spent']:.0f} units on this plant.")

# --- results -------------------------------------------------------------------------------------------------
else:
    st.title("Validation results")
    res = ROOT / "results"
    s5 = json.loads((res / "step5_validation.json").read_text())
    s6 = json.loads((res / "step6_validation.json").read_text())
    st.subheader("Oracle (held-out dev seeds 50-99)")
    held = s5["F19"]["heldout"]
    st.dataframe(pd.DataFrame([{"tier": t, "D": round(v["D"], 3), "raw R": round(v["R"], 3), "mean spend": round(v["spent_mean"], 1),
                                "max runtime (s)": round(v["runtime_max"], 2)} for t, v in held.items()]),
                 hide_index=True, use_container_width=True)
    st.subheader("Frontier agent vs configuration (normalised score)")
    cfg = s6["configs"]
    label = {"B40": "B = 40 (frozen)", "B26": "B = 26", "B26_subtle": "B = 26 + subtle faults",
             "B26_meters": "B = 26 + half the meters"}
    df = pd.DataFrame({label.get(k, c["label"]): {t: c["tiers"][t]["mean"] for t in ("easy", "medium", "hard")}
                       for k, c in cfg.items()})
    st.bar_chart(df.T, horizontal=True, stack=False, color=["#7fa3d1", "#ed7d20", "#1b4073"])
    st.caption("Kimi K3.1 Pro through Harbor, 105 episodes. Target band 0.35-0.65; the frozen configuration is budget 40.")
    base_file = res / "step7_baselines.json"
    if base_file.exists():
        st.subheader("Baselines on the 60 test seeds (normalised)")
        rows = json.loads(base_file.read_text())
        bdf = pd.DataFrame(rows).groupby(["policy", "tier"])["score_norm"].mean().unstack()[["easy", "medium", "hard"]]
        st.dataframe(bdf.round(3), use_container_width=True)
