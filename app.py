from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="Amazon ML Challenge 2026 — Entity Resolution",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .team-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        border: 1px solid #C7D2FE;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-pass {
        color: #059669;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.markdown('<div class="main-title">🏢 Amazon ML Challenge 2026</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Business Entity Resolution Pipeline — Offline, Precision-Sensitive Record Linkage</div>',
    unsafe_allow_html=True,
)

# Team Information Badges
col_team, col_empty = st.columns([3, 1])
with col_team:
    st.markdown(
        """
        <span class="team-badge">👑 Team Leader: Madhan Kumar T</span>
        <span class="team-badge">👥 Dharshini K</span>
        <span class="team-badge">👥 Allen Xavier K</span>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Sidebar Configuration
st.sidebar.header("⚙️ Pipeline Configuration")

DATASET_OPTIONS = {
    "Demo Dataset (Synthetic)": str(PROJECT_ROOT / "dataset"),
    "Official Competition Dataset": str(WORKSPACE_ROOT / "Dataset" / "student_resource" / "dataset"),
}

selected_dataset_label = st.sidebar.selectbox("Select Dataset", list(DATASET_OPTIONS.keys()), index=0)
data_root_path = DATASET_OPTIONS[selected_dataset_label]

st.sidebar.subheader("Hyperparameters")
threshold_val = st.sidebar.slider("Match Threshold", min_value=0.50, max_value=0.99, value=0.50, step=0.01)
source2_quota = st.sidebar.number_input("Source 2 Top-K Quota", min_value=2, max_value=30, value=8)
source3_quota = st.sidebar.number_input("Source 3 Top-K Quota", min_value=2, max_value=30, value=8)
cardinality_option = st.sidebar.selectbox("Cardinality Mode", ["infer", "one_to_one", "one_to_many", "many_to_many"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Actions")

run_pipeline_btn = st.sidebar.button("🚀 Run Full Pipeline", use_container_width=True, type="primary")
make_demo_btn = st.sidebar.button("🔄 Regenerate Demo Data", use_container_width=True)
check_data_btn = st.sidebar.button("🔍 Check Data Readiness", use_container_width=True)
validate_sub_btn = st.sidebar.button("🛡️ Validate Official Submission", use_container_width=True)


# Helper Functions
def load_json(path: Path) -> dict:
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_tsv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if path.is_file():
        try:
            return pd.read_csv(path, sep="\t", nrows=nrows)
        except Exception as e:
            st.error(f"Error loading {path.name}: {e}")
    return pd.DataFrame()


# Handle Quick Actions
if make_demo_btn:
    with st.spinner("Generating deterministic synthetic demo dataset..."):
        cmd = [sys.executable, "run.py", "--make-demo"]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if res.returncode == 0:
            st.success("✅ Synthetic demo data regenerated successfully!")
        else:
            st.error(f"Failed to generate demo data: {res.stderr}")

if run_pipeline_btn:
    with st.spinner("Executing Entity Resolution Pipeline (Preprocessing → Blocking → Features → Scoring → Decision → Submission)..."):
        cmd = [
            sys.executable,
            "run.py",
            "--data-root",
            data_root_path,
        ]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if res.returncode == 0:
            st.success("✅ Pipeline execution completed successfully!")
            with st.expander("Show Execution Output Log", expanded=False):
                st.code(res.stdout + "\n" + res.stderr)
        else:
            st.error(f"Pipeline execution failed (Exit code {res.returncode}):")
            st.code(res.stderr or res.stdout)

if check_data_btn:
    with st.spinner("Inspecting dataset files, row counts, and schema validation..."):
        cmd = [sys.executable, "run.py", "--check-data", "--data-root", data_root_path]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        st.info("Dataset Health Check Output:")
        st.code(res.stdout or res.stderr)

if validate_sub_btn:
    with st.spinner("Running official submission validator (validate_submission.py)..."):
        validator_path = WORKSPACE_ROOT / "Dataset" / "student_resource" / "utils" / "validate_submission.py"
        matching_path = PROJECT_ROOT / "output" / "submission" / "matching_results.tsv"
        candidate_path = PROJECT_ROOT / "output" / "submission" / "candidate_pairs.tsv"
        test_dir = Path(data_root_path) / "test"

        cmd = [
            sys.executable,
            str(validator_path),
            "--matching",
            str(matching_path),
            "--candidate",
            str(candidate_path),
            "--test-dir",
            str(test_dir),
            "--check-ids",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            st.success("🎉 PASS: Official validator confirmed zero errors! Ready for submission.")
        else:
            st.warning("⚠️ Validator reported notes or issues:")
        st.code(res.stdout or res.stderr)


# Main Tabs Layout
tabs = st.tabs([
    "📊 Pipeline Performance & Metrics",
    "🎯 Matching Results (Submission)",
    "⚡ Candidate Pairs & Blocking",
    "🧠 Model Diagnostics & Features",
    "📋 Raw Dataset Explorer",
])

# Paths to outputs
output_dir = PROJECT_ROOT / "output"
run_metadata_file = output_dir / "run_metadata.json"
diagnostics_file = output_dir / "diagnostics.json"
matching_file = output_dir / "submission" / "matching_results.tsv"
candidate_file = output_dir / "submission" / "candidate_pairs.tsv"
internal_matching_file = output_dir / "matching_results.tsv"
internal_candidate_file = output_dir / "candidate_pairs.tsv"

metadata = load_json(run_metadata_file)
diagnostics = load_json(diagnostics_file)

# TAB 1: Metrics & Overview
with tabs[0]:
    st.subheader("Leaderboard & Validation Performance")
    if metadata:
        val = metadata.get("validation", {})
        test = metadata.get("test", {})
        sub = metadata.get("submission", {})
        train = metadata.get("training", {})

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Macro F₀.₅ Score", f"{val.get('macro_f0_5', 0.0):.4f}")
        c2.metric("Micro Precision", f"{val.get('micro_precision', 0.0):.4f}")
        c3.metric("Micro Recall", f"{val.get('micro_recall', 0.0):.4f}")
        c4.metric("Blocking Candidate Recall", f"{val.get('candidate_recall', 0.0):.4f}")
        c5.metric("Operating Threshold", f"{metadata.get('threshold', 0.5):.3f}")

        st.write("")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        col_t1.metric("Test S1 Entities", f"{test.get('source1_records', 0)}")
        col_t2.metric("Final Matched Entities", f"{sub.get('matched_entities', 0)}")
        col_t3.metric("Singleton Entities (No Match)", f"{sub.get('singleton_entities', 0)}")
        col_t4.metric("Pair Search Reduction", f"{test.get('reduction_percentage', 0.0):.1f}%")

        st.markdown("---")
        st.subheader("Training & Hard-Negative Mining Summary")
        hn1, hn2, hn3, hn4 = st.columns(4)
        hn1.metric("Verified Positives", f"{train.get('positive_count', 0)}")
        hn2.metric("Mined Hard Negatives", f"{train.get('hard_negative_count', 0)}")
        hn3.metric("Hard-Negative Ratio", f"{train.get('hard_negative_ratio', 0.0):.2f}x")
        hn4.metric("Pipeline Runtime", f"{metadata.get('runtime_seconds', 0.0):.2f}s")

        # Architectural Upgrade Badges
        st.subheader("Active Architectural Upgrades")
        upgrades_list = [
            "✅ Upgrade 1: Unicode NFKD Normalization",
            "✅ Upgrade 2: French Commercial Routing (CEDEX/BP)",
            "✅ Upgrade 3: Landmark Extraction",
            "✅ Upgrade 4: Corsica/France Postal Handling",
            "✅ Upgrade 5: Source-Balanced Quotas (8 / 8)",
            "✅ Upgrade 6: Directional Indexing (S2∪S3)",
            "✅ Upgrade 7: Structural Subset Integrity",
            "✅ Upgrade 8: Hard-Negative Mining",
            "✅ Upgrade 9: Tri-State Street Number (+1, 0, -1)",
            "✅ Upgrade 10: Double Metaphone Phonetic Similarity",
            "✅ Upgrade 11: RapidFuzz Token Set Ratio",
            "✅ Upgrade 12: Postal Prefix Agreement",
            "✅ Upgrade 13: Restricted Auto-Accept Rule",
            "✅ Upgrade 14: Multi-Branch Chain Guard",
            "✅ Upgrade 15: Dynamic F0.5 Threshold Sweep",
            "✅ Upgrade 16: Global Conflict Resolution",
        ]
        u_cols = st.columns(4)
        for i, up in enumerate(upgrades_list):
            u_cols[i % 4].markdown(f"**{up}**")
    else:
        st.warning("No previous run metadata found. Click **🚀 Run Full Pipeline** in the sidebar to execute.")

# TAB 2: Matching Results
with tabs[1]:
    st.subheader("Official Submission: Final Entity Matches (`matching_results.tsv`)")
    df_matches = load_tsv(matching_file)
    df_internal_matches = load_tsv(internal_matching_file)

    if not df_matches.empty:
        st.write(f"Showing **{len(df_matches)}** submission records formatted exactly for the portal leaderboard:")
        st.dataframe(df_matches, use_container_width=True, height=280)

        st.subheader("Internal Detailed Match Inspections")
        if not df_internal_matches.empty:
            st.dataframe(df_internal_matches, use_container_width=True, height=280)
    else:
        st.info("No matching results found. Run the pipeline to generate outputs.")

# TAB 3: Candidate Pairs
with tabs[2]:
    st.subheader("Blocking Candidate Set (`candidate_pairs.tsv`)")
    df_cands = load_tsv(candidate_file)
    df_internal_cands = load_tsv(internal_candidate_file)

    if not df_cands.empty:
        st.write(f"Showing **{len(df_cands)}** candidate rows for submission verification:")
        st.dataframe(df_cands, use_container_width=True, height=280)

        st.subheader("Detailed Feature Matrix for Candidate Pairs")
        if not df_internal_cands.empty:
            st.dataframe(df_internal_cands, use_container_width=True, height=350)
    else:
        st.info("No candidate pairs found. Run the pipeline to generate candidate sets.")

# TAB 4: Model Diagnostics
with tabs[3]:
    st.subheader("Model Diagnostic Signals & Error Analysis")
    if diagnostics:
        c1, c2, c3 = st.columns(3)
        c1.metric("Held-out False Positives", diagnostics.get("false_positive_count", 0))
        c2.metric("Held-out False Negatives", diagnostics.get("false_negative_count", 0))
        c3.metric("Blocking Misses", diagnostics.get("candidate_miss_count", 0))

        st.markdown("#### Candidate Distribution by Country")
        country_counts = diagnostics.get("candidate_counts_by_country", {})
        if country_counts:
            st.bar_chart(pd.DataFrame(list(country_counts.items()), columns=["Country", "Candidates"]).set_index("Country"))

        st.markdown("#### Blocker Method Contributions")
        blocker_counts = diagnostics.get("candidate_counts_by_blocking_method", {})
        if blocker_counts:
            st.bar_chart(pd.DataFrame(list(blocker_counts.items()), columns=["Blocker", "Count"]).set_index("Blocker"))
    else:
        st.info("Run the pipeline to populate diagnostic metrics.")

# TAB 5: Raw Dataset Explorer
with tabs[4]:
    st.subheader("Raw TSV Dataset Inspection")
    split_sel = st.radio("Select Partition", ["train", "test"], horizontal=True)
    source_sel = st.selectbox("Select Source File", ["source1", "source2", "source3", "ground_truth" if split_sel == "train" else None])

    if source_sel:
        target_path = Path(data_root_path) / split_sel / f"{split_sel}_{source_sel}.tsv"
        st.write(f"Reading: `{target_path}`")
        df_raw = load_tsv(target_path, nrows=50)
        if not df_raw.empty:
            st.dataframe(df_raw, use_container_width=True)
        else:
            st.warning(f"File {target_path} not found or empty.")
