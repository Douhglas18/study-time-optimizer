from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


FEATURES = ["Study hours", "Difficulty", "Confidence", "Days to exam", "Recent score"]


def sample_data() -> pd.DataFrame:
    """Small synthetic class dataset for demonstrating the analysis pipeline."""
    rng = np.random.default_rng(17)
    subjects = ["Mathematics", "Physics", "Chemistry", "Computer Science", "English"]
    rows = []
    for learner in range(1, 31):
        for subject_index, subject in enumerate(subjects):
            difficulty = float(np.clip([4.5, 4.0, 3.8, 3.4, 2.8][subject_index] + rng.normal(0, 0.65), 1, 5))
            confidence = float(np.clip(6.0 - difficulty + rng.normal(0, 0.9), 1, 5))
            study_hours = float(np.clip(7.5 - difficulty * 0.6 + confidence * 0.65 + rng.normal(0, 1.1), 0.5, 12))
            days_to_exam = int(np.clip(rng.integers(2, 31) + (subject_index % 2) * 2, 1, 40))
            score = float(np.clip(48 + study_hours * 3.2 + confidence * 3.5 - difficulty * 2 + rng.normal(0, 7), 25, 98))
            rows.append({"Learner": f"Learner {learner:02d}", "Subject": subject, "Study hours": study_hours,
                         "Difficulty": difficulty, "Confidence": confidence, "Days to exam": days_to_exam,
                         "Recent score": score})
    return pd.DataFrame(rows)


def modified_gram_schmidt(matrix: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, list[int]]:
    """Return orthonormal columns and original feature indices kept."""
    basis: list[np.ndarray] = []
    kept: list[int] = []
    for j in range(matrix.shape[1]):
        vector = matrix[:, j].astype(float).copy()
        for q in basis:
            vector -= np.dot(q, vector) * q
        norm = np.linalg.norm(vector)
        if norm > tol:
            basis.append(vector / norm)
            kept.append(j)
    if not basis:
        return np.empty((matrix.shape[0], 0)), []
    return np.column_stack(basis), kept


def run_pipeline(history: pd.DataFrame, current: pd.DataFrame) -> dict:
    x = history[FEATURES].astype(float).to_numpy()
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale

    rank = int(np.linalg.matrix_rank(z))
    q, independent = modified_gram_schmidt(z)
    covariance = z.T @ z / max(1, z.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    keep = max(1, min(3, int(np.sum(eigenvalues > 1e-10))))
    components = eigenvectors[:, :keep]

    current_x = current[FEATURES].astype(float).to_numpy()
    current_z = (current_x - mean) / scale
    projected = current_z @ components

    # Adaptation for this project: distance from the projected class profile of
    # stronger recent outcomes contributes to each subject's risk priority.
    performance_cut = history["Recent score"].quantile(0.70)
    strong = z[history["Recent score"].to_numpy() >= performance_cut]
    reference = (strong.mean(axis=0) @ components) if len(strong) else np.zeros(keep)
    spread = np.sqrt(np.maximum(eigenvalues[:keep], 1e-9))
    profile_distance = np.sqrt(np.mean(((projected - reference) / spread) ** 2, axis=1))
    urgency = 1 / np.maximum(current["Days to exam"].to_numpy(dtype=float), 1)
    low_confidence = (6 - current["Confidence"].to_numpy(dtype=float)) / 5
    difficulty = current["Difficulty"].to_numpy(dtype=float) / 5
    score_gap = (100 - current["Recent score"].to_numpy(dtype=float)) / 100
    # Explicit project-specific heuristic, not a formula prescribed by the PDF.
    priority = 0.40 * profile_distance + 0.25 * low_confidence + 0.20 * difficulty + 0.10 * score_gap + 0.05 * urgency
    priority = np.maximum(priority, 1e-6)

    result = current[["Subject", *FEATURES]].copy()
    result["Projected profile distance"] = profile_distance
    result["Priority score"] = priority
    return {"rank": rank, "q": q, "independent": independent, "eigenvalues": eigenvalues,
            "explained": eigenvalues / max(eigenvalues.sum(), 1e-12), "components": components,
            "projected": projected, "result": result, "mean": mean, "scale": scale}


st.set_page_config(page_title="Study-Time Allocation Optimizer", page_icon="📚", layout="wide")
st.title("📚 Student Study-Time Allocation Optimizer")
st.caption("A student-data analysis pipeline that turns study patterns into an explainable weekly plan.")
st.info("The built-in class records are synthetic examples. Replace them with permitted data for your project demonstration.")

with st.sidebar:
    st.header("Your weekly availability")
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_hours = {day: st.number_input(day, min_value=0.0, max_value=12.0,
                                        value=[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 2.0][i],
                                        step=0.5, key=f"hours_{day}") for i, day in enumerate(day_names)}
    weekly_hours = sum(daily_hours.values())
    st.metric("Total available this week", f"{weekly_hours:g} hours")
    st.caption("Hours are distributed across subjects by priority, then arranged into daily sessions.")

history = sample_data()
st.subheader("Your subjects")
default_current = pd.DataFrame({
    "Subject": ["Mathematics", "Physics", "Chemistry", "Computer Science", "English"],
    "Study hours": [4.0, 3.0, 3.0, 4.0, 2.0],
    "Difficulty": [5.0, 4.0, 4.0, 3.0, 2.0],
    "Confidence": [2.0, 3.0, 3.0, 4.0, 4.0],
    "Days to exam": [10, 14, 18, 21, 25],
    "Recent score": [58.0, 66.0, 63.0, 78.0, 82.0],
})
current = st.data_editor(default_current, num_rows="dynamic", use_container_width=True,
                         column_config={"Difficulty": st.column_config.NumberColumn(min_value=1, max_value=5),
                                        "Confidence": st.column_config.NumberColumn(min_value=1, max_value=5),
                                        "Days to exam": st.column_config.NumberColumn(min_value=1, max_value=365),
                                        "Recent score": st.column_config.NumberColumn(min_value=0, max_value=100)})

if current.empty or current["Subject"].astype(str).str.strip().eq("").any():
    st.warning("Add at least one subject and give each subject a name.")
    st.stop()

try:
    current[FEATURES] = current[FEATURES].apply(pd.to_numeric, errors="raise")
    if current[FEATURES].isna().any().any():
        raise ValueError("Fill in all numeric subject fields.")
    if ((current["Difficulty"] < 1) | (current["Difficulty"] > 5)).any() or ((current["Confidence"] < 1) | (current["Confidence"] > 5)).any():
        raise ValueError("Difficulty and confidence must be between 1 and 5.")
    if (current["Days to exam"] < 1).any() or ((current["Recent score"] < 0) | (current["Recent score"] > 100)).any():
        raise ValueError("Days to exam must be at least 1; recent score must be between 0 and 100.")
except (ValueError, TypeError) as exc:
    st.error(f"Please correct the subject data: {exc}")
    st.stop()

analysis = run_pipeline(history, current)
result = analysis["result"]
result["Recommended hours"] = weekly_hours * result["Priority score"] / result["Priority score"].sum()
result["Recommended hours"] = result["Recommended hours"].round(1)
# Adjust rounding on the final subject so the displayed total matches availability.
if len(result):
    result.loc[result.index[-1], "Recommended hours"] += round(weekly_hours - result["Recommended hours"].sum(), 1)

st.subheader("Recommended weekly schedule")
st.dataframe(result[["Subject", "Priority score", "Recommended hours", "Projected profile distance"]].sort_values("Recommended hours", ascending=False),
             hide_index=True, use_container_width=True)
st.bar_chart(result.set_index("Subject")[["Recommended hours"]])
st.caption("Priority combines projected distance from the higher-performing historical profile with difficulty, confidence, score gap, and exam urgency. The weights are a project design choice, not a formula specified in the PDF.")

st.subheader("Day-by-day timetable")
remaining = {row["Subject"]: float(row["Recommended hours"]) for _, row in result.iterrows()}
priority_lookup = result.set_index("Subject")["Priority score"].to_dict()
schedule_rows = []
for day, available in daily_hours.items():
    day_remaining = float(available)
    # Fill each day in 30-minute blocks, favoring subjects with the most
    # remaining recommended hours weighted by their priority.
    while day_remaining >= 0.5 and any(hours >= 0.5 for hours in remaining.values()):
        candidates = [subject for subject, hours in remaining.items() if hours >= 0.5]
        subject = max(candidates, key=lambda s: remaining[s] * priority_lookup[s])
        block = min(0.5, day_remaining, remaining[subject])
        if block < 0.5:
            break
        schedule_rows.append({"Day": day, "Subject": subject, "Study time": f"{block:g} h", "Completed": False})
        remaining[subject] -= block
        day_remaining -= block
schedule = pd.DataFrame(schedule_rows, columns=["Day", "Subject", "Study time", "Completed"])
if schedule.empty:
    st.info("Set at least half an hour of availability on one or more days to generate a timetable.")
else:
    st.caption("Edit the day or subject to fit your routine, then tick sessions as you finish them.")
    editor_key = "schedule_editor_" + str((tuple(current["Subject"].astype(str)), tuple(daily_hours.values())))
    edited_schedule = st.data_editor(
        schedule,
        hide_index=True,
        use_container_width=True,
        key=editor_key,
        column_config={
            "Day": st.column_config.SelectboxColumn("Day", options=day_names, required=True),
            "Subject": st.column_config.SelectboxColumn("Subject", options=current["Subject"].astype(str).tolist(), required=True),
            "Study time": st.column_config.TextColumn("Study time", disabled=True),
            "Completed": st.column_config.CheckboxColumn("Completed", default=False),
        },
        disabled=["Study time"],
    )
    completed_count = int(edited_schedule["Completed"].fillna(False).sum())
    session_count = len(edited_schedule)
    st.progress(completed_count / session_count if session_count else 0.0,
                text=f"{completed_count} of {session_count} sessions completed")
    allocated_by_day = edited_schedule.assign(_hours=edited_schedule["Study time"].str.replace(" h", "", regex=False).astype(float)).groupby("Day")["_hours"].sum()
    overloaded = {day: float(hours - daily_hours[day]) for day, hours in allocated_by_day.items() if hours > daily_hours[day] + 1e-9}
    if overloaded:
        details = ", ".join(f"{day}: {allocated_by_day[day]:g} h scheduled / {daily_hours[day]:g} h available" for day in overloaded)
        st.warning("The edited timetable exceeds availability on: " + details)
    elif any(hours >= 0.5 for hours in remaining.values()):
        st.warning("Your weekly availability is less than the recommended subject allocation. Increase daily availability to include all recommended sessions.")
    else:
        st.success("The timetable fits within your daily availability.")
    st.download_button("Download timetable as CSV", edited_schedule.to_csv(index=False),
                       file_name="weekly_study_timetable.csv", mime="text/csv")

with st.expander("See the mathematics pipeline"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Feature matrix rank", f"{analysis['rank']} / {len(FEATURES)}")
    c2.metric("Independent directions found", str(len(analysis["independent"])))
    c3.metric("PCA components retained", str(analysis["components"].shape[1]))
    st.markdown("**1. Matrix and standardization.** Historical student-subject records form a matrix. Each feature is centered and scaled so units such as hours, days, and scores can be compared.")
    st.markdown("**2. Rank and independence.** Rank reports the number of independent feature directions. Modified Gram–Schmidt constructs an orthonormal basis from independent columns.")
    st.markdown("**3. Eigenanalysis and PCA.** The covariance matrix is decomposed into eigenvalues and eigenvectors. The retained eigenvectors define principal directions.")
    st.markdown("**4. Projection and allocation.** Current subject records are projected onto those directions. Their distance from the stronger-outcome reference profile contributes to priority; priorities are normalized to divide the available hours.")
    st.write("Explained variance by component:", [round(float(v), 3) for v in analysis["explained"][:analysis["components"].shape[1]]])
    st.write("Orthonormal basis shape:", analysis["q"].shape)
    st.write("Projection coordinates:")
    projection_table = pd.DataFrame(analysis["projected"], columns=[f"PC{i+1}" for i in range(analysis["projected"].shape[1])])
    projection_table.insert(0, "Subject", current["Subject"].to_numpy())
    st.dataframe(projection_table, hide_index=True, use_container_width=True)

st.subheader("Project notes")
st.markdown("- **PDF-supported concepts:** matrices, rank, linear independence, Gram–Schmidt, orthogonality, eigenvalues/eigenvectors, PCA, and projection appear in the student analytics and integrated project descriptions.\n- **Proposed adaptation:** using the projected profile distance to prioritize subjects, then distributing hours proportionally, is our design choice. It should be described as an application of the PDF concepts, not quoted as a prescribed formula.\n- **Verification for the report:** compare the matrix rank and eigenvalues with an independent numerical calculation, check that basis vectors are orthonormal, and confirm projected coordinates and hour totals on a hand-worked example.")
