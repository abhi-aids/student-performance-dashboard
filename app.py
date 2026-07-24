"""
app.py
------
Student Performance Analytics Dashboard (Streamlit)

Run with:
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/student_data.csv")

df = load_data()

# ---------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------
st.sidebar.header("🔍 Filters")

branches = st.sidebar.multiselect(
    "Branch", options=sorted(df["branch"].unique()), default=sorted(df["branch"].unique())
)
years = st.sidebar.multiselect(
    "Year", options=sorted(df["year"].unique()), default=sorted(df["year"].unique())
)
income = st.sidebar.multiselect(
    "Income Bracket", options=sorted(df["income_bracket"].unique()), default=sorted(df["income_bracket"].unique())
)

filtered = df[
    df["branch"].isin(branches) & df["year"].isin(years) & df["income_bracket"].isin(income)
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(filtered)}** of {len(df)} students")

# ---------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------
st.title("🎓 Student Performance Analytics Dashboard")
st.caption("Exploratory analysis + a simple pass/fail prediction model on student academic data.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", len(filtered))
col2.metric("Average Marks", f"{filtered['final_marks'].mean():.1f}")
col3.metric("Pass Rate", f"{(filtered['result'] == 'Pass').mean() * 100:.1f}%")
col4.metric("Avg Attendance", f"{filtered['attendance_pct'].mean():.1f}%")

st.markdown("---")

# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🔗 Correlations", "🏆 Branch Comparison", "🤖 Predict Pass/Fail"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            filtered, x="final_marks", nbins=25, color="result",
            title="Distribution of Final Marks", barmode="overlay"
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        grade_counts = filtered["grade"].value_counts().reindex(["A+", "A", "B", "C", "D", "F"]).fillna(0)
        fig = px.bar(
            x=grade_counts.index, y=grade_counts.values,
            title="Grade Distribution", labels={"x": "Grade", "y": "Count"}
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered, x="study_hours_per_week", y="final_marks", color="result",
        hover_data=["branch", "year"], title="Study Hours vs Final Marks"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    numeric_cols = ["study_hours_per_week", "attendance_pct", "extracurricular_hours", "final_marks"]
    corr = filtered[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        filtered, x="income_bracket", y="final_marks", color="income_bracket",
        category_orders={"income_bracket": ["Low", "Middle", "High"]},
        title="Final Marks by Income Bracket"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    branch_avg = filtered.groupby("branch")["final_marks"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(
        branch_avg, x="final_marks", y="branch", orientation="h",
        title="Average Final Marks by Branch", color="final_marks", color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)

    subject_cols = ["Mathematics", "Programming", "Data Structures", "Electronics", "Communication Skills"]
    subj_avg = filtered.groupby("branch")[subject_cols].mean().reset_index()
    subj_avg_melt = subj_avg.melt(id_vars="branch", var_name="Subject", value_name="Average Marks")
    fig = px.bar(
        subj_avg_melt, x="branch", y="Average Marks", color="Subject", barmode="group",
        title="Subject-wise Average Marks by Branch"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Logistic Regression: Predicting Pass/Fail")
    st.caption("A simple baseline model trained on study hours, attendance, and extracurricular load.")

    model_df = df.copy()
    le_result = LabelEncoder()
    model_df["result_encoded"] = le_result.fit_transform(model_df["result"])  # Fail=0, Pass=1

    features = ["study_hours_per_week", "attendance_pct", "extracurricular_hours"]
    X = model_df[features]
    y = model_df["result_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    colA, colB = st.columns([1, 1])
    with colA:
        st.metric("Model Accuracy (test set)", f"{acc * 100:.1f}%")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm, text_auto=True, x=["Predicted Fail", "Predicted Pass"], y=["Actual Fail", "Actual Pass"],
            color_continuous_scale="Blues", title="Confusion Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("#### Try it yourself")
        sh = st.slider("Study hours per week", 0, 35, 12)
        att = st.slider("Attendance %", 40, 100, 75)
        extra = st.slider("Extracurricular hours per week", 0, 15, 3)

        pred_input = pd.DataFrame([[sh, att, extra]], columns=features)
        pred_prob = clf.predict_proba(pred_input)[0][1]
        pred_label = "Pass ✅" if pred_prob >= 0.5 else "Fail ❌"

        st.metric("Predicted Result", pred_label)
        st.progress(float(pred_prob))
        st.caption(f"Predicted probability of passing: {pred_prob * 100:.1f}%")

st.markdown("---")
st.caption("Built with Streamlit, Plotly, and scikit-learn · Synthetic dataset for demonstration purposes")
