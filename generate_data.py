"""
generate_data.py
-----------------
Generates a synthetic but realistic student performance dataset.
Run this once to create data/student_data.csv, which the dashboard reads.

Usage:
    python generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 600  # number of students

genders = np.random.choice(["Male", "Female"], size=N, p=[0.55, 0.45])
branches = np.random.choice(
    ["AI & DS", "Computer Engineering", "IT", "E&TC", "Mechanical", "Civil"],
    size=N,
    p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.1],
)
years = np.random.choice(["FE", "SE", "TE", "BE"], size=N, p=[0.3, 0.3, 0.2, 0.2])

# Study hours per week — influences performance
study_hours = np.clip(np.random.normal(12, 5, N), 1, 35)

# Attendance % — also influences performance, mildly correlated with study hours
attendance = np.clip(
    60 + study_hours * 1.3 + np.random.normal(0, 8, N), 40, 100
)

# Extracurricular involvement (hours/week) — slight negative effect on marks if excessive
extracurricular = np.clip(np.random.exponential(3, N), 0, 15)

# Family income bracket (categorical, for exploratory segmentation only)
income_bracket = np.random.choice(
    ["Low", "Middle", "High"], size=N, p=[0.35, 0.45, 0.20]
)

# Internet access at home (affects online resource usage)
internet_access = np.random.choice(["Yes", "No"], size=N, p=[0.85, 0.15])

# Base score model: weighted combination + noise
base_score = (
    15
    + study_hours * 2.1
    + attendance * 0.25
    - extracurricular * 0.8
    + np.where(internet_access == "Yes", 4, -3)
    + np.random.normal(0, 12, N)
)

final_marks = np.clip(base_score, 0, 100).round(1)

# Derive letter grade
def grade(m):
    if m >= 85:
        return "A+"
    elif m >= 75:
        return "A"
    elif m >= 65:
        return "B"
    elif m >= 50:
        return "C"
    elif m >= 35:
        return "D"
    else:
        return "F"

grades = [grade(m) for m in final_marks]
pass_fail = np.where(final_marks >= 35, "Pass", "Fail")

# Subject-wise marks (correlated with final_marks but with subject-specific noise)
subjects = ["Mathematics", "Programming", "Data Structures", "Electronics", "Communication Skills"]
subject_scores = {}
for subj in subjects:
    noise = np.random.normal(0, 6, N)
    subject_scores[subj] = np.clip(final_marks + noise, 0, 100).round(1)

df = pd.DataFrame(
    {
        "student_id": [f"STU{i+1:04d}" for i in range(N)],
        "gender": genders,
        "branch": branches,
        "year": years,
        "study_hours_per_week": study_hours.round(1),
        "attendance_pct": attendance.round(1),
        "extracurricular_hours": extracurricular.round(1),
        "income_bracket": income_bracket,
        "internet_access": internet_access,
        **subject_scores,
        "final_marks": final_marks,
        "grade": grades,
        "result": pass_fail,
    }
)

df.to_csv("data/student_data.csv", index=False)
print(f"Generated {len(df)} student records -> data/student_data.csv")
print(df.head())
