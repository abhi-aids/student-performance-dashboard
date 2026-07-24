"""
analysis.py
-----------
Exploratory Data Analysis on the student performance dataset.
Generates summary statistics and saves charts to the visuals/ folder.

Usage:
    python analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("data/student_data.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Total students: {len(df)}")
print(f"Branches: {df['branch'].unique().tolist()}")
print(f"\nPass rate: {(df['result'] == 'Pass').mean() * 100:.1f}%")
print(f"\nGrade distribution:\n{df['grade'].value_counts().sort_index()}")
print(f"\nFinal marks summary:\n{df['final_marks'].describe()}")

# 1. Distribution of final marks
plt.figure(figsize=(8, 5))
sns.histplot(df["final_marks"], bins=25, kde=True, color="#4C72B0")
plt.title("Distribution of Final Marks")
plt.xlabel("Final Marks")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("visuals/marks_distribution.png")
plt.close()

# 2. Study hours vs final marks
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="study_hours_per_week", y="final_marks", hue="result", alpha=0.7)
plt.title("Study Hours vs Final Marks")
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Marks")
plt.tight_layout()
plt.savefig("visuals/study_hours_vs_marks.png")
plt.close()

# 3. Average marks by branch
plt.figure(figsize=(8, 5))
branch_avg = df.groupby("branch")["final_marks"].mean().sort_values(ascending=False)
sns.barplot(x=branch_avg.values, y=branch_avg.index, hue=branch_avg.index, palette="viridis", legend=False)
plt.title("Average Final Marks by Branch")
plt.xlabel("Average Marks")
plt.tight_layout()
plt.savefig("visuals/avg_marks_by_branch.png")
plt.close()

# 4. Attendance vs marks correlation heatmap
plt.figure(figsize=(7, 5))
numeric_cols = ["study_hours_per_week", "attendance_pct", "extracurricular_hours", "final_marks"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("visuals/correlation_heatmap.png")
plt.close()

# 5. Pass/Fail by income bracket
plt.figure(figsize=(8, 5))
pass_rate_income = df.groupby("income_bracket")["result"].apply(lambda x: (x == "Pass").mean() * 100)
pass_rate_income = pass_rate_income.reindex(["Low", "Middle", "High"])
sns.barplot(x=pass_rate_income.index, y=pass_rate_income.values, hue=pass_rate_income.index, palette="Blues_d", legend=False)
plt.title("Pass Rate (%) by Income Bracket")
plt.ylabel("Pass Rate (%)")
plt.tight_layout()
plt.savefig("visuals/pass_rate_by_income.png")
plt.close()

print("\nSaved 5 charts to visuals/")
