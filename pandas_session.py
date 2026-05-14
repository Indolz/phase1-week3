import pandas as pd
import numpy as np

# ── Creating s DataFrame from scratch ───────────────────────────────────────────────
# Think of it as a dictionary where each key is a column name
# and the value is a list of that column's data

students = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [20, 22, 21, 23, 20],
    "score": [85, 92, 78, 96, 88],
    "passed": [True, True, False, True, True]
})

print("Our DataFrame:")
print(students)
print()
print("Shape:", students.shape)
print("Columns:", students.columns.tolist())
print("Data types:\n", students.dtypes)

# # ── First look at any dataset — always run these three ─────────────
print("\n--- First 3 rows: ---")
print(students.head(3))

print("\n--- Summary Statistics ---")
print(students.describe())

print("\n--- Quick info ---")
students.info()

print("\n" + "-"*50)
print("BLOCK 2 - Selecting, Filtering, Adding Columns")
print("-"*50)

# ── Selecting columns ──────────────────────────────────────────────
print("One column (Series):")
print(students["score"])

print("\nTwo columns (DataFrame):")
print(students[["name", "score"]])

# ── Filtering rows by condition ────────────────────────────────────
print("\nStudents who passed:")
print(students[students["passed"] == True])

print("\nStudents with score above 85:")
high_scorers = students[students["score"] > 85]
print(high_scorers)

# ── Adding a new column ─────────────────────────────────────────────
# Grade based on score
def assign_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "F"
    
students["grade"] = students["score"].apply(assign_grade)
print("\nWith grades added:")
print(students)

# ── Sorting ─────────────────────────────────────────────────────────
print("\nSorted by score (highest first):")
print(students.sort_values("score", ascending=False))

print("\n" + "="*50)
print("BLOCK 3 - groupby and Missing data")
print("="*50)

# ── groupby ───────────────────────────────────────────────────────
# Real world dataset — students, subjects, scores
results = pd.DataFrame({
    "student": ["Alice", "Bob", "Alice", "Bob", "Alice", "Bob"],
    "subject": ["Maths", "Maths", "English", "English", "Science", "Science"],
    "score": [85, 92, 78, 88, 90, 76]
})

print("Raw results:")
print(results)

# Average score per subject
print("\nAverage score per subject:")
print(results.groupby("subject")["score"].mean().round(1))

# Average score per student
print("\nAverage score per student:")
print(results.groupby("student")["score"].mean().round(1))

# Multiple stats at once
print("\nFull stats per subject:")
print(results.groupby("subject")["score"].agg(["mean", "min", "max"]))

# ── Missing data ──────────────────────────────────────────────────
# Real datasets always have missing values — this is how you handle them
data_with_gaps = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "score": [85, None, 78, 96],    # None becomes NaN in pandas
    "age": [20, 22, None, 23]
})

print("\nDataFrame with missing values:")
print(data_with_gaps)

# Check where values are missing
print("\nMissing values per column:")
print(data_with_gaps.isnull().sum())

# Fill missing values with the column mean
mean_score = data_with_gaps["score"].mean()
data_with_gaps["score"] = data_with_gaps["score"].fillna(mean_score)
print(f":\nFilled missing score with mean ({mean_score:.1f}):")
print(data_with_gaps)

# Drop rows that still have any missing values
clean_data = data_with_gaps.dropna()
print("\nAfter dropping rows with missing values:")
print(clean_data)

print("\n" + "="*50)
print("BLOCK 4 - Real CSV Data Analysis")
print("="*50)

# ── Load a real CSV file ──────────────────────────────────────────
df = pd.read_csv("students_data.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# ── Explore the data ──────────────────────────────────────────────
print("\n--- Summary statistics ---")
print(df.describe().round(1))

# ── Answer real questions with groupby ────────────────────────────
print("\n--- Average score per subject ---")
print(df.groupby("subject")["score"].mean().round(1))

print("\n--- Average hours studied per grade ---")
print(df.groupby("grade")["hours_studied"].mean().round(1))

print("\n--- Best student overall ---")
student_avg = df.groupby("name")["score"].mean().round(1)
print(student_avg.sort_values(ascending=False))

# ── Find patterns ─────────────────────────────────────────────────
print("\n--- Correlation: hours studied vs score ---")
correlation = df["hours_studied"].corr(df["score"])
print(f"Correlation coefficient: {correlation:.3f}")
print("(1.0 = perfect positive, 0 = no relationship, -1 = perfect negative)")

# ── Filter and export ─────────────────────────────────────────────
top_students = df[df["score"] >= 90]
print(f"\nStudents scoring 90+: {len(top_students)} results")
print(top_students[["name", "subject", "score"]].sort_values("score",ascending=False))

# Save results to a new CSV
top_students.to_csv("top_students.csv", index=False)
print("\nSaved top students to top_students.csv")