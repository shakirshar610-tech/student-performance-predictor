import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/student_data.csv")

X = df[["study_hours", "attendance", "previous_score"]]
y = df["final_score"]

model = LinearRegression()
model.fit(X, y)

new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [90],
    "previous_score": [75],
})

prediction = model.predict(new_student)[0]
print(f"Predicted final score: {prediction:.2f}")
