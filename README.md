# Student Performance Predictor

A beginner-friendly machine learning learning project that predicts a student's final score from study-related features.

> **Learning project:** Understand the code, run it, experiment with it, and improve it before presenting it as your own original work.

## What this project teaches

- Python data handling
- CSV files
- Feature selection
- Train/test splitting
- Linear Regression with scikit-learn
- Model evaluation with Mean Absolute Error (MAE)
- Making predictions for new data

## Features

The model uses:
- `study_hours`
- `attendance`
- `previous_score`

Target:
- `final_score`

## Run

```bash
pip install -r requirements.txt
python train.py
python predict.py
```

## Structure

```text
student-performance-predictor/
├── data/
│   └── student_data.csv
├── train.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Future improvements

- Add more data
- Compare Linear Regression with Random Forest
- Add graphs
- Build a Streamlit interface
