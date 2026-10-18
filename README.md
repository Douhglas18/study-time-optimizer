# Student Study-Time Allocation Optimizer

A Python and Streamlit mini-project that applies linear algebra and Principal Component Analysis (PCA) to student-subject data, then suggests a weekly study-time plan.

## Features

- Enter subjects, difficulty, confidence, days until each exam, and recent scores.
- Analyze student-subject data using matrix operations, rank, Gram–Schmidt orthogonalization, covariance, eigenvalues and eigenvectors, PCA, and projection.
- View recommended study hours and a day-by-day timetable.
- Edit timetable sessions, mark them complete, and download the plan as a CSV file.

## Run locally

Install Python, then open a terminal in this folder and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address shown in the terminal, usually `http://localhost:8501`.

## Data and limitations

The built-in historical records are synthetic examples, not real student data. The priority formula and hour-allocation method are project design choices; they are not prescribed by the project-topic PDF, and the recommendations have not been validated as predictors of exam performance.

## Project context

This project is adapted from the “Academic Performance Principal Pattern Analyzer” topic in the *Mini Projects – Linear Algebra & Vector Calculus* project brief. The app demonstrates a student-data analysis pipeline and extends it with a study-time planning feature.
