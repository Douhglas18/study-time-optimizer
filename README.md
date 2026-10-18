# Student Study-Time Allocation Optimizer

An educational Streamlit prototype that analyzes synthetic student-subject records and recommends a weekly study-hour allocation.

## Run locally

1. Install Python 3.10 or newer.
2. In this folder, install dependencies: `python -m pip install -r requirements.txt`
3. Start the app: `streamlit run app.py`

## Data and math

The demonstration dataset is generated in the app and contains no real student information. The computational pipeline forms and standardizes a feature matrix, calculates rank, builds an orthonormal basis with modified Gram–Schmidt, performs covariance eigenanalysis/PCA, projects current records, and uses distance from a higher-performing reference profile as one input to priority. The resulting priority scores are normalized to allocate weekly hours. Daily availability then turns those allocations into a day-by-day timetable in 30-minute blocks. You can move sessions between days and subjects, mark them complete, review a progress bar, and download the edited plan as CSV.

Matrix, rank, independence, Gram–Schmidt, orthogonality, eigenanalysis, PCA, and projection are concepts named in the supplied project brief. The profile-distance priority formula and proportional hour allocation are proposed project adaptations, not formulas prescribed by the brief. Replace the synthetic records with approved data and document verification before submission.
