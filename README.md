# ModelLab - MVP

ModelLab is a local-first model training and visualization tool supporting EBM (Explainable Boosting Machine) and R mgcv (GAMs).

## Quick Start (Local)

**Prerequisites**: Python 3.11+, Node.js 18+

1. **Setup & Run**:
   Run the helper script (Windows):
   ```
   run_dev.bat
   ```
   This script will:
   - Start the FastAPI Backend on port 8000
   - Start the Next.js Frontend on port 3000

2. **Manual Setup**:
   
   *Backend*:
   ```bash
   cd backend
   python -m venv venv
   .\venv\Scripts\activate
   pip install -e .
   python -m uvicorn app.main:app --reload
   ```

   *Frontend*:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Features
- **Dataset Upload**: CSV/Parquet support.
- **Profiling**: Basic column statistics.
- **Model Plugins**:
  - **EBM**: Classification/Regression with global explanations.
  - **mgcv**: R-based GAMs (requires Rscript in PATH).
- **Run Orchestration**: Local background execution.

## Verification
To run the backend verification script:
```
verify_env.bat
```
