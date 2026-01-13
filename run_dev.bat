@echo off
cd backend
echo Starting Backend...
start "ModelLab Backend" .\venv\Scripts\python.exe -m uvicorn app.main:app --reload

cd ..\frontend
echo Starting Frontend...
start "ModelLab Frontend" npm run dev

echo ModelLab is starting!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
pause
