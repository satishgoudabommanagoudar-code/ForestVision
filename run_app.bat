@echo off
echo Starting ForestVision Backend and Frontend...
start cmd /k "cd backend && uvicorn app.main:app --reload"
start cmd /k "streamlit run frontend/app.py"
