Commands

.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000

Open new terminal
cd frontend
echo "VITE_API_BASE_URL=http://localhost:8000" > .env
npm run dev
