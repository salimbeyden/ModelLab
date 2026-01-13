# ModelLab - Setup Guide

## 🚀 Quick Start Options

### Option 1: Local Development (Recommended for first-time setup)

**Prerequisites:**
- Python 3.11+
- Node.js 18+
- R (optional, for mgcv models)

**Windows:**
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/ModelLab.git
cd ModelLab

# Run the dev script
.\run_dev.bat
```

**Manual Setup:**
```bash
# Backend
cd backend
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Option 2: Docker Compose (Easiest)

**Prerequisites:** Docker Desktop

```bash
git clone https://github.com/YOUR_USERNAME/ModelLab.git
cd ModelLab
docker-compose up --build
```

Access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## 🌐 Cloud Deployment

### Railway.app (Recommended)

1. Push code to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your ModelLab repo
5. Railway auto-detects docker-compose.yml
6. Get your public URL!

### Render.com

Deploy as two services:
1. **Backend**: Web Service from `./backend` with Dockerfile
2. **Frontend**: Static Site from `./frontend`

---

## 📁 Project Structure

```
ModelLab/
├── backend/           # FastAPI Python backend
│   ├── app/
│   │   ├── api/       # API routes
│   │   ├── core/      # Schemas, config
│   │   ├── plugins/   # Model plugins (EBM, mgcv)
│   │   └── services/  # Business logic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/          # Next.js React frontend
│   ├── app/           # Next.js app router
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🔧 Environment Variables

### Backend
| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | 8000 |
| `DATA_DIR` | Data storage path | ./data |
| `RUNS_DIR` | Model runs path | ./runs |

### Frontend
| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | http://localhost:8000 |

---

## 🐛 Troubleshooting

**Backend won't start:**
```bash
pip install -r requirements.txt --upgrade
```

**Frontend build fails:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**R/mgcv not working:**
- Ensure R is installed and `Rscript` is in PATH
- Install mgcv: `Rscript -e "install.packages('mgcv')"`

---

## 📝 Feedback

Found a bug or have suggestions? Open an issue on GitHub!
