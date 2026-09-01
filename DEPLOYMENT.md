# CodeAsk Deployment Guide

This guide covers all options for deploying CodeAsk into production, from one-click containerized setups with Docker to cloud hosting (Vercel, Render, Railway, Fly.io, Cloud Run).

---

## 📋 Required API Keys & Environment Variables

Before deploying, ensure you have credentials for both AI providers:

| Variable | Description | Required | Where to Obtain |
| :--- | :--- | :---: | :--- |
| `DEEPSEEK_API_KEY` | Powers code reasoning and LangGraph agent | **Yes** | [DeepSeek API Portal](https://platform.deepseek.com) |
| `GEMINI_API_KEY` | Powers Gemini Embedding 2 (`gemini-embedding-2`) | **Yes** | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins (e.g., frontend domain) | **Yes** | Your deployed frontend URL or `*` |
| `VITE_API_URL` | Frontend pointer to backend API URL | **Yes** | Your deployed backend URL |
| `PORT` | Dynamic port binding (default: `8000`) | Optional | Auto-set by cloud providers |

---

## 🚀 Option 1: Vercel (Frontend) + Render / Railway (Backend) — *Recommended*

This is the most cost-effective and scalable setup: static frontend on high-speed global CDN (Vercel) paired with a persistent containerized backend (Render or Railway).

### Step 1: Deploy Backend (Render)
1. Fork or push this repository to GitHub.
2. Log into [Render Dashboard](https://dashboard.render.com/) and click **New +** → **Blueprint** (or **Web Service**).
   - If using **Blueprint**, connect your repo and Render will automatically parse [`render.yaml`](render.yaml).
   - If creating manually as a **Web Service**:
     - **Root Directory:** `backend`
     - **Environment:** `Python` or `Docker`
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
     - **Health Check Path:** `/health`
3. In **Environment Variables**, configure:
   - `DEEPSEEK_API_KEY` = `your_deepseek_key`
   - `GEMINI_API_KEY` = `your_gemini_key`
   - `ALLOWED_ORIGINS` = `https://your-frontend.vercel.app` (or `*` during initial testing)
4. Deploy and copy your backend service URL (e.g., `https://codeask-backend.onrender.com`).

---

### Step 2: Deploy Frontend (Vercel)
1. Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **Add New...** → **Project**.
2. Select your `codeask` repository.
3. In **Project Settings**:
   - **Framework Preset:** `Vite`
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. In **Environment Variables**, add:
   - `VITE_API_URL` = `https://codeask-backend.onrender.com` (your Render backend URL from Step 1)
5. Click **Deploy**. Your UI will be live at `https://your-project.vercel.app`.

---

## 🐳 Option 2: Docker Compose (Single Command / VPS)

Deploy the entire stack (FastAPI backend + Vite/Nginx frontend + persistent ChromaDB volume) to any Linux VM (AWS EC2, DigitalOcean Droplet, Hetzner, GCP VM).

### 1. Clone & Configure
```bash
git clone https://github.com/juslookin/codeask.git
cd codeask

# Copy and edit environment variables
cp .env.example .env
nano .env  # Fill in DEEPSEEK_API_KEY and GEMINI_API_KEY
```

### 2. Launch Stack
```bash
docker compose up -d --build
```

### 3. Verify
- **Frontend UI:** Open `http://<your-server-ip>:3000`
- **Backend API Docs:** Open `http://<your-server-ip>:8000/docs`
- **Health Check:** `curl http://localhost:8000/health`

### 4. Stopping / Restarting
```bash
docker compose down
docker compose logs -f
```

---

## ☁️ Option 3: Railway Full-Stack Deployment

1. Install Railway CLI or visit [railway.app](https://railway.app).
2. Create a new project from your GitHub repository.
3. Add a service for `backend`:
   - Set root directory to `backend`.
   - Add environment variables `DEEPSEEK_API_KEY` and `GEMINI_API_KEY`.
   - Generate domain (e.g. `https://codeask-backend.up.railway.app`).
4. Add a service for `frontend`:
   - Set root directory to `frontend`.
   - Set environment variable `VITE_API_URL` = `https://codeask-backend.up.railway.app`.

---

## 🪶 Option 4: Fly.io Container Deployment

### Backend on Fly.io
```bash
cd backend
fly launch --dockerfile Dockerfile --name codeask-backend

# Set secrets
fly secrets set DEEPSEEK_API_KEY="your-key" GEMINI_API_KEY="your-key" ALLOWED_ORIGINS="*"

# Deploy
fly deploy
```

---

## 🔍 Verification & Health Checks

Once deployed, verify the endpoints:
1. **API Status:**
   ```bash
   curl https://your-backend-url/health
   # Returns: {"status": "healthy"}
   ```
2. **API Documentation:**
   Visit `https://your-backend-url/docs` to inspect interactive OpenAPI / Swagger docs.
3. **Repository Ingestion Test:**
   ```bash
   curl -X POST https://your-backend-url/ingest \
     -H "Content-Type: application/json" \
     -d '{"github_url": "https://github.com/pallets/flask"}'
   ```

---

## 🛠️ Production Best Practices & Notes

1. **Git Dependency:**
   The backend uses `GitPython` to clone repositories during ingestion. Both the Dockerfile and Render/Railway environments have `git` available.
2. **Rate Limits & API Quotas:**
   - Gemini Embedding 2 has generous free/paid tier limits. The embedder includes built-in exponential backoff for `429 RESOURCE_EXHAUSTED` responses.
   - DeepSeek-V4 Flash is used for high-speed streaming answers with lower cost and high throughput.
3. **ChromaDB Storage:**
   - In Docker Compose, ChromaDB data persists in the `chroma_data` volume.
   - In ephemeral container environments (Cloud Run without volume mounts), ingested collections exist for the container lifetime. For multi-tenant persistent storage, mount a persistent volume (e.g., Render Disks, Railway Volume, AWS EFS).
