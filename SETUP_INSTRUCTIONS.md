# 🚀 NASA RUL LSTM - Quick Setup Guide

## ✅ What Was Fixed

This is a **CLEAN version** with all deployment errors resolved:

- ✅ Removed exposed `.env` file with credentials
- ✅ Updated `requirements.txt` with pytest
- ✅ Fixed GitHub Actions workflow (downloads all data files)
- ✅ Updated `.gitignore` properly
- ✅ Removed database files (will regenerate)
- ✅ Cleaned up pycache
- ✅ Created `.env.example` for reference

## 🔧 Setup on Your Computer (Windows/Mac/Linux)

### Step 1: Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Data Files
```bash
# Create data folder if needed
mkdir -p data

# Windows (PowerShell)
curl.exe -L "https://dagshub.com/rashdisaad20/nasa-rul-lstm/raw/main/data/train_FD001.txt" -o data/train_FD001.txt
curl.exe -L "https://dagshub.com/rashdisaad20/nasa-rul-lstm/raw/main/data/test_FD001.txt" -o data/test_FD001.txt
curl.exe -L "https://dagshub.com/rashdisaad20/nasa-rul-lstm/raw/main/data/RUL_FD001.txt" -o data/RUL_FD001.txt

# Mac/Linux
curl -L "https://dagshub.com/rashdisaad20/nasa-rul-lstm/raw/main/data/train_FD001.txt" -o data/train_FD001.txt
curl -L "https://dagshub.com/rashdisaad20/nasa-rul-lstm/raw/main/data/test_FD001.txt" -o data/test_FD001.txt
curl -L "https://dagshub.com/rashdisaad20/nasa-rul-lstm/raw/main/data/RUL_FD001.txt" -o data/RUL_FD001.txt
```

### Step 4: Create .env File
Copy `.env.example` to `.env` and fill with YOUR real credentials:

```bash
# Copy the example
cp .env.example .env

# Edit .env with your real values:
API_KEY=your_real_api_key
DAGSHUB_USERNAME=rashdisaad20
DAGSHUB_TOKEN=your_real_dagshub_token
MLFLOW_TRACKING_URI=https://dagshub.com/rashdisaad20/nasa-rul-lstm.mlflow
```

### Step 5: Run Tests (Optional)
```bash
pytest -v
```

### Step 6: Run the App
```bash
python app.py
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 7: Open in Browser
```
http://localhost:8000/ui
```

You'll see the beautiful dashboard! 🎨

---

## 🚀 Deploy to GitHub

### Step 1: Initialize Git (if needed)
```bash
git init
git add .
git commit -m "initial commit: NASA RUL LSTM project"
```

### Step 2: Create GitHub Repository
1. Go to https://github.com/new
2. Create a new repository named `nasa-rul-lstm`
3. Copy the commands and run them:

```bash
git remote add origin https://github.com/YOUR_USERNAME/nasa-rul-lstm.git
git branch -M main
git push -u origin main
```

### Step 3: Add GitHub Secrets
1. Go to your GitHub repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add these secrets (one by one):

| Name | Value |
|------|-------|
| `DAGSHUB_USERNAME` | your DagsHub username |
| `DAGSHUB_TOKEN` | your DagsHub token |
| `API_KEY` | your API key |

⚠️ **DO NOT paste credentials directly in code!** Use GitHub Secrets!

### Step 4: Push to GitHub
```bash
git push origin main
```

### Step 5: Monitor Workflow
1. Go to your GitHub repo → Actions tab
2. Watch the workflow run
3. You should see:
   - ✅ Install Python
   - ✅ Install dependencies
   - ✅ Download data files
   - ✅ Run tests
   - ✅ Build Docker image

All should be green! ✅

---

## ⚠️ Important Notes

1. **`.env` is IGNORED by git**: Never commit credentials!
2. **Data files**: Downloaded automatically by GitHub Actions
3. **Model weights**: Will be created when training or loaded from `models/`
4. **Database**: Created automatically at runtime

---

## 🆘 Troubleshooting

### "pytest: command not found"
```bash
pip install pytest
```

### "curl command not found"
You're on an old system. Try:
```bash
python -m pip install wget
wget https://dagshub.com/.../train_FD001.txt -O data/train_FD001.txt
```

### "Port 8000 already in use"
```bash
python app.py --port 8001
# Or open http://localhost:8001/ui
```

### "Model weights not found"
The model will train on first run, or copy `lstm_weights.pth` to `models/` folder

---

## 📁 Project Structure
```
nasa-rul-lstm/
├── app.py                          # Main FastAPI application
├── ui.py                           # UI components
├── requirements.txt                # Dependencies
├── .env.example                    # Example environment file
├── .gitignore                      # Git ignore rules
├── .github/
│   └── workflows/
│       └── python-app.yml         # GitHub Actions workflow
├── src/
│   ├── model.py                   # LSTM model
│   ├── dataset.py                 # Data loading
│   ├── train.py                   # Training script
│   └── __init__.py
├── tests/
│   ├── test_app.py
│   ├── test_dataset.py
│   ├── test_predict_integration.py
│   └── conftest.py
├── models/
│   └── lstm_weights.pth           # Model weights (Git LFS)
├── data/
│   ├── train_FD001.txt           # Training data (DVC)
│   ├── test_FD001.txt            # Test data (DVC)
│   └── RUL_FD001.txt             # RUL labels (DVC)
├── config/
│   └── config.yaml               # Configuration
├── Dockerfile                     # Docker build file
└── README.md                      # Project readme
```

---

## ✅ Success Checklist

- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Data files downloaded to `data/` folder
- [ ] `.env` file created with real credentials
- [ ] Tests pass: `pytest -v`
- [ ] App runs: `python app.py`
- [ ] Dashboard loads: http://localhost:8000/ui
- [ ] GitHub repo created
- [ ] GitHub Secrets added
- [ ] Code pushed to GitHub
- [ ] GitHub Actions workflow passes

---

## 🎉 You're Ready!

Your project is now:
- ✅ Error-free
- ✅ Secure (no exposed credentials)
- ✅ Deployable (GitHub Actions configured)
- ✅ Tested (pytest ready)
- ✅ Production-ready

Happy coding! 🚀

---

**Last Updated**: 2026-06-25
**Status**: FIXED ✅
