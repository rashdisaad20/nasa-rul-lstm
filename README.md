
# API key-value: a1ed63f24809c7b5
# NASA RUL LSTM

## Overview
This project trains an LSTM model to predict Remaining Useful Life (RUL) for NASA turbofan engine data.

## Files
- `src/train.py` - main training script
- `src/dataset.py` - data loading and sequence preparation
- `src/model.py` - LSTM regression model
- `app.py` - FastAPI inference server
- `requirements.txt` - required Python dependencies

## Setup
Use the `nasa_venv` conda environment or any Python 3.11 environment with the listed dependencies.

```powershell
conda activate nasa_venv
python -m pip install -r requirements.txt
```

## Run training
From the project root:

```powershell
conda activate nasa_venv
python -m src.train
```

This will create:
- `models/lstm_weights.pth`
- `mlflow.db`

## Run the API
After training, start the API server:

```powershell
conda activate nasa_venv
uvicorn app:app --reload
```

Or run directly:

```powershell
python app.py
```

Then POST to `/predict` with a JSON payload containing a `sequence` of shape `[30, 24]`.

A simple web interface is available at `/ui` after starting the app, where you can paste a 30×24 JSON sequence and submit it from your browser.

Example request body:

```json
{
  "sequence": [
    [0.1, 0.2, ..., 0.24],
    ..., 
    30 rows total
  ]
}
```

Example Python client:

```python
import requests

payload = {
    "sequence": [[0.1] * 24] * 30
}
response = requests.post("http://127.0.0.1:8000/predict", json=payload)
print(response.json())
```

## Docker
Build and run the container:

```powershell
docker build -t nasa-rul-lstm .
docker run -p 8000:8000 nasa-rul-lstm
```

### Production Docker image
A production-optimized Dockerfile is available at `Dockerfile.prod`.

```powershell
docker build -f Dockerfile.prod -t nasa-rul-lstm:prod .
docker run -p 8000:8000 nasa-rul-lstm:prod
```

You can also use Docker Compose:

```powershell
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## Environment variables
A `.env.example` file is provided for recommended configuration values.

```text
API_KEY=your_api_key_here
PREDICTION_DB_PATH=app.db
SSL_CERT_PATH=/path/to/fullchain.pem
SSL_KEY_PATH=/path/to/privkey.pem
```

## GitHub Actions
A GitHub Actions workflow is included at `.github/workflows/python-app.yml`. It runs the test suite and builds the production Docker image on push or pull request to `main`/`master`.

## Security & HTTPS
You can enable simple API-key authentication by setting the `API_KEY` environment variable before starting the server. When `API_KEY` is set, all protected endpoints (`/predict`, `/predictions`) require the `x-api-key` header with the matching value.

Example (Linux/macOS):

```bash
export API_KEY=mysecretkey
uvicorn app:app --reload
```

To run with TLS directly using `app.py`, provide `SSL_CERT_PATH` and `SSL_KEY_PATH` environment variables pointing to your certificate and key files. Example:

```bash
export SSL_CERT_PATH=/path/to/fullchain.pem
export SSL_KEY_PATH=/path/to/privkey.pem
python app.py
```

Note: For production, prefer terminating TLS at a reverse proxy (nginx, AWS ALB) and manage secrets with a vault or platform service.

## Tests
Run the test suite from the project root:

```powershell
pytest
```

The test files cover:
- API health check and request validation
- dataset loading and preprocessing

## Notes
- `src/train.py` uses a local SQLite MLflow backend by default to avoid remote tracking configuration issues.
- `app.py` loads the locally saved model weights.
