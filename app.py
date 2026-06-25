# Importing libraries
from datetime import datetime, timezone # Import date and time according to timzezone
import json 
import os
import sqlite3

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch
import numpy as np
from src.model import LSTMRULRegressor

app = FastAPI(title="NASA Turbofan Engine RUL Prediction API", version="1.0")

# Load model weights onto memory structure during bootstrap phase
INPUT_DIM = 24  
HIDDEN_DIM = 64
NUM_LAYERS = 2
DB_PATH = os.getenv("PREDICTION_DB_PATH", "app.db")
API_KEY = os.getenv("API_KEY")

device = torch.device('cpu')
model = LSTMRULRegressor(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
MODEL_LOADED = False

try:
    weights_path = "models/lstm_weights.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        MODEL_LOADED = True
        print(f"SUCCESS: Model weights loaded from {weights_path}")
    else:
        print(f"WARNING: Weights not found at {weights_path}. Prediction endpoint will be disabled.")
except Exception as e:
    print(f"ERROR: Failed to load model weights: {e}")


def init_db(path: str):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            sequence_json TEXT NOT NULL,
            prediction REAL NOT NULL,
            unit_status TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn

_db_conn = init_db(DB_PATH)

class InferencePayload(BaseModel):
    # Expects a list of rows containing the 24 scaled columns over a lookback sequence length of 30
    sequence: list[list[float]]


def verify_api_key(x_api_key: str | None = Header(None)) -> bool:
    """Verify the incoming `x-api-key` header if `API_KEY` is set in env.
    If `API_KEY` is not set, this check is a no-op to preserve local testing convenience.
    """
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key")
    return True

@app.get("/")
def health_check():
    status = "healthy" if MODEL_LOADED else "model_unavailable"
    auth_enabled = bool(API_KEY)
    return {"status": status, "model": "LSTM RUL Regressor", "model_loaded": MODEL_LOADED, "auth_enabled": auth_enabled}

@app.post("/predict")
def predict_rul(payload: InferencePayload, _auth: bool = Depends(verify_api_key)):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model weights are not loaded. Please ensure models/lstm_weights.pth exists.")

    try:
        input_data = np.array(payload.sequence, dtype=np.float32)

        # Verify sequence structural bounds [30 timesteps, 24 features]
        if input_data.shape != (30, 24):
            raise HTTPException(
                status_code=400,
                detail=f"Inference window expects matrix shape (30, 24). Received: {input_data.shape}"
            )

        # Reshape to match batch format requirements [1, 30, 24]
        tensor_input = torch.tensor(input_data).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(tensor_input)
            predicted_rul = float(prediction.item())
            unit_status = "Critical Attention Required" if predicted_rul < 30 else "Normal Operational Status"

        _db_conn.execute(
            "INSERT INTO prediction_log (created_at, sequence_json, prediction, unit_status) VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(payload.sequence), predicted_rul, unit_status),
        )
        _db_conn.commit()

        return {
            "predicted_remaining_useful_life": round(predicted_rul, 2),
            "unit_status": unit_status,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
def get_predictions(limit: int = 20, _auth: bool = Depends(verify_api_key)):
    rows = _db_conn.execute(
        "SELECT id, created_at, prediction, unit_status FROM prediction_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [{"id": row["id"], "created_at": row["created_at"], "prediction": row["prediction"], "unit_status": row["unit_status"]} for row in rows]

@app.get("/ui", response_class=HTMLResponse)
def web_ui():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NASA Turbofan RUL Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 font-sans leading-normal tracking-normal">
        <nav class="bg-blue-900 p-4 shadow-lg">
            <div class="container mx-auto flex justify-between items-center">
                <span class="text-white text-xl font-bold">🚀 NASA Engine RUL Predictor</span>
                <div class="text-blue-200 text-sm">Deep Learning Model: LSTM</div>
            </div>
        </nav>

        <div class="container mx-auto mt-8 p-4">
            <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <!-- Sidebar Controls -->
                <div class="lg:col-span-1 bg-white p-6 rounded-lg shadow-md h-fit">
                    <h2 class="text-lg font-semibold mb-4 border-b pb-2">Control Panel</h2>
                    <div class="mb-4">
                        <label class="block text-gray-700 text-sm font-bold mb-2">API Authentication</label>
                        <input id="apiKey" type="password" placeholder="Enter API Key" class="w-full p-2 border rounded text-sm mb-2">
                    </div>
                    <div class="space-y-2">
                        <button onclick="loadSampleData()" class="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition">
                            Load Sample Data
                        </button>
                        <button onclick="sendTablePrediction()" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition">
                            Run Prediction
                        </button>
                        <button onclick="fillZeros()" class="w-full bg-gray-200 hover:bg-gray-300 text-gray-700 font-bold py-2 px-4 rounded transition text-sm">
                            Reset to Zeros
                        </button>
                    </div>

                    <div id="resultCard" class="mt-6 hidden p-4 rounded-lg border">
                        <p class="text-xs uppercase font-bold text-gray-500">Prediction Result</p>
                        <div id="rulValue" class="text-3xl font-black mt-1">--</div>
                        <div id="statusLabel" class="text-xs font-bold mt-2 px-2 py-1 rounded-full text-center"></div>
                    </div>
                </div>

                <!-- Data Table -->
                <div class="lg:col-span-3 bg-white p-6 rounded-lg shadow-md">
                    <h2 class="text-lg font-semibold mb-4 border-b pb-2">Input Sequence (30 Timesteps × 24 Features)</h2>
                    <div class="overflow-x-auto max-h-[600px] border border-gray-200 rounded">
                        <table id="sequenceTable" class="min-w-full text-xs">
                            <thead class="bg-gray-50 sticky top-0">
                                <tr>
                                    <th class="p-2 border">Step</th>
                                    """
    for c in range(1, 25):
        html += f'<th class="p-2 border">F{c}</th>'
    html += """
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-gray-200">
    """
    for r in range(1, 31):
        html += f'<tr><td class="p-1 border bg-gray-50 font-bold text-center">{r}</td>'
        for _ in range(24):
            html += '<td class="p-0 border"><input type="text" value="0.0" class="w-12 p-1 text-center outline-none focus:bg-blue-50 transition"></td>'
        html += "</tr>"
    
    html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function fillZeros() {
                document.querySelectorAll('#sequenceTable input').forEach(i => i.value = '0.0');
            }

            function loadSampleData() {
                document.querySelectorAll('#sequenceTable input').forEach(i => {
                    i.value = (Math.random() * 0.5 + 0.2).toFixed(4);
                });
            }

            function buildSequence() {
                const rows = [];
                document.querySelectorAll('#sequenceTable tbody tr').forEach(tr => {
                    const values = Array.from(tr.querySelectorAll('input')).map(i => {
                        const n = parseFloat(i.value);
                        if (isNaN(n)) throw new Error('Values must be numbers');
                        return n;
                    });
                    rows.push(values);
                });
                return rows;
            }

            async function sendTablePrediction() {
                const resultCard = document.getElementById('resultCard');
                const rulVal = document.getElementById('rulValue');
                const statusLab = document.getElementById('statusLabel');
                const apiKey = document.getElementById('apiKey').value.trim();
                
                try {
                    const sequence = buildSequence();
                    const headers = {'Content-Type': 'application/json'};
                    if(apiKey) headers['x-api-key'] = apiKey;

                    const res = await fetch('/predict', { method: 'POST', headers, body: JSON.stringify({sequence}) });
                    const data = await res.json();

                    resultCard.classList.remove('hidden');
                    if(res.ok) {
                        rulVal.innerText = data.predicted_remaining_useful_life;
                        rulVal.className = "text-3xl font-black mt-1 " + (data.predicted_remaining_useful_life < 30 ? "text-red-600" : "text-green-600");
                        statusLab.innerText = data.unit_status;
                        statusLab.className = "text-xs font-bold mt-2 px-2 py-1 rounded-full text-center " + (data.predicted_remaining_useful_life < 30 ? "bg-red-100 text-red-800" : "bg-green-100 text-green-800");
                    } else {
                        alert("Error: " + (data.detail || "Unknown error"));
                    }
                } catch (err) {
                    alert(err.message);
                }
            }
        </script>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    import uvicorn

    # Conditional SSL support if environment variables are provided
    ssl_cert = os.getenv("SSL_CERT_PATH")
    ssl_key = os.getenv("SSL_KEY_PATH")

    # Default to 0.0.0.0 to allow access from other devices on your local network
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn_kwargs = {"host": host, "port": port, "reload": True}
    if ssl_cert and ssl_key:
        uvicorn_kwargs.update({"ssl_certfile": ssl_cert, "ssl_keyfile": ssl_key})

    uvicorn.run("app:app", **uvicorn_kwargs)
