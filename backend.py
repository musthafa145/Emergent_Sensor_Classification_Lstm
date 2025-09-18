from fastapi import FastAPI, UploadFile, File, WebSocket
from pydantic import BaseModel
from typing import List
import uvicorn
import random
import asyncio
from fastapi.responses import JSONResponse
from threading import Thread
import time

app = FastAPI()

# ====== Data Models ======
class SensorData(BaseModel):
    sensor_data: List[List[float]]

# ====== Training Status ======
training_status = {"status": None}

# ====== Endpoints ======
@app.get("/api")
def root():
    return {"message": "Sensor LSTM API running", "version": "0.1"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "model_status": "not_loaded"}

@app.get("/api/model-info")
def model_info():
    return {"trained": False, "classes": [], "sequence_length": 128}

@app.post("/api/predict")
def predict(data: SensorData):
    return {"predicted_activity": "unknown", "confidence": 0.0}

@app.post("/api/upload-sensor-data")
def upload_csv(file: UploadFile = File(...)):
    return {"samples": 0, "activities": []}

# ====== Generate Training Data ======
@app.post("/api/generate-training-data")
def generate_training_data(samples: int = 1000):
    activities = ["walking", "running", "jumping"]
    data = [{"acc_x": random.uniform(-1,1),
             "acc_y": random.uniform(-1,1),
             "acc_z": random.uniform(-1,1),
             "activity": random.choice(activities)} for _ in range(samples)]
    return JSONResponse(content={"samples": len(data), "activities": activities})

# ====== Train Model ======
def train_model_thread():
    global training_status
    training_status["status"] = "training"
    time.sleep(5)  # simulate training
    training_status["status"] = "trained"

@app.post("/api/train-model")
def train_model():
    if training_status["status"] == "training":
        return {"status": "already_training"}
    Thread(target=train_model_thread).start()
    return {"status": "training_started"}

# ====== Training Status ======
@app.get("/api/training-status")
def get_training_status():
    return JSONResponse(status_code=200, content=[{"status": training_status["status"], "accuracy": None}])

# ====== WebSocket Streaming ======
@app.websocket("/api/stream")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    for i in range(5):
        data = {"acc_x": random.random(),
                "acc_y": random.random(),
                "acc_z": random.random(),
                "prediction": random.choice(["walking","running","jumping"])}
        await ws.send_json(data)
        await asyncio.sleep(1)
    await ws.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
