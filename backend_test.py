#!/usr/bin/env python3
import requests
import json
import sys
import time
import asyncio
import websockets
import numpy as np

class SensorLSTMAPITester:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.api_url = f"{base_url}/api"
        self.ws_url = f"ws://127.0.0.1:8000/api/stream"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"{status} - {name} | {details}")
        self.test_results.append({'name': name,'success': success,'details': details})
        return success

    def test_root_endpoint(self):
        try:
            r = requests.get(f"{self.api_url}/")
            return self.log_test("Root Endpoint", r.status_code==200,
                                 f"Message: {r.json().get('message')}, Version: {r.json().get('version')}")
        except Exception as e:
            return self.log_test("Root Endpoint", False, str(e))

    def test_health_check(self):
        try:
            r = requests.get(f"{self.api_url}/health")
            return self.log_test("Health Check", r.status_code==200,
                                 f"Status: {r.json().get('status')}, Model: {r.json().get('model_status')}")
        except Exception as e:
            return self.log_test("Health Check", False, str(e))

    def test_generate_training_data(self):
        try:
            r = requests.post(f"{self.api_url}/generate-training-data", params={"samples":1000})
            return self.log_test("Generate Training Data", r.status_code==200,
                                 f"Generated {r.json().get('samples')} samples, Activities: {r.json().get('activities')}")
        except Exception as e:
            return self.log_test("Generate Training Data", False, str(e))

    def test_model_info(self):
        try:
            r = requests.get(f"{self.api_url}/model-info")
            trained = r.json().get("trained", False)
            details = "Model not trained" if not trained else f"Trained: {r.json().get('trained')}"
            self.log_test("Model Info", True, details)
            return r.json()
        except Exception as e:
            self.log_test("Model Info", False, str(e))
            return {}

    def test_train_model(self):
        try:
            r = requests.post(f"{self.api_url}/train-model", json={"epochs":5,"batch_size":32,"test_size":0.2})
            return self.log_test("Train Model", r.status_code==200,
                                 f"Status: {r.json().get('status')}, Accuracy: {r.json().get('accuracy')}")
        except Exception as e:
            return self.log_test("Train Model", False, str(e))

    def test_training_status(self):
        try:
            r = requests.get(f"{self.api_url}/training-status")
            data = r.json()[0]
            details = f"Latest status: {data.get('status')}, Accuracy: {data.get('accuracy')}"
            return self.log_test("Training Status", True, details)
        except Exception as e:
            return self.log_test("Training Status", False, str(e))

    def test_prediction(self):
        try:
            sample_data = [[np.random.random(), np.random.random(), np.random.random()] for _ in range(128)]
            r = requests.post(f"{self.api_url}/predict", json={"sensor_data": sample_data})
            return self.log_test("Prediction", r.status_code==200,
                                 f"Predicted: {r.json().get('predicted_activity')}, Confidence: {r.json().get('confidence'):.3f}")
        except Exception as e:
            return self.log_test("Prediction", False, str(e))

    async def test_websocket_streaming(self):
        try:
            async with websockets.connect(self.ws_url) as ws:
                data_points=[]
                for _ in range(5):
                    msg = await ws.recv()
                    data_points.append(json.loads(msg))
                has_prediction = "prediction" in data_points[0]
                return self.log_test("WebSocket Streaming", len(data_points)>0,
                                     f"Received {len(data_points)} data points, Has predictions: {has_prediction}")
        except Exception as e:
            return self.log_test("WebSocket Streaming", False, str(e))

    def test_csv_upload(self):
        try:
            csv_content = "x_accel,y_accel,z_accel,activity\n0.1,0.2,9.8,walking"
            files = {'file': ('test.csv', csv_content,'text/csv')}
            r = requests.post(f"{self.api_url}/upload-sensor-data", files=files)
            return self.log_test("CSV Upload", r.status_code==200,
                                 f"Uploaded {r.json().get('samples')} samples, Activities: {r.json().get('activities')}")
        except Exception as e:
            return self.log_test("CSV Upload", False, str(e))

    async def run_all_tests(self):
        print("🚀 Starting Sensor LSTM Backend API Tests")
        self.test_root_endpoint()
        self.test_health_check()
        self.test_generate_training_data()
        self.test_csv_upload()
        model_info = self.test_model_info()
        if not model_info.get('trained', False):
            print("\n📚 Model not trained, starting training process...")
            self.test_train_model()
            time.sleep(2)
        self.test_training_status()
        self.test_model_info()
        self.test_prediction()
        await self.test_websocket_streaming()
        print(f"\nTotal Tests: {self.tests_run}, Passed: {self.tests_passed}, Failed: {self.tests_run - self.tests_passed}, Success Rate: {self.tests_passed/self.tests_run*100:.1f}%")

def main():
    tester = SensorLSTMAPITester()
    asyncio.run(tester.run_all_tests())

if __name__=="__main__":
    main()
