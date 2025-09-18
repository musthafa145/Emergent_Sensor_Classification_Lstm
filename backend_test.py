#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Sensor Classification LSTM Project
Tests all API endpoints including data generation, model training, predictions, and WebSocket streaming
"""

import requests
import json
import sys
import time
import asyncio
import websockets
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

class SensorLSTMAPITester:
    def __init__(self, base_url="https://sensor-detect-ai.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.ws_url = base_url.replace('https', 'wss') + "/api/stream"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name: str, success: bool, details: str = ""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        result = f"{status} - {name}"
        if details:
            result += f" | {details}"
        
        print(result)
        self.test_results.append({
            'name': name,
            'success': success,
            'details': details
        })
        return success

    def test_health_check(self) -> bool:
        """Test basic health check endpoint"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Status: {data.get('status')}, Model: {data.get('model_status')}"
            else:
                details = f"Status code: {response.status_code}"
                
            return self.log_test("Health Check", success, details)
        except Exception as e:
            return self.log_test("Health Check", False, f"Error: {str(e)}")

    def test_root_endpoint(self) -> bool:
        """Test root API endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Message: {data.get('message')}, Version: {data.get('version')}"
            else:
                details = f"Status code: {response.status_code}"
                
            return self.log_test("Root Endpoint", success, details)
        except Exception as e:
            return self.log_test("Root Endpoint", False, f"Error: {str(e)}")

    def test_generate_training_data(self, samples: int = 1000) -> bool:
        """Test training data generation"""
        try:
            response = requests.post(
                f"{self.api_url}/generate-training-data",
                params={"samples": samples},
                timeout=30
            )
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Generated {data.get('samples')} samples, Activities: {data.get('activities')}"
            else:
                details = f"Status code: {response.status_code}, Response: {response.text[:200]}"
                
            return self.log_test("Generate Training Data", success, details)
        except Exception as e:
            return self.log_test("Generate Training Data", False, f"Error: {str(e)}")

    def test_model_info(self) -> Dict[str, Any]:
        """Test model info endpoint"""
        try:
            response = requests.get(f"{self.api_url}/model-info", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                if data.get('trained'):
                    details = f"Trained: {data.get('trained')}, Classes: {data.get('classes')}, Seq Length: {data.get('sequence_length')}"
                else:
                    details = "Model not trained"
            else:
                details = f"Status code: {response.status_code}"
                data = {}
                
            self.log_test("Model Info", success, details)
            return data
        except Exception as e:
            self.log_test("Model Info", False, f"Error: {str(e)}")
            return {}

    def test_train_model(self) -> bool:
        """Test model training"""
        try:
            training_params = {
                "epochs": 5,  # Reduced for testing
                "batch_size": 32,
                "test_size": 0.2
            }
            
            print("Starting model training (this may take a few minutes)...")
            response = requests.post(
                f"{self.api_url}/train-model",
                json=training_params,
                timeout=300  # 5 minutes timeout
            )
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Status: {data.get('status')}, Accuracy: {data.get('accuracy')}"
            else:
                details = f"Status code: {response.status_code}, Response: {response.text[:200]}"
                
            return self.log_test("Train Model", success, details)
        except Exception as e:
            return self.log_test("Train Model", False, f"Error: {str(e)}")

    def test_training_status(self) -> bool:
        """Test training status endpoint"""
        try:
            response = requests.get(f"{self.api_url}/training-status", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                if data:
                    latest = data[0]
                    details = f"Latest status: {latest.get('status')}, Accuracy: {latest.get('accuracy')}"
                else:
                    details = "No training history found"
            else:
                details = f"Status code: {response.status_code}"
                
            return self.log_test("Training Status", success, details)
        except Exception as e:
            return self.log_test("Training Status", False, f"Error: {str(e)}")

    def test_prediction(self) -> bool:
        """Test prediction endpoint"""
        try:
            # Generate sample sensor data (128 readings with 3 axes each)
            sample_data = []
            for i in range(128):  # Default sequence length
                sample_data.append([
                    np.random.normal(0.2, 0.8),  # x_accel
                    np.random.normal(0.1, 0.6),  # y_accel  
                    np.random.normal(9.8, 0.4)   # z_accel
                ])
            
            prediction_request = {
                "sensor_data": sample_data
            }
            
            response = requests.post(
                f"{self.api_url}/predict",
                json=prediction_request,
                timeout=15
            )
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Predicted: {data.get('predicted_activity')}, Confidence: {data.get('confidence'):.3f}"
            else:
                details = f"Status code: {response.status_code}, Response: {response.text[:200]}"
                
            return self.log_test("Prediction", success, details)
        except Exception as e:
            return self.log_test("Prediction", False, f"Error: {str(e)}")

    async def test_websocket_streaming(self) -> bool:
        """Test WebSocket streaming endpoint"""
        try:
            print("Testing WebSocket streaming...")
            
            # Connect to WebSocket
            async with websockets.connect(self.ws_url) as websocket:
                # Send configuration
                config = {
                    "activity": "walking",
                    "duration": 5  # 5 seconds
                }
                await websocket.send(json.dumps(config))
                
                # Receive data points
                data_points = []
                start_time = time.time()
                
                while time.time() - start_time < 6:  # Wait up to 6 seconds
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        data = json.loads(message)
                        data_points.append(data)
                        
                        if len(data_points) >= 10:  # Got enough data points
                            break
                            
                    except asyncio.TimeoutError:
                        break
                
                success = len(data_points) > 0
                if success:
                    sample_data = data_points[0]
                    has_prediction = 'prediction' in sample_data and sample_data['prediction'] is not None
                    details = f"Received {len(data_points)} data points, Has predictions: {has_prediction}"
                else:
                    details = "No data received from WebSocket"
                    
                return self.log_test("WebSocket Streaming", success, details)
                
        except Exception as e:
            return self.log_test("WebSocket Streaming", False, f"Error: {str(e)}")

    def test_csv_upload(self) -> bool:
        """Test CSV file upload (simulated)"""
        try:
            # Create sample CSV data
            csv_content = """x_accel,y_accel,z_accel,activity
0.2,0.1,9.8,walking
0.3,0.2,9.7,walking
0.1,0.0,9.9,walking
0.4,0.3,9.6,running
0.5,0.4,9.5,running
0.0,0.0,9.8,sitting
0.0,0.0,9.8,sitting"""
            
            files = {'file': ('test_data.csv', csv_content, 'text/csv')}
            
            response = requests.post(
                f"{self.api_url}/upload-sensor-data",
                files=files,
                timeout=15
            )
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Uploaded {data.get('samples')} samples, Activities: {data.get('activities')}"
            else:
                details = f"Status code: {response.status_code}, Response: {response.text[:200]}"
                
            return self.log_test("CSV Upload", success, details)
        except Exception as e:
            return self.log_test("CSV Upload", False, f"Error: {str(e)}")

    async def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Sensor LSTM Backend API Tests")
        print("=" * 60)
        
        # Basic connectivity tests
        self.test_root_endpoint()
        self.test_health_check()
        
        # Data management tests
        self.test_generate_training_data()
        self.test_csv_upload()
        
        # Model info before training
        model_info = self.test_model_info()
        
        # Training tests (only if no model exists)
        if not model_info.get('trained', False):
            print("\n📚 Model not trained, starting training process...")
            self.test_train_model()
            time.sleep(2)  # Wait for training to process
        else:
            print("\n📚 Model already trained, skipping training test")
        
        # Post-training tests
        self.test_training_status()
        self.test_model_info()  # Check again after training
        
        # Prediction tests
        self.test_prediction()
        
        # WebSocket streaming test
        await self.test_websocket_streaming()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        failed_tests = []
        for result in self.test_results:
            if not result['success']:
                failed_tests.append(result)
        
        print(f"Total Tests: {self.tests_run}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  - {test['name']}: {test['details']}")
        else:
            print("\n🎉 ALL TESTS PASSED!")
        
        return self.tests_passed == self.tests_run

def main():
    """Main test execution"""
    tester = SensorLSTMAPITester()
    
    try:
        # Run async tests
        result = asyncio.run(tester.run_all_tests())
        return 0 if result else 1
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())