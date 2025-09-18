#!/usr/bin/env python3
"""
Quick Backend API Test for Sensor Classification LSTM Project
"""

import requests
import json
import sys
import time

def test_api_endpoint(url, method="GET", data=None, timeout=10):
    """Test a single API endpoint"""
    try:
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            if data:
                response = requests.post(url, json=data, timeout=timeout)
            else:
                response = requests.post(url, timeout=timeout)
        
        print(f"{method} {url}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response: {json.dumps(result, indent=2)[:200]}...")
            except:
                print(f"Response: {response.text[:200]}...")
        else:
            print(f"Error: {response.text[:200]}...")
        print("-" * 50)
        return response.status_code == 200
    except Exception as e:
        print(f"Exception: {str(e)}")
        print("-" * 50)
        return False

def main():
    base_url = "https://sensor-detect-ai.preview.emergentagent.com/api"
    
    print("🚀 Quick Backend API Tests")
    print("=" * 50)
    
    # Test basic endpoints
    tests = [
        ("Root", f"{base_url}/", "GET"),
        ("Health", f"{base_url}/health", "GET"),
        ("Model Info", f"{base_url}/model-info", "GET"),
        ("Training Status", f"{base_url}/training-status", "GET"),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, url, method in tests:
        print(f"Testing {name}...")
        if test_api_endpoint(url, method):
            passed += 1
            print("✅ PASS")
        else:
            print("❌ FAIL")
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    return passed == total

if __name__ == "__main__":
    sys.exit(0 if main() else 1)