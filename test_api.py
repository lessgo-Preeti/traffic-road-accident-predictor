"""
Test script for the Traffic Accident Prediction API
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test /health endpoint"""
    print("\n" + "="*50)
    print("Testing /health endpoint")
    print("="*50)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_info():
    """Test /info endpoint"""
    print("\n" + "="*50)
    print("Testing /info endpoint")
    print("="*50)
    try:
        response = requests.get(f"{BASE_URL}/info", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_schema():
    """Test /schema endpoint"""
    print("\n" + "="*50)
    print("Testing /schema endpoint")
    print("="*50)
    try:
        response = requests.get(f"{BASE_URL}/schema", timeout=5)
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Total Features: {len(data.get('features', []))}")
        print(f"Categorical Features: {data.get('categorical_features', [])}")
        print(f"Numerical Features Count: {len(data.get('numerical_features', []))}")
        print(f"\nExample Input:")
        print(json.dumps(data.get('example', {}), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_predict():
    """Test /predict endpoint with sample data"""
    print("\n" + "="*50)
    print("Testing /predict endpoint")
    print("="*50)
    
    # Sample input based on the expected schema
    sample_input = {
        "hour": 18,
        "day_of_week": 4,
        "month": 12,
        "is_weekend": 0,
        "is_rush_hour": 1,
        "is_night": 0,
        "is_monsoon": 1,
        "weather_main": "Rain",
        "road_type": "arterial",
        "is_junction": 1,
        "is_urban": 1,
        "is_highway": 0,
        "Year": 2023,
        "Number of Vehicles Involved": 2,
        "Number of Casualties": 1,
        "Number of Fatalities": 0,
        "Speed Limit (km/h)": 60,
        "Driver Age": 35,
        "year": 2023
    }
    
    try:
        print("Input Data:")
        print(json.dumps(sample_input, indent=2))
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample_input,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_predict_missing_features():
    """Test /predict endpoint with missing features (should return 400)"""
    print("\n" + "="*50)
    print("Testing /predict endpoint with missing features")
    print("="*50)
    
    incomplete_input = {
        "hour": 18,
        "day_of_week": 4
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=incomplete_input,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 400  # Should return error
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "#"*50)
    print("# Traffic Accident Prediction API Tests")
    print("#"*50)
    
    # Wait a moment for server to be ready
    print("\nWaiting for server to be ready...")
    time.sleep(2)
    
    results = {
        "health": test_health(),
        "info": test_info(),
        "schema": test_schema(),
        "predict": test_predict(),
        "predict_error": test_predict_missing_features()
    }
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

if __name__ == "__main__":
    main()
