import requests
import json
import time
import sys

# Test configuration
BASE_URL = "http://localhost:5000"
TEST_URLS = [
    "https://www.reuters.com/world/us/trump-says-he-would-not-protect-nato-members-who-dont-pay-bills-2024-02-11/",
    "https://www.bbc.com/news/health-68274002",
    "https://www.theguardian.com/environment/2024/feb/12/climate-crisis-extreme-weather-el-nino"
]
TEST_CONTENT = [
    "COVID-19 vaccines contain microchips to track people.",
    "Climate change is a hoax created by scientists for funding.",
    "Drinking water with lemon every morning can cure cancer."
]

def test_url_verification():
    """Test verification with different URLs"""
    print("\n=== Testing URL Verification ===")
    
    for url in TEST_URLS:
        print(f"\nTesting URL: {url}")
        
        payload = {
            "url": url,
            "type": "text"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/verify", json=payload)
            response_data = response.json()
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Status: {response_data.get('status')}")
            
            if response.status_code == 200:
                results = response_data.get('results', [])
                print(f"Claims detected: {len(results)}")
                
                for i, result in enumerate(results):
                    print(f"\nClaim {i+1}: {result.get('claim')}")
                    print(f"Confidence: {result.get('score', {}).get('confidence_label')}")
                    print(f"Score: {result.get('score', {}).get('score')}")
                    
                    # Print score components
                    components = result.get('score', {}).get('components', {})
                    if components:
                        print("Score Components:")
                        for component, value in components.items():
                            print(f"  - {component}: {value}")
                    
                    # Check for notifications
                    notification = result.get('notification')
                    if notification:
                        print(f"Notification: {notification.get('message')}")
            else:
                print(f"Error: {response_data.get('message')}")
                
        except Exception as e:
            print(f"Error during test: {str(e)}")
        
        # Wait between requests to avoid overwhelming the server
        time.sleep(1)

def test_content_verification():
    """Test verification with direct content input"""
    print("\n=== Testing Content Verification ===")
    
    for content in TEST_CONTENT:
        print(f"\nTesting Content: {content}")
        
        payload = {
            "content": content,
            "type": "text"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/api/verify", json=payload)
            response_data = response.json()
            
            print(f"Status Code: {response.status_code}")
            print(f"Response Status: {response_data.get('status')}")
            
            if response.status_code == 200:
                results = response_data.get('results', [])
                print(f"Claims detected: {len(results)}")
                
                for i, result in enumerate(results):
                    print(f"\nClaim {i+1}: {result.get('claim')}")
                    print(f"Confidence: {result.get('score', {}).get('confidence_label')}")
                    print(f"Score: {result.get('score', {}).get('score')}")
                    
                    # Print score components
                    components = result.get('score', {}).get('components', {})
                    if components:
                        print("Score Components:")
                        for component, value in components.items():
                            print(f"  - {component}: {value}")
                    
                    # Check for notifications
                    notification = result.get('notification')
                    if notification:
                        print(f"Notification: {notification.get('message')}")
            else:
                print(f"Error: {response_data.get('message')}")
                
        except Exception as e:
            print(f"Error during test: {str(e)}")
        
        # Wait between requests to avoid overwhelming the server
        time.sleep(1)

def test_image_verification():
    """Test verification with image input (should return notification)"""
    print("\n=== Testing Image Verification ===")
    
    payload = {
        "content": "base64_encoded_image_data_would_go_here",
        "type": "image"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/verify", json=payload)
        response_data = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Status: {response_data.get('status')}")
        print(f"Message: {response_data.get('message')}")
        
        # Check feature status
        feature_status = response_data.get('feature_status', {})
        if feature_status:
            print("Feature Status:")
            for feature, status in feature_status.items():
                print(f"  - {feature}: {status}")
                
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    print("VerifySense API Testing")
    print("======================")
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "url":
            test_url_verification()
        elif test_type == "content":
            test_content_verification()
        elif test_type == "image":
            test_image_verification()
        else:
            print(f"Unknown test type: {test_type}")
    else:
        # Run all tests
        test_url_verification()
        test_content_verification()
        test_image_verification()
    
    print("\nTesting completed!")