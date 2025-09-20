import requests
import json
import base64
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base URL for the API
BASE_URL = 'http://localhost:5000'

def test_health_endpoint():
    """Test the health check endpoint"""
    response = requests.get(f'{BASE_URL}/api/health')
    print(f"Health Check Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("\n" + "-"*50 + "\n")

def test_verify_endpoint_text():
    """Test the verify endpoint with text input"""
    payload = {
        'input_type': 'text',
        'content': 'The Earth is flat and vaccines cause autism. Climate change is a hoax.'
    }
    
    response = requests.post(f'{BASE_URL}/api/verify', json=payload)
    print(f"Verify Endpoint (Text) Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Claims extracted:")
        for idx, claim_result in enumerate(result.get('results', [])):
            print(f"\nClaim {idx+1}: {claim_result.get('claim')}")
            print(f"Score: {claim_result.get('score', {}).get('confidence_label')} ({claim_result.get('score', {}).get('confidence_score')}/100)")
            print(f"Explanation: {claim_result.get('explanation', {}).get('summary')}")
    else:
        print(f"Error: {response.text}")
    
    print("\n" + "-"*50 + "\n")

def test_verify_endpoint_url():
    """Test the verify endpoint with URL input"""
    payload = {
        'input_type': 'url',
        'content': 'https://www.who.int/news-room/questions-and-answers/item/vaccines-and-immunization-vaccine-safety'
    }
    
    response = requests.post(f'{BASE_URL}/api/verify', json=payload)
    print(f"Verify Endpoint (URL) Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("Claims extracted:")
        for idx, claim_result in enumerate(result.get('results', [])):
            print(f"\nClaim {idx+1}: {claim_result.get('claim')}")
            print(f"Score: {claim_result.get('score', {}).get('confidence_label')} ({claim_result.get('score', {}).get('confidence_score')}/100)")
    else:
        print(f"Error: {response.text}")
    
    print("\n" + "-"*50 + "\n")

def test_feedback_endpoint():
    """Test the feedback endpoint"""
    payload = {
        'claim_id': 'test_claim_123',
        'feedback_type': 'agree',
        'comments': 'This verification was accurate and helpful.'
    }
    
    response = requests.post(f'{BASE_URL}/api/feedback', json=payload)
    print(f"Feedback Endpoint Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("\n" + "-"*50 + "\n")

def run_all_tests():
    """Run all API tests"""
    print("\n===== TESTING VERIFYSENSE API =====\n")
    
    # Test health endpoint
    print("Testing Health Endpoint...")
    test_health_endpoint()
    
    # Test verify endpoint with text
    print("Testing Verify Endpoint with Text Input...")
    test_verify_endpoint_text()
    
    # Test verify endpoint with URL
    print("Testing Verify Endpoint with URL Input...")
    test_verify_endpoint_url()
    
    # Test feedback endpoint
    print("Testing Feedback Endpoint...")
    test_feedback_endpoint()
    
    print("\n===== ALL TESTS COMPLETED =====\n")

if __name__ == "__main__":
    run_all_tests()