import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Fact Check Tools API endpoint
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def check_facts(claim):
    """
    Check a claim against Google's Fact Check Tools API
    
    Args:
        claim (str): The claim to check
        
    Returns:
        list: A list of fact check results
    """
    try:
        # Get API key from environment variables
        api_key = os.getenv('FACT_CHECK_API_KEY')
        
        if not api_key:
            print("Warning: FACT_CHECK_API_KEY not found in environment variables")
            return []
        
        # Prepare parameters for the API request
        params = {
            'key': api_key,
            'query': claim,
            'languageCode': 'en-US'
        }
        
        # Make the API request
        response = requests.get(FACT_CHECK_API_URL, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant information from the response
            fact_checks = []
            if 'claims' in data:
                for item in data['claims']:
                    fact_check = {
                        'claim_date': item.get('claimDate', ''),
                        'claim_source': item.get('claimant', ''),
                        'claim_reviewed': item.get('text', ''),
                        'publisher': {
                            'name': item.get('claimReview', [{}])[0].get('publisher', {}).get('name', ''),
                            'site': item.get('claimReview', [{}])[0].get('publisher', {}).get('site', '')
                        },
                        'rating': item.get('claimReview', [{}])[0].get('textualRating', ''),
                        'rating_explanation': item.get('claimReview', [{}])[0].get('title', ''),
                        'url': item.get('claimReview', [{}])[0].get('url', '')
                    }
                    fact_checks.append(fact_check)
            
            return fact_checks
        else:
            print(f"Error: Fact Check API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return []
    
    except Exception as e:
        print(f"Error checking facts: {e}")
        return []