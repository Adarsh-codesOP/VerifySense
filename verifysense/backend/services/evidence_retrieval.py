import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Custom Search API endpoint
SEARCH_API_URL = "https://www.googleapis.com/customsearch/v1"

# List of trusted news sources for better evidence quality
TRUSTED_DOMAINS = [
    'reuters.com',
    'apnews.com',
    'bbc.com',
    'bbc.co.uk',
    'npr.org',
    'nytimes.com',
    'washingtonpost.com',
    'wsj.com',
    'economist.com',
    'theguardian.com',
    'cnn.com',
    'nbcnews.com',
    'cbsnews.com',
    'abcnews.go.com',
    'politifact.com',
    'factcheck.org',
    'snopes.com',
    'usatoday.com',
    'time.com',
    'theatlantic.com'
]

def get_evidence(claim, max_results=5):
    """
    Retrieve evidence for a claim using Google's Custom Search JSON API
    
    Args:
        claim (str): The claim to search for evidence
        max_results (int): Maximum number of results to return
        
    Returns:
        list: A list of evidence items
    """
    try:
        # Get API key and search engine ID from environment variables
        api_key = os.getenv('CUSTOM_SEARCH_API_KEY')
        search_engine_id = os.getenv('SEARCH_ENGINE_ID')
        
        if not api_key or not search_engine_id:
            print("Warning: CUSTOM_SEARCH_API_KEY or SEARCH_ENGINE_ID not found in environment variables")
            return []
        
        # Prepare parameters for the API request
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': claim,
            'num': max_results
        }
        
        # Make the API request
        response = requests.get(SEARCH_API_URL, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant information from the response
            evidence = []
            if 'items' in data:
                for item in data['items']:
                    # Determine source reliability based on domain
                    domain = extract_domain(item.get('link', ''))
                    reliability = 'high' if domain in TRUSTED_DOMAINS else 'medium'
                    
                    evidence_item = {
                        'title': item.get('title', ''),
                        'snippet': item.get('snippet', ''),
                        'link': item.get('link', ''),
                        'source': domain,
                        'reliability': reliability,
                        'date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', '')
                    }
                    evidence.append(evidence_item)
            
            return evidence
        else:
            print(f"Error: Custom Search API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return []
    
    except Exception as e:
        print(f"Error retrieving evidence: {e}")
        return []

def extract_domain(url):
    """
    Extract domain name from a URL
    
    Args:
        url (str): The URL to extract domain from
        
    Returns:
        str: The extracted domain name
    """
    try:
        # Remove protocol (http://, https://) and get the domain
        domain = url.split('//')[1].split('/')[0]
        
        # Remove 'www.' if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        return domain
    except:
        return url