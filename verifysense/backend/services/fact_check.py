import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Fact Check Tools API endpoint
FACT_CHECK_API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

import os
import logging
import requests
import json
import hashlib
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize sentence transformer model for semantic similarity
try:
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    logger.info("NLP model loaded successfully for fact checking")
except Exception as e:
    logger.error(f"Error loading NLP model for fact checking: {str(e)}")
    model = None

# Trusted fact-checking sources
FACT_CHECK_SOURCES = [
    "factcheck.org",
    "politifact.com",
    "snopes.com",
    "apnews.com/hub/ap-fact-check",
    "reuters.com/fact-check",
    "bbc.com/news/reality_check",
    "fullfact.org"
]

# Cache for fact checks to avoid repeated API calls
fact_check_cache = {}

def check_facts(claim, request_id=None):
    """
    Check a claim against multiple fact-checking services and methods
    
    Args:
        claim (str): The claim to check
        request_id (str, optional): Unique identifier for the request for tracking
        
    Returns:
        dict: Fact check results with status information
    """
    # Generate request ID if not provided
    if not request_id:
        request_id = hashlib.md5(f"{claim}_{datetime.now().isoformat()}".encode()).hexdigest()
    
    # Check cache first (only if not in development mode)
    cache_key = hashlib.md5(claim.encode()).hexdigest()
    if not os.environ.get('DEVELOPMENT_MODE') and cache_key in fact_check_cache:
        logger.info(f"Using cached fact check for claim: '{claim[:50]}...'")
        cached_result = fact_check_cache[cache_key]
        cached_result['cached'] = True
        return cached_result
    
    logger.info(f"Checking facts for claim: '{claim[:50]}...' (Request ID: {request_id})")
    
    results = {
        'google_fact_check': [],
        'claim_buster': [],
        'custom_verification': {},
        'status': 'success',
        'request_id': request_id,
        'cached': False
    }
    
    # 1. Try Google Fact Check API
    try:
        google_results = check_google_fact_check_api(claim)
        if google_results:
            results['google_fact_check'] = google_results
            logger.info(f"Found {len(google_results)} Google Fact Check results")
    except Exception as e:
        logger.error(f"Error checking Google Fact Check API: {str(e)}")
        results['status'] = 'partial'
        results['errors'] = results.get('errors', []) + [{'service': 'google_fact_check', 'error': str(e)}]
    
    # 2. Try ClaimBuster API (if available)
    try:
        claimbuster_results = check_claimbuster_api(claim)
        if claimbuster_results:
            results['claim_buster'] = claimbuster_results
            logger.info(f"ClaimBuster score: {claimbuster_results.get('score', 'N/A')}")
    except Exception as e:
        logger.error(f"Error checking ClaimBuster API: {str(e)}")
        results['status'] = 'partial'
        results['errors'] = results.get('errors', []) + [{'service': 'claim_buster', 'error': str(e)}]
    
    # 3. Perform custom verification using NLP and web search
    try:
        custom_results = perform_custom_verification(claim)
        if custom_results:
            results['custom_verification'] = custom_results
            logger.info(f"Custom verification completed with confidence: {custom_results.get('confidence', 'N/A')}")
    except Exception as e:
        logger.error(f"Error performing custom verification: {str(e)}")
        results['status'] = 'partial'
        results['errors'] = results.get('errors', []) + [{'service': 'custom_verification', 'error': str(e)}]
    
    # If all methods failed, mark as error
    if not results['google_fact_check'] and not results['claim_buster'] and not results['custom_verification']:
        if results['status'] == 'partial':
            results['status'] = 'error'
            results['message'] = 'All fact-checking methods failed'
    
    # Cache the result (only if successful)
    if results['status'] == 'success' or results['status'] == 'partial':
        fact_check_cache[cache_key] = results
    
    return results

def check_google_fact_check_api(claim):
    """
    Check a claim using Google Fact Check API
    
    Args:
        claim (str): The claim to check
        
    Returns:
        list: List of fact check results from Google Fact Check API
    """
    # TODO: Implement actual Google Fact Check API integration
    # For now, return mock data with a note that this is a placeholder
    logger.info("Using mock Google Fact Check API data (API integration pending)")
    
    return [
        {
            "publisher": {
                "name": "Example Fact Checker",
                "site": "https://example.com"
            },
            "claim": claim,
            "rating": "Mostly True",
            "url": "https://example.com/fact-check/123",
            "date": "2023-01-15",
            "implementation_status": "in-progress"
        }
    ]

def check_claimbuster_api(claim):
    """
    Check a claim using ClaimBuster API
    
    Args:
        claim (str): The claim to check
        
    Returns:
        dict: ClaimBuster API results
    """
    # TODO: Implement actual ClaimBuster API integration
    # For now, return mock data with a note that this is a placeholder
    logger.info("Using mock ClaimBuster API data (API integration pending)")
    
    return {
        "score": 0.75,  # Score between 0-1, higher means more likely to be check-worthy
        "explanation": "This claim contains factual assertions that could be verified",
        "implementation_status": "in-progress"
    }

def perform_custom_verification(claim):
    """
    Perform custom verification using NLP and web search
    
    Args:
        claim (str): The claim to check
        
    Returns:
        dict: Custom verification results
    """
    # Implement a basic verification using cosine similarity with known facts
    # In a real implementation, this would search trusted sources and compare
    
    # For demonstration, we'll use a simple approach with pre-defined facts
    known_facts = [
        "The Earth orbits around the Sun",
        "Water boils at 100 degrees Celsius at sea level",
        "The human body has 206 bones",
        "Mount Everest is the tallest mountain on Earth",
        "The Great Wall of China is visible from space"
    ]
    
    if model:
        try:
            # Encode the claim
            claim_embedding = model.encode([claim])[0]
            
            # Encode known facts
            fact_embeddings = model.encode(known_facts)
            
            # Calculate similarities
            similarities = cosine_similarity([claim_embedding], fact_embeddings)[0]
            
            # Find the most similar fact and its similarity score
            max_similarity_idx = similarities.argmax()
            max_similarity = similarities[max_similarity_idx]
            most_similar_fact = known_facts[max_similarity_idx]
            
            # Determine confidence based on similarity
            if max_similarity > 0.8:
                confidence = "high"
            elif max_similarity > 0.5:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                "most_similar_fact": most_similar_fact,
                "similarity_score": float(max_similarity),
                "confidence": confidence,
                "method": "semantic_similarity"
            }
        except Exception as e:
            logger.error(f"Error in custom verification: {str(e)}")
            return {
                "error": str(e),
                "confidence": "low",
                "method": "semantic_similarity_failed"
            }
    
    # Fallback if model is not available
    return {
        "confidence": "low",
        "method": "rule_based",
        "explanation": "NLP model not available, using basic rule-based verification"
    }