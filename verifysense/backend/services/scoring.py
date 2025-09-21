import logging
import numpy as np
from datetime import datetime
import re
import os
import json
import hashlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize sentence transformer model for semantic similarity
try:
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    logger.info("NLP model loaded successfully")
except Exception as e:
    logger.error(f"Error loading NLP model: {str(e)}")
    model = None

# Source reliability database - can be expanded or moved to a separate file
SOURCE_RELIABILITY = {
    'bbc.com': 0.9,
    'nytimes.com': 0.85,
    'reuters.com': 0.9,
    'apnews.com': 0.9,
    'theguardian.com': 0.85,
    'npr.org': 0.85,
    'washingtonpost.com': 0.85,
    'economist.com': 0.85,
    'nature.com': 0.95,
    'science.org': 0.95,
    'who.int': 0.95,
    'cdc.gov': 0.95,
    'nih.gov': 0.95,
    'wikipedia.org': 0.75,
    'factcheck.org': 0.85,
    'politifact.com': 0.8,
    'snopes.com': 0.8,
}

def calculate_score(claim, fact_checks, evidence, request_id=None):
    """
    Calculate a credibility score for a claim based on fact checks and evidence
    using NLP models and multiple verification signals
    
    Args:
        claim (str): The claim being evaluated
        fact_checks (list): List of fact check results from Google Fact Check API
        evidence (list): List of evidence items from web search
        request_id (str, optional): Unique identifier for the request for tracking
        
    Returns:
        dict: Score information including numerical score, confidence label, and component scores
    """
    # Generate request ID if not provided
    if not request_id:
        request_id = hashlib.md5(f"{claim}_{datetime.now().isoformat()}".encode()).hexdigest()
    
    logger.info(f"Calculating score for claim: '{claim[:50]}...' (Request ID: {request_id})")
    
    # Initialize score components with default values
    scores = {
        'claim_match_score': 50,
        'source_reliability_score': 50,
        'semantic_similarity_score': 50,
        'sentiment_consistency_score': 50,
        'cross_source_consistency_score': 50,
        'temporal_relevance_score': 50,
        'fact_check_score': 50
    }
    
    # 1. Calculate Fact Check Score based on external fact-checking services
    if fact_checks:
        fact_check_scores = []
        for check in fact_checks:
            rating = check.get('rating', '').lower()
            publisher = check.get('publisher', {}).get('name', '')
            
            # Log the fact check information
            logger.info(f"Fact check from {publisher}: Rating '{rating}'")
            
            # Convert rating to numerical score
            if 'false' in rating or 'pants on fire' in rating:
                score = 20
            elif 'mostly false' in rating or 'misleading' in rating:
                score = 30
            elif 'half true' in rating or 'mixture' in rating or 'mixed' in rating:
                score = 50
            elif 'mostly true' in rating:
                score = 70
            elif 'true' in rating:
                score = 90
            else:
                score = 50  # Neutral for unknown ratings
            
            # Adjust score based on publisher reliability
            publisher_domain = extract_domain(check.get('url', ''))
            publisher_reliability = SOURCE_RELIABILITY.get(publisher_domain, 0.7)
            
            # Weight the score by publisher reliability
            weighted_score = score * publisher_reliability
            fact_check_scores.append(weighted_score)
        
        if fact_check_scores:
            scores['fact_check_score'] = sum(fact_check_scores) / len(fact_check_scores)
    
    # 2. Calculate Source Reliability Score
    if evidence:
        reliability_scores = []
        for item in evidence:
            source_url = item.get('url', '')
            domain = extract_domain(source_url)
            
            # Get reliability score from our database or default to medium
            reliability = SOURCE_RELIABILITY.get(domain, 0.6)
            reliability_score = reliability * 100
            
            reliability_scores.append(reliability_score)
            logger.info(f"Source reliability for {domain}: {reliability_score}")
        
        if reliability_scores:
            scores['source_reliability_score'] = sum(reliability_scores) / len(reliability_scores)
    
    # 3. Calculate Semantic Similarity Score using NLP
    if model and evidence:
        try:
            # Encode claim
            claim_embedding = model.encode([claim])[0]
            
            similarity_scores = []
            for item in evidence:
                content = item.get('content', '')
                if content:
                    # Take a sample of the content if it's too long
                    content_sample = content[:1000]
                    content_embedding = model.encode([content_sample])[0]
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity([claim_embedding], [content_embedding])[0][0]
                    
                    # Convert to score (0-100)
                    similarity_score = (similarity + 1) * 50  # Convert from [-1,1] to [0,100]
                    similarity_scores.append(similarity_score)
                    
                    logger.info(f"Semantic similarity score: {similarity_score:.2f}")
            
            if similarity_scores:
                scores['semantic_similarity_score'] = sum(similarity_scores) / len(similarity_scores)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
    
    # 4. Calculate Sentiment Consistency Score
    if evidence:
        claim_sentiment = TextBlob(claim).sentiment.polarity
        
        sentiment_diffs = []
        for item in evidence:
            content = item.get('content', '')
            if content:
                # Calculate sentiment of evidence
                evidence_sentiment = TextBlob(content[:1000]).sentiment.polarity
                
                # Calculate absolute difference in sentiment
                sentiment_diff = abs(claim_sentiment - evidence_sentiment)
                
                # Convert to consistency score (0-100)
                # Lower difference means higher consistency
                consistency_score = 100 - (sentiment_diff * 50)
                sentiment_diffs.append(consistency_score)
                
                logger.info(f"Sentiment consistency score: {consistency_score:.2f}")
        
        if sentiment_diffs:
            scores['sentiment_consistency_score'] = sum(sentiment_diffs) / len(sentiment_diffs)
    
    # 5. Calculate Cross-Source Consistency
    if evidence:
        # Count high reliability sources
        high_reliability_sources = sum(1 for item in evidence if 
                                      SOURCE_RELIABILITY.get(extract_domain(item.get('url', '')), 0) > 0.8)
        
        # Calculate score based on number of high reliability sources
        if high_reliability_sources >= 3:
            scores['cross_source_consistency_score'] = 90
        elif high_reliability_sources >= 2:
            scores['cross_source_consistency_score'] = 75
        elif high_reliability_sources >= 1:
            scores['cross_source_consistency_score'] = 60
        else:
            scores['cross_source_consistency_score'] = 40
        
        logger.info(f"Cross-source consistency score: {scores['cross_source_consistency_score']} (from {high_reliability_sources} high-reliability sources)")
    
    # 6. Calculate Temporal Relevance
    # For now, we'll use a default value, but this could be improved with publication dates
    scores['temporal_relevance_score'] = 70
    
    # Calculate final score as weighted average of components
    weights = {
        'fact_check_score': 0.25,
        'source_reliability_score': 0.2,
        'semantic_similarity_score': 0.2,
        'sentiment_consistency_score': 0.1,
        'cross_source_consistency_score': 0.15,
        'temporal_relevance_score': 0.1
    }
    
    # Calculate weighted score
    weighted_scores = [scores[key] * weights[key.replace('_score', '')] for key in scores]
    final_score = sum(weighted_scores)
    
    # Round to nearest integer
    final_score = round(final_score)
    
    # Determine confidence label based on score
    if final_score >= 75:
        confidence_label = 'Likely True'
    elif final_score >= 60:
        confidence_label = 'Somewhat True'
    elif final_score <= 30:
        confidence_label = 'Likely False'
    elif final_score <= 45:
        confidence_label = 'Somewhat False'
    else:
        confidence_label = 'Mixed / Needs Verification'
    
    # Log the final score and components
    logger.info(f"Final score: {final_score} ({confidence_label})")
    logger.info(f"Score components: {json.dumps(scores)}")
    
    return {
        'score': final_score,
        'confidence_label': confidence_label,
        'components': scores,
        'request_id': request_id
    }

def extract_domain(url):
    """Extract domain from URL"""
    if not url:
        return ""
    
    # Remove protocol and www
    domain = re.sub(r'^https?://(www\.)?', '', url.lower())
    
    # Get domain part (before path)
    domain = domain.split('/')[0]
    
    # Extract base domain (e.g., example.com from sub.example.com)
    parts = domain.split('.')
    if len(parts) > 2:
        # Check for country code TLDs like .co.uk
        if parts[-2] in ['co', 'com', 'org', 'net', 'edu', 'gov', 'mil'] and len(parts[-1]) == 2:
            return '.'.join(parts[-3:])
        else:
            return '.'.join(parts[-2:])
    
    return domain
    