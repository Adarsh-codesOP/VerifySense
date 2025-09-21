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
    """
    # Generate request ID if not provided
    if not request_id:
        request_id = hashlib.md5(f"{claim}_{datetime.now().isoformat()}".encode()).hexdigest()
    
    logger.info(f"Calculating score for claim: '{claim[:50]}...' (Request ID: {request_id})")
    
    # Initialize score components
    scores = {
        'claim_match_score': 50,
        'source_reliability_score': 50,
        'semantic_similarity_score': 50,
        'sentiment_consistency_score': 50,
        'cross_source_consistency_score': 50,
        'temporal_relevance_score': 50,
        'fact_check_score': 50
    }

    # 1. Fact Check Score
    if fact_checks:
        fact_check_scores = []
        for check in fact_checks:
            if isinstance(check, dict):
                rating = check.get('rating', '').lower()
                publisher = check.get('publisher', {}).get('name', '')
                url = check.get('url', '')
            else:
                # If string, assume it's a URL or unknown rating
                rating = ''
                publisher = ''
                url = str(check)
            
            logger.info(f"Fact check from {publisher or url}: Rating '{rating}'")

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
                score = 50  # Neutral

            # Publisher reliability
            publisher_domain = extract_domain(url)
            publisher_reliability = SOURCE_RELIABILITY.get(publisher_domain, 0.7)
            weighted_score = score * publisher_reliability
            fact_check_scores.append(weighted_score)

        if fact_check_scores:
            scores['fact_check_score'] = sum(fact_check_scores) / len(fact_check_scores)

    # 2. Source Reliability Score
    if evidence:
        reliability_scores = []
        for item in evidence:
            if isinstance(item, dict):
                source_url = item.get('url', '')
            else:
                source_url = str(item)
            
            domain = extract_domain(source_url)
            reliability = SOURCE_RELIABILITY.get(domain, 0.6)
            reliability_scores.append(reliability * 100)
            logger.info(f"Source reliability for {domain}: {reliability * 100}")
        
        if reliability_scores:
            scores['source_reliability_score'] = sum(reliability_scores) / len(reliability_scores)

    # 3. Semantic Similarity Score
    if model and evidence:
        try:
            claim_embedding = model.encode([claim])[0]
            similarity_scores = []
            for item in evidence:
                content = ''
                if isinstance(item, dict):
                    content = item.get('content', '')
                else:
                    content = str(item)
                if content:
                    content_sample = content[:1000]
                    content_embedding = model.encode([content_sample])[0]
                    similarity = cosine_similarity([claim_embedding], [content_embedding])[0][0]
                    similarity_scores.append((similarity + 1) * 50)
                    logger.info(f"Semantic similarity score: {(similarity + 1) * 50:.2f}")
            if similarity_scores:
                scores['semantic_similarity_score'] = sum(similarity_scores) / len(similarity_scores)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")

    # 4. Sentiment Consistency Score
    if evidence:
        claim_sentiment = TextBlob(claim).sentiment.polarity
        sentiment_scores = []
        for item in evidence:
            content = ''
            if isinstance(item, dict):
                content = item.get('content', '')
            else:
                content = str(item)
            if content:
                evidence_sentiment = TextBlob(content[:1000]).sentiment.polarity
                consistency_score = 100 - abs(claim_sentiment - evidence_sentiment) * 50
                sentiment_scores.append(consistency_score)
                logger.info(f"Sentiment consistency score: {consistency_score:.2f}")
        if sentiment_scores:
            scores['sentiment_consistency_score'] = sum(sentiment_scores) / len(sentiment_scores)

    # 5. Cross-Source Consistency
    if evidence:
        high_reliability_sources = sum(
            1 for item in evidence
            if SOURCE_RELIABILITY.get(
                extract_domain(item.get('url', '') if isinstance(item, dict) else str(item)), 0
            ) > 0.8
        )
        if high_reliability_sources >= 3:
            scores['cross_source_consistency_score'] = 90
        elif high_reliability_sources >= 2:
            scores['cross_source_consistency_score'] = 75
        elif high_reliability_sources >= 1:
            scores['cross_source_consistency_score'] = 60
        else:
            scores['cross_source_consistency_score'] = 40
        logger.info(f"Cross-source consistency score: {scores['cross_source_consistency_score']} (from {high_reliability_sources} high-reliability sources)")

    # 6. Temporal Relevance
    scores['temporal_relevance_score'] = 70

    # Weighted Final Score
    weights = {
        'fact_check_score': 0.25,
        'source_reliability_score': 0.2,
        'semantic_similarity_score': 0.2,
        'sentiment_consistency_score': 0.1,
        'cross_source_consistency_score': 0.15,
        'temporal_relevance_score': 0.1
    }
    # Match weights correctly with score keys
    weighted_scores = [scores[key] * weights[key] for key in weights]
    final_score = round(sum(weighted_scores))

    # Confidence Label
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
    