import logging
import os
import json
from datetime import datetime

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Create formatters and handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# File handler for all logs
file_handler = logging.FileHandler(os.path.join(log_dir, f'verifysense_{datetime.now().strftime("%Y%m%d")}.log'))
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Create loggers
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# Main application logger
app_logger = get_logger('verifysense')

# Specialized loggers
extraction_logger = get_logger('verifysense.extraction')
claim_logger = get_logger('verifysense.claim')
factcheck_logger = get_logger('verifysense.factcheck')
scoring_logger = get_logger('verifysense.scoring')

def log_request(request_id, url=None, content=None, content_type=None):
    """Log incoming verification request"""
    app_logger.info(f"New verification request: {request_id}")
    if url:
        app_logger.info(f"URL: {url}")
    if content_type:
        app_logger.info(f"Content type: {content_type}")
    
def log_extraction(request_id, extracted_content, source=None):
    """Log content extraction results"""
    extraction_logger.info(f"Content extraction for request {request_id}")
    if source:
        extraction_logger.info(f"Source: {source}")
    extraction_logger.info(f"Extracted content length: {len(extracted_content) if extracted_content else 0}")
    extraction_logger.debug(f"Extracted content: {extracted_content[:500]}...")

def log_claims(request_id, claims):
    """Log detected claims"""
    claim_logger.info(f"Claim detection for request {request_id}")
    claim_logger.info(f"Number of claims detected: {len(claims)}")
    for i, claim in enumerate(claims):
        claim_logger.info(f"Claim {i+1}: {claim}")

def log_factcheck(request_id, claim, verification_results):
    """Log fact-checking results"""
    factcheck_logger.info(f"Fact-checking for request {request_id}, claim: {claim[:100]}...")
    factcheck_logger.info(f"Verification sources used: {len(verification_results['sources']) if 'sources' in verification_results else 0}")
    factcheck_logger.debug(f"Verification results: {json.dumps(verification_results, indent=2)}")

def log_scoring(request_id, claim, score_data):
    """Log scoring results"""
    scoring_logger.info(f"Scoring for request {request_id}, claim: {claim[:100]}...")
    scoring_logger.info(f"Overall score: {score_data['score']}, Confidence: {score_data['confidence_label']}")
    if 'components' in score_data:
        for component, value in score_data['components'].items():
            scoring_logger.info(f"Component score - {component}: {value}")

def log_error(request_id, error_message, error_type=None, stack_trace=None):
    """Log errors"""
    app_logger.error(f"Error in request {request_id}: {error_message}")
    if error_type:
        app_logger.error(f"Error type: {error_type}")
    if stack_trace:
        app_logger.error(f"Stack trace: {stack_trace}")