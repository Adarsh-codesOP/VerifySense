import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from utils.logger import log_request, log_extraction, log_claims, log_factcheck, log_scoring, log_error

# Import services
from services.claim_extraction import extract_claims
from services.fact_check import check_facts
from services.evidence_retrieval import get_evidence
from services.scoring import calculate_score
from services.explainability import generate_explanation
from services.ocr import extract_text_from_image

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'VerifySense API is running'})

@app.route('/api/verify', methods=['POST'])
def verify():
    data = request.json
    
    # Generate a unique request ID for tracking
    request_id = data.get('request_id') or f"req_{int(datetime.now().timestamp())}"
    
    # Extract input type and content
    input_type = data.get('input_type', 'text')  # text, url, image
    content = data.get('content', '')
    
    # Log the incoming request
    log_request(request_id, data.get('url', ''), content, input_type)
    
    # Feature availability status
    feature_status = {
        'image_analysis': {'status': 'in-progress', 'available_by': '2023-12-31'},
        'video_analysis': {'status': 'planned', 'available_by': '2024-03-31'},
        'social_media_verification': {'status': 'in-progress', 'available_by': '2023-11-30'},
        'google_fact_check_api': {'status': 'in-progress', 'available_by': '2023-10-31'},
        'claimbuster_api': {'status': 'in-progress', 'available_by': '2023-10-31'}
    }
    
    # Process based on input type
    if input_type == 'image' and content:
        # Extract text from image using OCR
        try:
            content = extract_text_from_image(content)
            log_extraction(request_id, content, "image OCR")
        except Exception as e:
            log_error(request_id, f"OCR error: {str(e)}", "OCRException", None)
            return jsonify({
                'status': 'error',
                'message': 'Image processing failed',
                'feature_status': feature_status['image_analysis'],
                'request_id': request_id
            }), 400
    elif input_type == 'video':
        # Video analysis not yet implemented
        log_error(request_id, "Video analysis not yet implemented")
        return jsonify({
            'status': 'not_implemented',
            'message': 'Video analysis is not yet available',
            'feature_status': feature_status['video_analysis'],
            'request_id': request_id
        }), 501
    
    # Extract claims from content
    try:
        claims = extract_claims(content)
        log_claims(request_id, claims)
        app.logger.info(f"Extracted {len(claims)} claims from content")
    except Exception as e:
        app.logger.error(f"Claim extraction error: {str(e)}")
        log_error(request_id, f"Claim extraction error: {str(e)}", type(e).__name__, None)
        claims = []
    
    if not claims:
        log_error(request_id, "No claims could be extracted from the provided content")
        return jsonify({
            'status': 'error',
            'message': 'No claims could be extracted from the provided content',
            'request_id': request_id
        }), 400
    
    # For each claim, check facts and gather evidence
    results = []
    for claim in claims:
        app.logger.info(f"Processing claim: {claim[:50]}...")
        
        # Check against fact-checking services
        fact_checks = check_facts(claim, request_id)
        log_factcheck(request_id, claim, fact_checks)
        
        # Retrieve evidence from web search
        evidence = get_evidence(claim)
        
        # Calculate credibility score
        score = calculate_score(claim, fact_checks, evidence, request_id)
        log_scoring(request_id, claim, score)
        
        # Generate explanation for verification process
        explanation = generate_explanation(claim, fact_checks, evidence, score)
        
        # Check for features that are not fully implemented
        unavailable_features = []
        if fact_checks.get('google_fact_check') and any(check.get('implementation_status') == 'in-progress' for check in fact_checks.get('google_fact_check', [])):
            unavailable_features.append('google_fact_check_api')
        
        if fact_checks.get('claim_buster') and fact_checks.get('claim_buster', {}).get('implementation_status') == 'in-progress':
            unavailable_features.append('claimbuster_api')
        
        # Add feature status information for unavailable features
        feature_notifications = {feature: feature_status[feature] for feature in unavailable_features if feature in feature_status}
        
        results.append({
            'claim': claim,
            'fact_checks': fact_checks,
            'evidence': evidence,
            'score': score,
            'explanation': explanation,
            'feature_notifications': feature_notifications,
            'request_id': request_id
        })
    
    return jsonify({
        'status': 'success',
        'results': results,
        'request_id': request_id
    })

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    # TODO: Store feedback in Firestore
    return jsonify({'status': 'success', 'message': 'Feedback received'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
