import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

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

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'VerifySense API is running'})

@app.route('/api/verify', methods=['POST'])
def verify():
    data = request.json
    
    # Extract input type and content
    input_type = data.get('input_type', 'text')  # text, url, image
    content = data.get('content', '')
    
    # Process based on input type
    if input_type == 'image' and content:
        # Extract text from image using OCR
        content = extract_text_from_image(content)
    
    # Extract claims from content
    claims = extract_claims(content)
    
    if not claims:
        return jsonify({
            'status': 'error',
            'message': 'No claims could be extracted from the provided content'
        }), 400
    
    # For each claim, check facts and gather evidence
    results = []
    for claim in claims:
        # Check against Google Fact Check API
        fact_checks = check_facts(claim)
        
        # Retrieve evidence from web search
        evidence = get_evidence(claim)
        
        # Calculate credibility score
        score = calculate_score(claim, fact_checks, evidence)
        
        # Generate explanation for verification process
        explanation = generate_explanation(claim, fact_checks, evidence, score)
        
        results.append({
            'claim': claim,
            'fact_checks': fact_checks,
            'evidence': evidence,
            'score': score,
            'explanation': explanation
        })
    
    return jsonify({
        'status': 'success',
        'results': results
    })

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    # TODO: Store feedback in Firestore
    return jsonify({'status': 'success', 'message': 'Feedback received'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=True)
