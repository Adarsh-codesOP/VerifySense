def calculate_score(claim, fact_checks, evidence):
    """
    Calculate a credibility score for a claim based on fact checks and evidence
    
    Args:
        claim (str): The claim being evaluated
        fact_checks (list): List of fact check results from Google Fact Check API
        evidence (list): List of evidence items from web search
        
    Returns:
        dict: Score information including numerical score and confidence label
    """
    # Initialize score components
    claim_match_score = 0
    source_reliability_avg = 0
    cross_source_consistency = 0
    temporal_relevance = 0
    
    # Calculate ClaimMatchScore - based on exact/near matches in Fact Check API
    if fact_checks:
        # If we have fact checks, they're direct matches, so high score
        claim_match_score = 80
        
        # Adjust based on ratings if available
        for check in fact_checks:
            rating = check.get('rating', '').lower()
            
            # Adjust score based on fact check rating
            if 'false' in rating or 'pants on fire' in rating:
                claim_match_score = 20  # Likely false
            elif 'mostly false' in rating or 'misleading' in rating:
                claim_match_score = 30
            elif 'half true' in rating or 'mixture' in rating or 'mixed' in rating:
                claim_match_score = 50
            elif 'mostly true' in rating:
                claim_match_score = 70
            elif 'true' in rating:
                claim_match_score = 90
    else:
        # No fact checks found, neutral score
        claim_match_score = 50
    
    # Calculate SourceReliabilityAvg - reliability heuristic of sources
    if evidence:
        reliability_sum = 0
        for item in evidence:
            if item.get('reliability') == 'high':
                reliability_sum += 80
            elif item.get('reliability') == 'medium':
                reliability_sum += 50
            else:
                reliability_sum += 30
        
        source_reliability_avg = reliability_sum / len(evidence)
    else:
        # No evidence, neutral score
        source_reliability_avg = 50
    
    # Calculate CrossSourceConsistency - # of independent reputable sources confirming
    high_reliability_count = sum(1 for item in evidence if item.get('reliability') == 'high')
    if high_reliability_count >= 3:
        cross_source_consistency = 80
    elif high_reliability_count >= 2:
        cross_source_consistency = 70
    elif high_reliability_count >= 1:
        cross_source_consistency = 60
    else:
        cross_source_consistency = 50
    
    # Calculate Temporal Relevance - recent claims get lower certainty until verified
    # For simplicity, we'll use a default value for now
    temporal_relevance = 70
    
    # Calculate final score as weighted average of components
    weights = {
        'claim_match': 0.4,
        'source_reliability': 0.3,
        'cross_source': 0.2,
        'temporal': 0.1
    }
    
    final_score = (
        claim_match_score * weights['claim_match'] +
        source_reliability_avg * weights['source_reliability'] +
        cross_source_consistency * weights['cross_source'] +
        temporal_relevance * weights['temporal']
    )
    
    # Round to nearest integer
    final_score = round(final_score)
    
    # Determine confidence label based on score
    if final_score >= 70:
        confidence_label = 'Likely True'
    elif final_score <= 40:
        confidence_label = 'Likely False'
    else:
        confidence_label = 'Mixed / Needs Verification'
    
    return {
        'score': final_score,
        'confidence_label': confidence_label,
        'components': {
            'claim_match_score': claim_match_score,
            'source_reliability_avg': source_reliability_avg,
            'cross_source_consistency': cross_source_consistency,
            'temporal_relevance': temporal_relevance
        }
    }