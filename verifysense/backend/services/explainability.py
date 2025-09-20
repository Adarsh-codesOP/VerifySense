import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def generate_explanation(claim, fact_checks, evidence, score):
    """
    Generate a human-readable explanation of the verification process using Gemini API
    
    Args:
        claim (str): The claim being evaluated
        fact_checks (list): List of fact check results from Google Fact Check API
        evidence (list): List of evidence items from web search
        score (dict): Score information including numerical score and confidence label
        
    Returns:
        dict: Explanation with summary and steps
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prepare fact check information
        fact_check_info = ""
        if fact_checks:
            fact_check_info = "\nFact Check Results:\n"
            for i, check in enumerate(fact_checks, 1):
                fact_check_info += f"- {check.get('publisher', {}).get('name', 'Fact Checker')}: {check.get('rating', 'No rating')}\n"
        else:
            fact_check_info = "\nNo direct fact checks were found for this claim.\n"
        
        # Prepare evidence information
        evidence_info = ""
        if evidence:
            evidence_info = "\nEvidence Sources:\n"
            for i, item in enumerate(evidence, 1):
                evidence_info += f"- {item.get('title', 'Source')}: {item.get('snippet', 'No snippet')}\n"
        else:
            evidence_info = "\nNo supporting evidence was found for this claim.\n"
        
        # Prepare score information
        score_info = f"\nConfidence: {score.get('confidence_label', 'Unknown')} ({score.get('score', 0)}/100)\n"
        
        # Prompt for explanation generation
        prompt = f"""
        As a fact-checking educator, explain how the following claim was verified in a clear, educational way.
        
        Claim: "{claim}"
        {fact_check_info}
        {evidence_info}
        {score_info}
        
        Create a brief, educational explanation of how this claim was verified. Include:
        1. A short summary paragraph explaining the verification result
        2. 3-4 clear steps showing how someone could verify this information themselves
        3. A brief conclusion with advice on evaluating similar claims
        
        Format the response as JSON with two fields:
        - summary: A concise paragraph explaining the verification result
        - steps: An array of 3-4 verification steps anyone could follow
        """
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Process response to extract explanation
        explanation_text = response.text.strip()
        
        # Parse the response - handle both JSON and non-JSON responses
        try:
            import json
            explanation = json.loads(explanation_text)
        except:
            # If JSON parsing fails, create a structured response manually
            lines = explanation_text.split('\n')
            summary = lines[0] if lines else "We analyzed this claim using fact-checking services and evidence from reliable sources."
            
            # Extract steps - look for numbered lines
            steps = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() and line[1:3] in ['. ', ') ']):
                    step = line[3:] if line[1:3] == '. ' else line[2:]
                    steps.append(step)
            
            # If no steps were found, create generic ones
            if not steps:
                steps = [
                    "Check official fact-checking websites for this claim",
                    "Look for reporting from multiple reliable news sources",
                    "Verify the original context and source of the claim",
                    "Consider the evidence quality and consistency across sources"
                ]
            
            explanation = {
                'summary': summary,
                'steps': steps
            }
        
        return explanation
    
    except Exception as e:
        print(f"Error generating explanation: {e}")
        # Return a default explanation if generation fails
        return {
            'summary': f"We analyzed this claim and found it to be {score.get('confidence_label', 'uncertain')} based on available evidence.",
            'steps': [
                "Check official fact-checking websites for this claim",
                "Look for reporting from multiple reliable news sources",
                "Verify the original context and source of the claim",
                "Consider the evidence quality and consistency across sources"
            ]
        }