import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

def extract_claims(content):
    """
    Extract claims from the provided content using Gemini API
    
    Args:
        content (str): The text content to extract claims from
        
    Returns:
        list: A list of extracted claims
    """
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Prompt for claim extraction
        prompt = f"""
        Extract the main factual claims from the following text. 
        A factual claim is a statement that can be verified as true or false.
        Focus only on objective, verifiable assertions, not opinions or subjective statements.
        Return only the claims as a numbered list, with each claim being concise and focused on a single fact.
        
        Text: {content}
        """
        
        # Generate response
        response = model.generate_content(prompt)
        
        # Process response to extract claims
        claims_text = response.text.strip()
        
        # Parse numbered list into separate claims
        claims = []
        for line in claims_text.split('\n'):
            # Remove numbering and whitespace
            if line.strip() and any(c.isdigit() for c in line[:2]):
                claim = line.split('.', 1)[-1].strip()
                if claim:
                    claims.append(claim)
        
        # If parsing failed, just return the whole text as one claim
        if not claims and claims_text:
            claims = [claims_text]
            
        return claims
    
    except Exception as e:
        print(f"Error extracting claims: {e}")
        # Return the original content as a single claim if extraction fails
        return [content] if content else []