import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.claim_extraction import extract_claims
from services.fact_check import check_facts
from services.evidence_retrieval import retrieve_evidence, get_evidence
from services.scoring import calculate_score
from services.explainability import generate_explanation

class TestClaimExtraction(unittest.TestCase):
    @patch('services.claim_extraction.genai')
    def test_extract_claims(self, mock_genai):
        # Mock the Gemini API response
        mock_response = MagicMock()
        mock_response.text = json.dumps(["Claim 1", "Claim 2"])
        mock_genai.GenerativeModel().generate_content().text = mock_response.text
        
        # Test the function
        result = extract_claims("Test content with claims")
        
        # Assertions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Claim 1")
        self.assertEqual(result[1], "Claim 2")

class TestFactCheck(unittest.TestCase):
    @patch('services.fact_check.requests.get')
    def test_check_facts(self, mock_get):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "claims": [
                {
                    "text": "Test claim",
                    "claimant": "Test source",
                    "claimDate": "2023-01-01",
                    "claimReview": [
                        {
                            "publisher": {"name": "Test publisher"},
                            "textualRating": "False",
                            "reviewDate": "2023-01-02",
                            "url": "https://example.com/review"
                        }
                    ]
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Test the function
        result = check_facts("Test claim")
        
        # Assertions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["claim"], "Test claim")
        self.assertEqual(result[0]["source"], "Test source")
        self.assertEqual(result[0]["rating"], "False")

class TestEvidenceRetrieval(unittest.TestCase):
    @patch('services.evidence_retrieval.build')
    def test_retrieve_evidence(self, mock_build):
        # Mock the API response
        mock_cse = MagicMock()
        mock_list = MagicMock()
        mock_execute = MagicMock()
        
        mock_execute.return_value = {
            "items": [
                {
                    "title": "Test title",
                    "link": "https://example.com",
                    "snippet": "Test snippet",
                    "displayLink": "example.com"
                }
            ]
        }
        
        mock_list.execute = mock_execute
        mock_cse.list.return_value = mock_list
        mock_build.return_value = mock_cse
        
        # Test the function
        result = retrieve_evidence("Test claim")
        
        # Assertions
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Test title")
        self.assertEqual(result[0]["url"], "https://example.com")
        self.assertEqual(result[0]["snippet"], "Test snippet")
        self.assertEqual(result[0]["domain"], "example.com")

class TestScoring(unittest.TestCase):
    def test_calculate_score(self):
        # Test data
        claim = "Test claim"
        fact_checks = [
            {
                "claim": "Test claim",
                "rating": "False",
                "source": "Test source",
                "url": "https://example.com"
            }
        ]
        evidence = [
            {
                "title": "Test title",
                "url": "https://example.com",
                "snippet": "Test snippet contradicts the claim",
                "domain": "example.com",
                "reliability": 0.8
            }
        ]
        
        # Test the function
        result = calculate_score(claim, fact_checks, evidence)
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertIn("confidence_score", result)
        self.assertIn("confidence_label", result)
        self.assertIn("component_scores", result)

class TestExplainability(unittest.TestCase):
    @patch('services.explainability.genai')
    def test_generate_explanation(self, mock_genai):
        # Mock the Gemini API response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "summary": "Test explanation summary",
            "details": "Test explanation details",
            "educational_points": ["Point 1", "Point 2"]
        })
        mock_genai.GenerativeModel().generate_content().text = mock_response.text
        
        # Test data
        claim = "Test claim"
        fact_checks = [{"claim": "Test claim", "rating": "False"}]
        evidence = [{"title": "Test title", "snippet": "Test snippet"}]
        score = {"confidence_score": 30, "confidence_label": "Likely False"}
        
        # Test the function
        result = generate_explanation(claim, fact_checks, evidence, score)
        
        # Assertions
        self.assertIsInstance(result, dict)
        self.assertEqual(result["summary"], "Test explanation summary")
        self.assertEqual(result["details"], "Test explanation details")
        self.assertEqual(result["educational_points"], ["Point 1", "Point 2"])

if __name__ == "__main__":
    unittest.main()