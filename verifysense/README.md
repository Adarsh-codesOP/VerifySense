# VerifySense - AI-Powered Misinformation Detection Tool

VerifySense is an AI-powered tool that detects potential misinformation and educates users on identifying credible, trustworthy content. Built for the Google Gen AI Hackathon, it leverages Google's AI services to provide fact-checking, evidence gathering, and educational explanations.

## 🎯 Problem Statement

In today's digital landscape, misinformation spreads rapidly, making it challenging for users to distinguish between credible and false information. VerifySense addresses this problem by providing a user-friendly tool that:

- Detects potential misinformation with clear confidence ratings
- Provides evidence snippets from multiple sources with reliability indicators
- Explains the verification process in a human-readable format
- Teaches users fact-checking skills they can apply themselves

## 💡 Core Features

- **Claim Extraction**: Extract factual claims from text, URLs, or images
- **Fact Check Lookup**: Check claims against Google's Fact Check Tools API
- **Evidence Retrieval**: Gather supporting or contradicting evidence from reputable sources
- **Credibility Scoring**: Calculate confidence scores based on multiple factors
- **Educational Explanations**: Teach users how to verify information themselves
- **OCR Capability**: Extract text from images and screenshots for verification

## 🛠 Technology Stack

### Google Services
- Google Fact Check Tools API
- Gemini API (Google AI)
- Cloud Natural Language API
- Custom Search JSON API
- Cloud Vision API (for OCR)
- Google Cloud Platform (GCP)

### Frontend
- React
- Tailwind CSS
- Axios for API requests

### Backend
- Flask (Python)
- Google API clients

### Data Storage
- Firestore (for caching and user feedback)

## ⚙️ Architecture

1. **Input**: User submits text, URL, or image
2. **Claim Extraction**: Extract factual claims using Gemini and NL API
3. **Fact Check Lookup**: Check claims against Google Fact Check Tools API
4. **Evidence Retrieval**: Gather evidence using Custom Search JSON API
5. **Scoring**: Calculate credibility score based on multiple factors
6. **Explanation**: Generate human-readable explanation using Gemini
7. **Results Display**: Show verification results, evidence, and educational content

## 🚀 Getting Started

### Prerequisites

- Node.js and npm for the frontend
- Python 3.8+ for the backend
- Google Cloud Platform account with API keys

### Environment Variables

Create a `.env` file in the backend directory with the following variables:

```
GEMINI_API_KEY=your_gemini_api_key_here
FACT_CHECK_API_KEY=your_fact_check_api_key_here
CUSTOM_SEARCH_API_KEY=your_custom_search_api_key_here
SEARCH_ENGINE_ID=your_search_engine_id_here
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_service_account_key.json
```

### Installation

#### Backend Setup

```bash
# Navigate to the backend directory
cd verifysense/backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py
```

#### Frontend Setup

```bash
# Navigate to the frontend directory
cd verifysense/frontend

# Install dependencies
npm install

# Start the development server
npm start
```

## 📊 Scoring System

VerifySense uses a multi-factor scoring system to evaluate claims:

- **ClaimMatchScore**: How closely the claim matches known fact-checked claims
- **SourceReliabilityAvg**: Average reliability of evidence sources
- **CrossSourceConsistency**: Agreement across multiple reliable sources
- **Temporal Relevance**: Recency and timeliness of the information

The final output includes:
- A confidence label (Likely True, Mixed/Needs Verification, or Likely False)
- A numerical confidence score (0-100)
- Component scores for transparency

## 🛡️ Ethics & Limitations

- We never present absolute verdicts - all results include confidence levels
- When evidence is insufficient, we clearly state this and guide users
- We respect user privacy and do not store full claims without consent
- Our system has limitations and should be used as one tool among many

## 🙏 Acknowledgements

- Google Gen AI Hackathon for the opportunity
- Google's AI and search services that power this application