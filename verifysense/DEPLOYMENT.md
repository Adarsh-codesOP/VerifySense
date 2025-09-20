# VerifySense Deployment Guide

This guide provides instructions for deploying the VerifySense application in both development and production environments.

## Prerequisites

- Google Cloud Platform account with the following APIs enabled:
  - Gemini API
  - Fact Check Tools API
  - Custom Search JSON API
  - Cloud Vision API (for OCR)
  - Firestore (optional, for caching and user feedback)
- Node.js 16+ and npm for the frontend
- Python 3.8+ for the backend
- Git

## Local Development Setup

### Backend Setup

1. Clone the repository and navigate to the backend directory:

```bash
git clone https://github.com/yourusername/verifysense.git
cd verifysense/backend
```

2. Create a virtual environment and activate it:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file based on the `.env.example` file:

```bash
cp .env.example .env
```

5. Edit the `.env` file with your API keys and configuration:

```
GEMINI_API_KEY=your_gemini_api_key_here
FACT_CHECK_API_KEY=your_fact_check_api_key_here
CUSTOM_SEARCH_API_KEY=your_custom_search_api_key_here
SEARCH_ENGINE_ID=your_search_engine_id_here
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_service_account_key.json

FLASK_APP=app.py
FLASK_ENV=development
PORT=5000
```

6. Run the Flask application:

```bash
python app.py
```

The backend should now be running at http://localhost:5000.

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd ../frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm start
```

The frontend should now be running at http://localhost:3000 and proxying API requests to the backend.

## Production Deployment

### Backend Deployment on Google Cloud Run

1. Create a Dockerfile in the backend directory:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
```

2. Build and push the Docker image to Google Container Registry:

```bash
# Set your Google Cloud project ID
export PROJECT_ID=your-project-id

# Build the image
gcloud builds submit --tag gcr.io/$PROJECT_ID/verifysense-backend
```

3. Deploy to Cloud Run:

```bash
gcloud run deploy verifysense-backend \
  --image gcr.io/$PROJECT_ID/verifysense-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GEMINI_API_KEY=your_gemini_api_key,FACT_CHECK_API_KEY=your_fact_check_api_key,CUSTOM_SEARCH_API_KEY=your_custom_search_api_key,SEARCH_ENGINE_ID=your_search_engine_id"
```

### Frontend Deployment on Firebase Hosting

1. Install Firebase CLI:

```bash
npm install -g firebase-tools
```

2. Login to Firebase:

```bash
firebase login
```

3. Initialize Firebase in the frontend directory:

```bash
cd ../frontend
firebase init hosting
```

4. Update the proxy in `package.json` to point to your Cloud Run backend URL:

```json
"proxy": "https://verifysense-backend-xxxxx-uc.a.run.app"
```

5. Build the production version of the frontend:

```bash
npm run build
```

6. Deploy to Firebase Hosting:

```bash
firebase deploy --only hosting
```

## Setting Up Firestore (Optional)

If you want to implement caching and user feedback storage:

1. Create a Firestore database in your Google Cloud Console

2. Update your backend `.env` file with the Firestore configuration:

```
FIRESTORE_PROJECT_ID=your-project-id
```

3. Implement the Firestore integration in the feedback endpoint in `app.py`

## Monitoring and Maintenance

- Set up Google Cloud Monitoring for your Cloud Run service
- Configure alerts for error rates and latency
- Regularly update dependencies to ensure security and performance
- Monitor API usage to stay within quota limits

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure all API keys are correctly set in the environment variables

2. **CORS Errors**: If you encounter CORS issues, check that the frontend proxy is correctly configured

3. **Deployment Failures**: Check Cloud Build and Cloud Run logs for detailed error messages

4. **Performance Issues**: Consider scaling up Cloud Run instances or optimizing API calls

### Getting Help

If you encounter issues not covered in this guide, please:

- Check the project documentation
- Open an issue on the GitHub repository
- Contact the project maintainers