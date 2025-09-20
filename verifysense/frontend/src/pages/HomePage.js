import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MagnifyingGlassIcon, DocumentTextIcon, LinkIcon, PhotoIcon } from '@heroicons/react/24/outline';
import axios from 'axios';

function HomePage() {
  const [inputType, setInputType] = useState('text');
  const [content, setContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [imagePreview, setImagePreview] = useState('');
  
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!content) {
      setError('Please enter some content to verify');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      // For demo purposes, we'll simulate the API call
      // In a real implementation, this would call the backend API
      const response = await axios.post('/api/verify', {
        input_type: inputType,
        content: content
      });
      
      // Store results in localStorage for the results page
      localStorage.setItem('verifyResults', JSON.stringify(response.data));
      
      // Navigate to results page
      navigate('/results');
    } catch (err) {
      console.error('Error verifying content:', err);
      setError('An error occurred while verifying the content. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setContent(reader.result);
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Verify Information with AI
        </h1>
        <p className="text-xl text-gray-600">
          Detect potential misinformation and learn how to identify credible content
        </p>
      </div>

      <div className="card mb-8">
        <form onSubmit={handleSubmit}>
          <div className="mb-6">
            <div className="flex space-x-4 mb-4">
              <button
                type="button"
                className={`flex-1 py-3 rounded-md flex justify-center items-center space-x-2 ${inputType === 'text' ? 'bg-primary-100 text-primary-700 border border-primary-300' : 'bg-gray-100 text-gray-700 border border-gray-300'}`}
                onClick={() => setInputType('text')}
              >
                <DocumentTextIcon className="h-5 w-5" />
                <span>Text</span>
              </button>
              <button
                type="button"
                className={`flex-1 py-3 rounded-md flex justify-center items-center space-x-2 ${inputType === 'url' ? 'bg-primary-100 text-primary-700 border border-primary-300' : 'bg-gray-100 text-gray-700 border border-gray-300'}`}
                onClick={() => setInputType('url')}
              >
                <LinkIcon className="h-5 w-5" />
                <span>URL</span>
              </button>
              <button
                type="button"
                className={`flex-1 py-3 rounded-md flex justify-center items-center space-x-2 ${inputType === 'image' ? 'bg-primary-100 text-primary-700 border border-primary-300' : 'bg-gray-100 text-gray-700 border border-gray-300'}`}
                onClick={() => setInputType('image')}
              >
                <PhotoIcon className="h-5 w-5" />
                <span>Image</span>
              </button>
            </div>

            {inputType === 'text' && (
              <textarea
                className="input h-40"
                placeholder="Paste the text you want to verify..."
                value={content}
                onChange={(e) => setContent(e.target.value)}
              />
            )}

            {inputType === 'url' && (
              <input
                type="url"
                className="input"
                placeholder="Enter the URL of the article or social media post..."
                value={content}
                onChange={(e) => setContent(e.target.value)}
              />
            )}

            {inputType === 'image' && (
              <div className="space-y-4">
                <div className="flex justify-center items-center border-2 border-dashed border-gray-300 rounded-md p-6">
                  {imagePreview ? (
                    <div className="text-center">
                      <img
                        src={imagePreview}
                        alt="Preview"
                        className="max-h-40 mx-auto mb-4"
                      />
                      <button
                        type="button"
                        className="text-primary-600 hover:text-primary-800"
                        onClick={() => {
                          setImagePreview('');
                          setContent('');
                        }}
                      >
                        Remove image
                      </button>
                    </div>
                  ) : (
                    <div className="text-center">
                      <PhotoIcon className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                      <p className="text-gray-500 mb-2">Upload an image with text to verify</p>
                      <label className="btn btn-outline cursor-pointer">
                        Choose Image
                        <input
                          type="file"
                          accept="image/*"
                          className="hidden"
                          onChange={handleImageUpload}
                        />
                      </label>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {error && (
            <div className="mb-4 p-3 bg-danger-100 text-danger-700 rounded-md">
              {error}
            </div>
          )}

          <button
            type="submit"
            className="btn btn-primary w-full flex justify-center items-center space-x-2"
            disabled={isLoading}
          >
            {isLoading ? (
              <span>Verifying...</span>
            ) : (
              <>
                <MagnifyingGlassIcon className="h-5 w-5" />
                <span>Verify Now</span>
              </>
            )}
          </button>
        </form>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card text-center">
          <div className="text-primary-600 mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold mb-2">Fact Check</h3>
          <p className="text-gray-600 text-sm">Checks claims against Google's Fact Check API database</p>
        </div>
        <div className="card text-center">
          <div className="text-primary-600 mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold mb-2">Evidence Analysis</h3>
          <p className="text-gray-600 text-sm">Gathers and analyzes evidence from trusted sources</p>
        </div>
        <div className="card text-center">
          <div className="text-primary-600 mb-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold mb-2">Learn to Verify</h3>
          <p className="text-gray-600 text-sm">Teaches you how to verify information yourself</p>
        </div>
      </div>
    </div>
  );
}

export default HomePage;