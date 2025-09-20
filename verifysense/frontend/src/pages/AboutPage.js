import React from 'react';
import { ShieldCheckIcon, MagnifyingGlassIcon, DocumentTextIcon, AcademicCapIcon } from '@heroicons/react/24/outline';

function AboutPage() {
  return (
    <div className="max-w-3xl mx-auto">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          About VerifySense
        </h1>
        <p className="text-xl text-gray-600">
          An AI-powered tool to detect misinformation and educate users
        </p>
      </div>

      <div className="card mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Our Mission</h2>
        <p className="text-gray-700 mb-4">
          VerifySense is designed to help users identify potential misinformation and develop critical thinking skills for evaluating online content. In today's digital landscape, misinformation spreads rapidly, making it essential to have tools that can quickly verify claims and provide educational context.
        </p>
        <p className="text-gray-700">
          Our goal is not just to tell you what's true or false, but to show you how we reached that conclusion and teach you the skills to evaluate information yourself.
        </p>
      </div>

      <div className="card mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">How It Works</h2>
        
        <div className="space-y-6">
          <div className="flex">
            <div className="flex-shrink-0 mr-4">
              <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary-100 text-primary-600">
                <DocumentTextIcon className="h-6 w-6" />
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-1">Claim Extraction</h3>
              <p className="text-gray-700">
                We use Google's Gemini AI and Natural Language API to identify and extract factual claims from the content you provide, whether it's text, a URL, or an image.
              </p>
            </div>
          </div>
          
          <div className="flex">
            <div className="flex-shrink-0 mr-4">
              <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary-100 text-primary-600">
                <MagnifyingGlassIcon className="h-6 w-6" />
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-1">Fact Checking & Evidence Gathering</h3>
              <p className="text-gray-700">
                We check claims against Google's Fact Check Tools API and gather evidence from reputable sources using Google's Custom Search API to provide a comprehensive analysis.
              </p>
            </div>
          </div>
          
          <div className="flex">
            <div className="flex-shrink-0 mr-4">
              <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary-100 text-primary-600">
                <ShieldCheckIcon className="h-6 w-6" />
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-1">Credibility Scoring</h3>
              <p className="text-gray-700">
                Our algorithm evaluates multiple factors including fact check matches, source reliability, cross-source consistency, and temporal relevance to determine a confidence score.
              </p>
            </div>
          </div>
          
          <div className="flex">
            <div className="flex-shrink-0 mr-4">
              <div className="flex items-center justify-center w-10 h-10 rounded-full bg-primary-100 text-primary-600">
                <AcademicCapIcon className="h-6 w-6" />
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-1">Educational Explanation</h3>
              <p className="text-gray-700">
                We provide a step-by-step explanation of how we verified the claim, teaching you the skills to evaluate similar information in the future.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="card mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Technology</h2>
        <p className="text-gray-700 mb-4">
          VerifySense is built using a combination of Google's AI and search technologies:
        </p>
        
        <ul className="list-disc pl-5 space-y-2 text-gray-700">
          <li><span className="font-medium">Google Fact Check Tools API</span> - For accessing a database of fact-checked claims</li>
          <li><span className="font-medium">Gemini API</span> - For claim extraction, summarization, and generating educational explanations</li>
          <li><span className="font-medium">Cloud Natural Language API</span> - For entity recognition and claim segmentation</li>
          <li><span className="font-medium">Custom Search JSON API</span> - For retrieving evidence from reputable sources</li>
          <li><span className="font-medium">Cloud Vision API</span> - For extracting text from images and screenshots</li>
          <li><span className="font-medium">Google Cloud Platform</span> - For hosting and infrastructure</li>
        </ul>
      </div>

      <div className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Ethics & Limitations</h2>
        <p className="text-gray-700 mb-4">
          We believe in transparent and responsible AI. Here are some important points about our approach:
        </p>
        
        <ul className="list-disc pl-5 space-y-2 text-gray-700">
          <li>We never present absolute verdicts - all results include confidence levels and explanations</li>
          <li>When evidence is insufficient, we clearly state this and guide users on how to verify manually</li>
          <li>We respect user privacy and do not store full claims without explicit consent</li>
          <li>Our system has limitations and should be used as one tool among many for evaluating information</li>
          <li>We continuously work to improve our algorithms and reduce potential biases</li>
        </ul>
      </div>
    </div>
  );
}

export default AboutPage;