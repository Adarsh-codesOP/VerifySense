import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircleIcon, XCircleIcon, ExclamationTriangleIcon, ArrowLeftIcon, ShareIcon, DocumentArrowDownIcon } from '@heroicons/react/24/outline';

function ResultsPage() {
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState('evidence');
  const [expandedExplanation, setExpandedExplanation] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // In a real implementation, this would come from the API response
    // For demo purposes, we'll retrieve it from localStorage or use mock data
    const storedResults = localStorage.getItem('verifyResults');
    
    if (storedResults) {
      setResults(JSON.parse(storedResults));
    } else {
      // Mock data for demo purposes
      setResults({
        status: 'success',
        results: [
          {
            claim: 'COVID-19 vaccines contain microchips to track people',
            fact_checks: [
              {
                claim_date: '2021-01-15',
                claim_source: 'Social Media',
                claim_reviewed: 'COVID-19 vaccines contain microchips to track people',
                publisher: {
                  name: 'PolitiFact',
                  site: 'politifact.com'
                },
                rating: 'False',
                rating_explanation: 'COVID-19 vaccines do not contain microchips or tracking devices',
                url: 'https://www.politifact.com/factchecks/2021/jan/15/facebook-posts/covid-19-vaccines-dont-contain-microchips-track-pe/'
              }
            ],
            evidence: [
              {
                title: 'Fact check: COVID-19 vaccines don\'t contain microchips',
                snippet: 'Medical experts and the manufacturers of the vaccines have repeatedly debunked the claim that COVID-19 vaccines contain microchips or any kind of tracking device.',
                link: 'https://www.reuters.com/article/factcheck-coronavirus-vaccine-microchip/fact-check-covid-19-vaccines-dont-contain-microchips-idUSL1N2M70MW',
                source: 'reuters.com',
                reliability: 'high',
                date: '2021-04-23'
              },
              {
                title: 'No, the COVID-19 vaccine doesn\'t contain a microchip',
                snippet: 'The COVID-19 vaccine does not contain a microchip or any other kind of tracking device. This false claim has been widely debunked by medical professionals.',
                link: 'https://www.bbc.com/news/technology-56570993',
                source: 'bbc.com',
                reliability: 'high',
                date: '2021-03-29'
              },
              {
                title: 'COVID-19 Vaccine Ingredients Explained',
                snippet: 'A detailed breakdown of what\'s actually in the COVID-19 vaccines, explaining each ingredient and its purpose. No microchips or tracking devices are present.',
                link: 'https://www.cdc.gov/coronavirus/2019-ncov/vaccines/different-vaccines/overview-mrna.html',
                source: 'cdc.gov',
                reliability: 'high',
                date: '2021-05-04'
              }
            ],
            score: {
              score: 25,
              confidence_label: 'Likely False',
              components: {
                claim_match_score: 20,
                source_reliability_avg: 80,
                cross_source_consistency: 80,
                temporal_relevance: 70
              }
            },
            explanation: {
              summary: 'This claim has been thoroughly debunked by multiple fact-checking organizations and health authorities. The COVID-19 vaccines do not contain microchips or any tracking devices. This is a common piece of misinformation that has spread on social media.',
              steps: [
                'Check official fact-checking websites like PolitiFact, which has rated this claim as False',
                'Verify the ingredients of COVID-19 vaccines from official sources like the CDC or FDA',
                'Look for reporting from multiple reliable news sources like Reuters and BBC, which have all debunked this claim',
                'Consider the technical feasibility - microchips small enough to fit through a vaccine needle with tracking capabilities do not currently exist'
              ]
            }
          }
        ]
      });
    }
  }, []);

  if (!results || !results.results || results.results.length === 0) {
    return (
      <div className="max-w-3xl mx-auto text-center py-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">No results found</h2>
        <p className="text-gray-600 mb-6">Please submit content to verify on the home page.</p>
        <button
          onClick={() => navigate('/')}
          className="btn btn-primary"
        >
          Go to Home
        </button>
      </div>
    );
  }

  // For simplicity, we'll just use the first result
  const result = results.results[0];

  const getScoreBadgeColor = (label) => {
    switch (label) {
      case 'Likely True':
        return 'bg-success-100 text-success-800';
      case 'Likely False':
        return 'bg-danger-100 text-danger-800';
      default:
        return 'bg-warning-100 text-warning-800';
    }
  };

  const getScoreIcon = (label) => {
    switch (label) {
      case 'Likely True':
        return <CheckCircleIcon className="h-8 w-8 text-success-600" />;
      case 'Likely False':
        return <XCircleIcon className="h-8 w-8 text-danger-600" />;
      default:
        return <ExclamationTriangleIcon className="h-8 w-8 text-warning-600" />;
    }
  };

  const handleShare = () => {
    // In a real implementation, this would generate a shareable link or report
    alert('Sharing functionality would be implemented in the full version');
  };

  const handleDownload = () => {
    // In a real implementation, this would generate a PDF report
    alert('Download report functionality would be implemented in the full version');
  };

  return (
    <div className="max-w-4xl mx-auto">
      <button
        onClick={() => navigate('/')}
        className="flex items-center text-primary-600 hover:text-primary-800 mb-6"
      >
        <ArrowLeftIcon className="h-4 w-4 mr-1" />
        Back to Home
      </button>

      <div className="card mb-6">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">Verification Results</h1>
            <p className="text-gray-600">Claim analyzed: <span className="font-medium">{result.claim}</span></p>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={handleShare}
              className="btn btn-outline flex items-center"
            >
              <ShareIcon className="h-4 w-4 mr-1" />
              Share
            </button>
            <button
              onClick={handleDownload}
              className="btn btn-outline flex items-center"
            >
              <DocumentArrowDownIcon className="h-4 w-4 mr-1" />
              Download
            </button>
          </div>
        </div>

        <div className="flex items-center space-x-4 p-4 rounded-lg bg-gray-50 mb-6">
          {getScoreIcon(result.score.confidence_label)}
          <div>
            <div className={`badge ${getScoreBadgeColor(result.score.confidence_label)} mb-1`}>
              {result.score.confidence_label}
            </div>
            <p className="text-sm text-gray-600">Confidence Score: {result.score.score}/100</p>
          </div>
        </div>

        <div className="mb-6">
          <div className="flex border-b border-gray-200">
            <button
              className={`px-4 py-2 font-medium text-sm ${activeTab === 'evidence' ? 'text-primary-600 border-b-2 border-primary-600' : 'text-gray-500 hover:text-gray-700'}`}
              onClick={() => setActiveTab('evidence')}
            >
              Evidence
            </button>
            <button
              className={`px-4 py-2 font-medium text-sm ${activeTab === 'factChecks' ? 'text-primary-600 border-b-2 border-primary-600' : 'text-gray-500 hover:text-gray-700'}`}
              onClick={() => setActiveTab('factChecks')}
            >
              Fact Checks
            </button>
            <button
              className={`px-4 py-2 font-medium text-sm ${activeTab === 'scoring' ? 'text-primary-600 border-b-2 border-primary-600' : 'text-gray-500 hover:text-gray-700'}`}
              onClick={() => setActiveTab('scoring')}
            >
              Scoring
            </button>
          </div>

          <div className="py-4">
            {activeTab === 'evidence' && (
              <div className="space-y-4">
                {result.evidence.map((item, index) => (
                  <div key={index} className="p-4 border border-gray-200 rounded-lg">
                    <div className="flex justify-between items-start mb-2">
                      <h3 className="font-medium text-gray-900">{item.title}</h3>
                      <span className={`badge ${item.reliability === 'high' ? 'bg-success-100 text-success-800' : 'bg-warning-100 text-warning-800'}`}>
                        {item.reliability === 'high' ? 'Highly Reliable' : 'Medium Reliability'}
                      </span>
                    </div>
                    <p className="text-gray-600 text-sm mb-2">{item.snippet}</p>
                    <div className="flex justify-between items-center text-xs text-gray-500">
                      <span>{item.source}</span>
                      <span>{item.date}</span>
                    </div>
                    <a
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-2 inline-block text-primary-600 hover:text-primary-800 text-sm"
                    >
                      Read full article →
                    </a>
                  </div>
                ))}
              </div>
            )}

            {activeTab === 'factChecks' && (
              <div className="space-y-4">
                {result.fact_checks.length > 0 ? (
                  result.fact_checks.map((check, index) => (
                    <div key={index} className="p-4 border border-gray-200 rounded-lg">
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="font-medium text-gray-900">{check.publisher.name}</h3>
                        <span className={`badge ${check.rating.toLowerCase().includes('false') ? 'bg-danger-100 text-danger-800' : check.rating.toLowerCase().includes('true') ? 'bg-success-100 text-success-800' : 'bg-warning-100 text-warning-800'}`}>
                          {check.rating}
                        </span>
                      </div>
                      <p className="text-gray-600 text-sm mb-2">{check.rating_explanation}</p>
                      <div className="flex justify-between items-center text-xs text-gray-500">
                        <span>Claimed by: {check.claim_source}</span>
                        <span>Date: {check.claim_date}</span>
                      </div>
                      <a
                        href={check.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="mt-2 inline-block text-primary-600 hover:text-primary-800 text-sm"
                      >
                        View fact check →
                      </a>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-6 text-gray-500">
                    <p>No direct fact checks found for this claim.</p>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'scoring' && (
              <div className="space-y-4">
                <div className="p-4 border border-gray-200 rounded-lg">
                  <h3 className="font-medium text-gray-900 mb-4">How we calculated the score</h3>
                  
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-gray-600">Claim Match Score</span>
                        <span className="text-sm font-medium">{result.score.components.claim_match_score}/100</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-primary-600 h-2 rounded-full"
                          style={{ width: `${result.score.components.claim_match_score}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">How closely this matches known fact-checked claims</p>
                    </div>
                    
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-gray-600">Source Reliability</span>
                        <span className="text-sm font-medium">{result.score.components.source_reliability_avg}/100</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-primary-600 h-2 rounded-full"
                          style={{ width: `${result.score.components.source_reliability_avg}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">Average reliability of evidence sources</p>
                    </div>
                    
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-gray-600">Cross-Source Consistency</span>
                        <span className="text-sm font-medium">{result.score.components.cross_source_consistency}/100</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-primary-600 h-2 rounded-full"
                          style={{ width: `${result.score.components.cross_source_consistency}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">Agreement across multiple reliable sources</p>
                    </div>
                    
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm text-gray-600">Temporal Relevance</span>
                        <span className="text-sm font-medium">{result.score.components.temporal_relevance}/100</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-primary-600 h-2 rounded-full"
                          style={{ width: `${result.score.components.temporal_relevance}%` }}
                        ></div>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">Recency and timeliness of the information</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="card mb-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">How We Checked This</h2>
        
        <p className="text-gray-700 mb-4">{result.explanation.summary}</p>
        
        <div className={`space-y-4 ${expandedExplanation ? '' : 'hidden'}`}>
          {result.explanation.steps.map((step, index) => (
            <div key={index} className="flex">
              <div className="flex-shrink-0 mr-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary-100 text-primary-800 font-medium">
                  {index + 1}
                </div>
              </div>
              <div className="pt-1">
                <p className="text-gray-700">{step}</p>
              </div>
            </div>
          ))}
        </div>
        
        <button
          onClick={() => setExpandedExplanation(!expandedExplanation)}
          className="mt-4 text-primary-600 hover:text-primary-800 font-medium"
        >
          {expandedExplanation ? 'Show Less' : 'Learn How to Verify This Yourself'}
        </button>
      </div>
    </div>
  );
}

export default ResultsPage;