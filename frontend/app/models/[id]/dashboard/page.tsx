"use client";

import { useState } from 'react';
import { useParams } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import Link from 'next/link';
import { getDashboardData, getWhatIfPrediction } from '@/app/lib/explanationApi';
import type { ModelDashboardData, FeatureRange } from '@/app/lib/explanationApi';

// Dashboard Components
import {
    ShapePlot,
    WhatIfSimulator,
    MetricsCards,
    ResidualPlot,
    FeatureImportanceChart,
    ActualVsPredictedPlot,
    WaterfallChart,
    ICEPlot,
} from '@/app/components/dashboard';

type TabType = 'overview' | 'shapes' | 'whatif' | 'performance';

export default function ModelDashboardPage() {
    const params = useParams();
    const runId = params.id as string;
    
    const [activeTab, setActiveTab] = useState<TabType>('overview');
    const [selectedFeature, setSelectedFeature] = useState<string>('');
    
    // TanStack Query for dashboard data
    const { data: dashboardData, isLoading, error } = useQuery({
        queryKey: ['dashboard', runId],
        queryFn: () => getDashboardData(runId),
        enabled: !!runId,
    });

    // What-If state
    const [featureValues, setFeatureValues] = useState<Record<string, number | string | null>>({});
    
    const { data: whatIfResult, refetch: refetchWhatIf, isFetching: whatIfLoading } = useQuery({
        queryKey: ['whatif', runId, featureValues],
        queryFn: () => getWhatIfPrediction(runId, featureValues),
        enabled: !!runId && Object.keys(featureValues).length > 0,
    });

    // Initialize feature values from ranges
    const initializeFeatureValues = (ranges: FeatureRange[]) => {
        const initial: Record<string, number | string | null> = {};
        ranges.forEach(fr => {
            if (fr.feature_type === 'numeric' && fr.numeric_range) {
                initial[fr.feature_name] = fr.numeric_range.mean;
            } else if (fr.feature_type === 'categorical' && fr.categorical_range) {
                initial[fr.feature_name] = fr.categorical_range.mode;
            }
        });
        if (Object.keys(featureValues).length === 0 && Object.keys(initial).length > 0) {
            setFeatureValues(initial);
        }
    };

    // Set initial feature when data loads
    if (dashboardData && !selectedFeature && dashboardData.global_explanation.feature_names.length > 0) {
        setSelectedFeature(dashboardData.global_explanation.feature_names[0]);
        initializeFeatureValues(dashboardData.feature_ranges);
    }

    if (isLoading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                    <p className="mt-4 text-gray-600">Loading model dashboard...</p>
                </div>
            </div>
        );
    }

    if (error || !dashboardData) {
        // Extract meaningful error message
        const errorMessage = error?.message || 'Failed to load dashboard data';
        const isDatasetMissing = errorMessage.includes('Dataset') && errorMessage.includes('not found');
        
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
                <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-lg">
                    <h2 className="text-lg font-semibold text-red-800">Error Loading Dashboard</h2>
                    {isDatasetMissing ? (
                        <div className="mt-2">
                            <p className="text-red-600">The training dataset for this model is no longer available.</p>
                            <p className="text-gray-600 text-sm mt-1">
                                Some features like What-If analysis and performance metrics require the original training data.
                            </p>
                        </div>
                    ) : (
                        <p className="text-red-600 mt-2">{errorMessage}</p>
                    )}
                    <Link href="/models" className="mt-4 inline-block text-blue-600 hover:underline">
                        ← Back to Models
                    </Link>
                </div>
            </div>
        );
    }

    const { summary, global_explanation, performance, feature_ranges } = dashboardData;

    const tabs = [
        { id: 'overview', label: 'Overview', icon: '📊' },
        { id: 'shapes', label: 'Shape Functions', icon: '📈' },
        { id: 'whatif', label: 'What-If Simulator', icon: '🎛️' },
        { id: 'performance', label: 'Performance', icon: '🎯' },
    ];

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white border-b border-gray-200">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="flex items-center gap-3">
                                <Link href="/models" className="text-gray-400 hover:text-gray-600">
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                                    </svg>
                                </Link>
                                <h1 className="text-2xl font-bold text-gray-900">
                                    Model Dashboard
                                </h1>
                                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                                    summary.task_type === 'regression' 
                                        ? 'bg-green-100 text-green-700' 
                                        : 'bg-yellow-100 text-yellow-700'
                                }`}>
                                    {summary.task_type}
                                </span>
                            </div>
                            <p className="text-gray-500 text-sm mt-1">
                                {summary.model_type.toUpperCase()} • Target: {summary.target_variable} • {summary.n_features} features
                            </p>
                        </div>
                        <div className="text-right">
                            <div className="text-3xl font-bold text-blue-600">
                                {summary.primary_metric.toFixed(4)}
                            </div>
                            <div className="text-sm text-gray-500">{summary.primary_metric_name}</div>
                        </div>
                    </div>

                    {/* Tabs */}
                    <div className="flex gap-1 mt-6">
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id as TabType)}
                                className={`px-4 py-2 rounded-t-lg font-medium text-sm transition-colors ${
                                    activeTab === tab.id
                                        ? 'bg-gray-50 text-blue-600 border-t border-l border-r border-gray-200'
                                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                                }`}
                            >
                                <span className="mr-2">{tab.icon}</span>
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="max-w-7xl mx-auto px-6 py-6">
                {/* Overview Tab */}
                {activeTab === 'overview' && (
                    <div className="space-y-6">
                        {/* Metrics Cards */}
                        <MetricsCards performance={performance} summary={summary} />

                        {/* Two Column Layout */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Feature Importance */}
                            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                                <h3 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance</h3>
                                <FeatureImportanceChart 
                                    importance={global_explanation.feature_importance} 
                                />
                            </div>

                            {/* Actual vs Predicted */}
                            {performance.actual_vs_predicted && (
                                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Actual vs Predicted</h3>
                                    <ActualVsPredictedPlot data={performance.actual_vs_predicted} />
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Shape Functions Tab */}
                {activeTab === 'shapes' && (
                    <div className="space-y-6">
                        {/* Feature Selector */}
                        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                Select Feature to Visualize
                            </label>
                            <select
                                value={selectedFeature}
                                onChange={(e) => setSelectedFeature(e.target.value)}
                                className="w-full max-w-md px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                            >
                                {global_explanation.feature_names.map(name => (
                                    <option key={name} value={name}>{name}</option>
                                ))}
                            </select>
                        </div>

                        {/* Shape Plot */}
                        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                            <h3 className="text-lg font-semibold text-gray-900 mb-4">
                                Shape Function: {selectedFeature}
                            </h3>
                            <ShapePlot 
                                shapeFunction={global_explanation.shape_functions.find(
                                    sf => sf.feature_name === selectedFeature
                                )}
                            />
                        </div>

                        {/* All Feature Shapes Grid */}
                        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                            <h3 className="text-lg font-semibold text-gray-900 mb-4">All Feature Shapes</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                {global_explanation.shape_functions.slice(0, 6).map(sf => (
                                    <div 
                                        key={sf.feature_name}
                                        className="border border-gray-200 rounded-lg p-3 cursor-pointer hover:border-blue-300 transition-colors"
                                        onClick={() => setSelectedFeature(sf.feature_name)}
                                    >
                                        <div className="text-sm font-medium text-gray-700 mb-2 truncate">
                                            {sf.feature_name}
                                        </div>
                                        <ShapePlot shapeFunction={sf} mini />
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {/* What-If Simulator Tab */}
                {activeTab === 'whatif' && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Sliders Panel */}
                        <div className="lg:col-span-1">
                            <WhatIfSimulator
                                featureRanges={feature_ranges}
                                featureValues={featureValues}
                                onChange={(name: string, value: number | string | null) => {
                                    setFeatureValues(prev => ({ ...prev, [name]: value }));
                                }}
                                onPredict={() => refetchWhatIf()}
                                loading={whatIfLoading}
                            />
                        </div>

                        {/* Results Panel */}
                        <div className="lg:col-span-2 space-y-6">
                            {/* Prediction Card */}
                            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                                <h3 className="text-lg font-semibold text-gray-900 mb-4">Prediction Result</h3>
                                {whatIfResult ? (
                                    <div className="flex items-center gap-8">
                                        <div>
                                            <div className="text-sm text-gray-500">Baseline</div>
                                            <div className="text-2xl font-bold text-gray-700">
                                                {whatIfResult.explanation.baseline.toFixed(4)}
                                            </div>
                                        </div>
                                        <div className="text-2xl text-gray-400">→</div>
                                        <div>
                                            <div className="text-sm text-gray-500">Prediction</div>
                                            <div className="text-3xl font-bold text-blue-600">
                                                {whatIfResult.explanation.prediction.toFixed(4)}
                                            </div>
                                        </div>
                                        <div className={`text-xl font-medium ${
                                            whatIfResult.explanation.total_contribution >= 0 
                                                ? 'text-green-600' 
                                                : 'text-red-600'
                                        }`}>
                                            {whatIfResult.explanation.total_contribution >= 0 ? '+' : ''}
                                            {whatIfResult.explanation.total_contribution.toFixed(4)}
                                        </div>
                                    </div>
                                ) : (
                                    <p className="text-gray-500 italic">
                                        Adjust feature values and click "Predict" to see results
                                    </p>
                                )}
                            </div>

                            {/* Waterfall Chart */}
                            {whatIfResult && (
                                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                                        Feature Contributions (Waterfall)
                                    </h3>
                                    <WaterfallChart contributions={whatIfResult.explanation.contributions} />
                                </div>
                            )}

                            {/* ICE Plots - Individual Conditional Expectation */}
                            {whatIfResult && whatIfResult.ice_plots && whatIfResult.ice_plots.length > 0 && (
                                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                                        Feature Effect Curves (ICE Plots)
                                    </h3>
                                    <p className="text-sm text-gray-500 mb-4">
                                        Shows how the prediction changes as each feature varies. The red dot/line indicates the current value.
                                    </p>
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        {whatIfResult.ice_plots.map(icePlot => (
                                            <div 
                                                key={icePlot.feature_name}
                                                className="border border-gray-200 rounded-lg p-3"
                                            >
                                                <div className="text-sm font-medium text-gray-700 mb-2 truncate">
                                                    {icePlot.feature_name}
                                                    <span className="text-gray-400 ml-2">
                                                        (current: {typeof icePlot.current_value === 'number' 
                                                            ? icePlot.current_value.toFixed(2) 
                                                            : icePlot.current_value})
                                                    </span>
                                                </div>
                                                <ICEPlot icePlot={icePlot} mini />
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Performance Tab */}
                {activeTab === 'performance' && (
                    <div className="space-y-6">
                        {/* Detailed Metrics */}
                        <MetricsCards performance={performance} summary={summary} detailed />

                        {/* Residual Analysis */}
                        {performance.residuals && (
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Residual Distribution</h3>
                                    <ResidualPlot residuals={performance.residuals} type="scatter" />
                                </div>
                                <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Residual Histogram</h3>
                                    <ResidualPlot residuals={performance.residuals} type="histogram" />
                                </div>
                            </div>
                        )}

                        {/* Actual vs Predicted (Full Size) */}
                        {performance.actual_vs_predicted && (
                            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                                <h3 className="text-lg font-semibold text-gray-900 mb-4">Actual vs Predicted</h3>
                                <ActualVsPredictedPlot data={performance.actual_vs_predicted} fullSize />
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
