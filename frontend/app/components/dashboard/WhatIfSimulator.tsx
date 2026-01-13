"use client";

import { useState } from 'react';
import type { FeatureRange } from '@/app/lib/explanationApi';

interface WhatIfSimulatorProps {
    featureRanges: FeatureRange[];
    featureValues: Record<string, number | string | null>;
    onChange: (name: string, value: number | string) => void;
    onPredict: () => void;
    loading: boolean;
}

export default function WhatIfSimulator({
    featureRanges,
    featureValues,
    onChange,
    onPredict,
    loading,
}: WhatIfSimulatorProps) {
    const [expandedFeatures, setExpandedFeatures] = useState<Set<string>>(new Set());

    const toggleExpand = (name: string) => {
        const newSet = new Set(expandedFeatures);
        if (newSet.has(name)) {
            newSet.delete(name);
        } else {
            newSet.add(name);
        }
        setExpandedFeatures(newSet);
    };

    // Group features: show first 5 by default, rest expandable
    const visibleFeatures = featureRanges.slice(0, 8);
    const hiddenFeatures = featureRanges.slice(8);

    return (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 sticky top-24">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
                🎛️ What-If Simulator
            </h3>
            <p className="text-sm text-gray-500 mb-6">
                Adjust feature values to see how predictions change
            </p>

            <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2">
                {visibleFeatures.map(fr => (
                    <FeatureSlider
                        key={fr.feature_name}
                        featureRange={fr}
                        value={featureValues[fr.feature_name]}
                        onChange={(val) => onChange(fr.feature_name, val)}
                    />
                ))}

                {hiddenFeatures.length > 0 && (
                    <div className="border-t border-gray-200 pt-4 mt-4">
                        <button
                            onClick={() => toggleExpand('more')}
                            className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
                        >
                            {expandedFeatures.has('more') ? '▼' : '▶'} 
                            {hiddenFeatures.length} more features
                        </button>

                        {expandedFeatures.has('more') && (
                            <div className="mt-4 space-y-4">
                                {hiddenFeatures.map(fr => (
                                    <FeatureSlider
                                        key={fr.feature_name}
                                        featureRange={fr}
                                        value={featureValues[fr.feature_name]}
                                        onChange={(val) => onChange(fr.feature_name, val)}
                                    />
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </div>

            <button
                onClick={onPredict}
                disabled={loading}
                className="w-full mt-6 px-4 py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
                {loading ? (
                    <>
                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                        Predicting...
                    </>
                ) : (
                    <>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        Predict
                    </>
                )}
            </button>

            <button
                onClick={() => {
                    // Reset to mean/mode values
                    featureRanges.forEach(fr => {
                        if (fr.feature_type === 'numeric' && fr.numeric_range) {
                            onChange(fr.feature_name, fr.numeric_range.mean);
                        } else if (fr.feature_type === 'categorical' && fr.categorical_range) {
                            onChange(fr.feature_name, fr.categorical_range.mode);
                        }
                    });
                }}
                className="w-full mt-2 px-4 py-2 text-gray-600 font-medium rounded-lg border border-gray-300 hover:bg-gray-50"
            >
                Reset to Defaults
            </button>
        </div>
    );
}

// Individual feature slider component
function FeatureSlider({
    featureRange,
    value,
    onChange,
}: {
    featureRange: FeatureRange;
    value: number | string | null | undefined;
    onChange: (val: number | string) => void;
}) {
    const { feature_name, feature_type, numeric_range, categorical_range } = featureRange;

    if (feature_type === 'categorical' && categorical_range) {
        return (
            <div>
                <label className="block text-sm font-medium text-gray-700 mb-1 truncate" title={feature_name}>
                    {feature_name}
                </label>
                <select
                    value={value as string || categorical_range.mode}
                    onChange={(e) => onChange(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                    {categorical_range.categories.map(cat => (
                        <option key={cat} value={cat}>{cat}</option>
                    ))}
                </select>
            </div>
        );
    }

    if (feature_type === 'numeric' && numeric_range) {
        const { min, max, mean } = numeric_range;
        const currentValue = typeof value === 'number' ? value : mean;
        const step = (max - min) / 100;

        return (
            <div>
                <div className="flex justify-between items-center mb-1">
                    <label className="text-sm font-medium text-gray-700 truncate max-w-[60%]" title={feature_name}>
                        {feature_name}
                    </label>
                    <span className="text-sm font-mono text-blue-600">
                        {currentValue.toFixed(2)}
                    </span>
                </div>
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={currentValue}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                    <span>{min.toFixed(1)}</span>
                    <span className="text-gray-500">μ={mean.toFixed(1)}</span>
                    <span>{max.toFixed(1)}</span>
                </div>
            </div>
        );
    }

    return null;
}
