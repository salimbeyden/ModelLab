"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { listRuns, getModels, deleteRun, RunState, PluginMeta } from '@/app/lib/api';

export default function ModelsPage() {
    const [models, setModels] = useState<RunState[]>([]);
    const [pluginMeta, setPluginMeta] = useState<Record<string, PluginMeta>>({});
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState<'all' | 'regression' | 'classification'>('all');
    const [sortBy, setSortBy] = useState<'date' | 'accuracy'>('date');
    const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
    const [deleting, setDeleting] = useState(false);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        setLoading(true);
        try {
            const [completedRuns, plugins] = await Promise.all([
                listRuns('completed'),
                getModels()
            ]);
            setModels(completedRuns);
            
            // Create lookup map for plugin metadata
            const metaMap: Record<string, PluginMeta> = {};
            plugins.forEach(p => metaMap[p.id] = p);
            setPluginMeta(metaMap);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleDelete = async (id: string) => {
        setDeleting(true);
        try {
            await deleteRun(id);
            setModels(prev => prev.filter(m => m.id !== id));
            setDeleteConfirm(null);
        } catch (err) {
            console.error('Failed to delete model:', err);
            alert('Failed to delete model');
        } finally {
            setDeleting(false);
        }
    };

    const filteredModels = models.filter(m => {
        if (filter === 'all') return true;
        return m.config.task === filter;
    });

    const sortedModels = [...filteredModels].sort((a, b) => {
        if (sortBy === 'date') {
            return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        }
        // Sort by primary metric (r2 for regression, accuracy for classification)
        const metricA = a.config.task === 'regression' ? (a.metrics?.r2 || 0) : (a.metrics?.accuracy || 0);
        const metricB = b.config.task === 'regression' ? (b.metrics?.r2 || 0) : (b.metrics?.accuracy || 0);
        return metricB - metricA;
    });

    const getModelIcon = (modelId: string) => {
        switch (modelId) {
            case 'ebm':
                return (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                );
            case 'mgcv':
                return (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                    </svg>
                );
            default:
                return (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                );
        }
    };

    const getPrimaryMetric = (run: RunState) => {
        if (!run.metrics) return null;
        if (run.config.task === 'regression') {
            return { label: 'R²', value: run.metrics.r2?.toFixed(4) || 'N/A' };
        } else {
            return { label: 'Accuracy', value: run.metrics.accuracy?.toFixed(4) || 'N/A' };
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-gray-50">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 p-8">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold text-gray-900 mb-2">Trained Models</h1>
                    <p className="text-gray-600">Browse and reuse your trained machine learning models</p>
                </div>

                {/* Filters & Controls */}
                <div className="bg-white rounded-lg shadow-sm p-4 mb-6 flex flex-wrap gap-4 items-center justify-between">
                    <div className="flex gap-2">
                        <button
                            onClick={() => setFilter('all')}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                                filter === 'all'
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                            }`}
                        >
                            All ({models.length})
                        </button>
                        <button
                            onClick={() => setFilter('regression')}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                                filter === 'regression'
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                            }`}
                        >
                            Regression ({models.filter(m => m.config.task === 'regression').length})
                        </button>
                        <button
                            onClick={() => setFilter('classification')}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                                filter === 'classification'
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                            }`}
                        >
                            Classification ({models.filter(m => m.config.task === 'classification').length})
                        </button>
                    </div>

                    <div className="flex items-center gap-2">
                        <label className="text-sm text-gray-600">Sort by:</label>
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value as 'date' | 'accuracy')}
                            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                            <option value="date">Date (Newest)</option>
                            <option value="accuracy">Performance</option>
                        </select>
                    </div>
                </div>

                {/* Models Grid */}
                {sortedModels.length === 0 ? (
                    <div className="bg-white rounded-lg shadow-sm p-12 text-center">
                        <svg className="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                        <h3 className="text-lg font-medium text-gray-900 mb-2">No trained models yet</h3>
                        <p className="text-gray-500 mb-6">Start training your first model to see it here</p>
                        <Link
                            href="/runs/new"
                            className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                        >
                            <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                            </svg>
                            New Training Run
                        </Link>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {sortedModels.map((model) => {
                            const metric = getPrimaryMetric(model);
                            const plugin = pluginMeta[model.config.model_id];
                            
                            return (
                                <div key={model.id} className="relative group">
                                <Link
                                    href={`/runs/${model.id}`}
                                    className="block bg-white rounded-lg shadow-sm border border-gray-200 hover:shadow-md hover:border-blue-300 transition-all overflow-hidden"
                                >
                                    {/* Header */}
                                    <div className="p-6 pb-4">
                                        <div className="flex items-start justify-between mb-4">
                                            <div className="flex items-center gap-3">
                                                <div className={`p-3 rounded-lg ${
                                                    model.config.model_id === 'ebm'
                                                        ? 'bg-blue-100 text-blue-600'
                                                        : 'bg-purple-100 text-purple-600'
                                                }`}>
                                                    {getModelIcon(model.config.model_id)}
                                                </div>
                                                <div>
                                                    <h3 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
                                                        {plugin?.name || model.config.model_id.toUpperCase()}
                                                    </h3>
                                                    <p className="text-xs text-gray-500 mt-0.5">
                                                        {new Date(model.created_at).toLocaleDateString()}
                                                    </p>
                                                </div>
                                            </div>
                                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                                                model.config.task === 'regression'
                                                    ? 'bg-green-100 text-green-700'
                                                    : 'bg-yellow-100 text-yellow-700'
                                            }`}>
                                                {model.config.task}
                                            </span>
                                        </div>

                                        {/* Target Info */}
                                        <div className="mb-4">
                                            <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Target</div>
                                            <div className="font-mono text-sm bg-gray-50 px-2 py-1 rounded inline-block">
                                                {model.config.target}
                                            </div>
                                        </div>

                                        {/* Primary Metric */}
                                        {metric && (
                                            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4">
                                                <div className="text-xs text-gray-600 uppercase tracking-wider mb-1">
                                                    {metric.label}
                                                </div>
                                                <div className="text-2xl font-bold text-gray-900">
                                                    {metric.value}
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    {/* Footer with Actions */}
                                    <div className="px-6 py-3 bg-gray-50 border-t border-gray-100 flex items-center justify-between text-xs">
                                        <span className="text-gray-500">
                                            {model.artifacts.length} artifacts
                                        </span>
                                        <span className="text-blue-600 group-hover:text-blue-700 font-medium flex items-center gap-1">
                                            View Details
                                            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                            </svg>
                                        </span>
                                    </div>
                                </Link>
                                
                                {/* Action buttons - Separate from main card link */}
                                <div className="absolute bottom-14 right-4 opacity-0 group-hover:opacity-100 transition-opacity flex gap-2">
                                    <Link
                                        href={`/models/${model.id}/dashboard`}
                                        onClick={(e) => e.stopPropagation()}
                                        className="inline-flex items-center gap-1 px-3 py-1.5 bg-indigo-600 text-white text-xs font-medium rounded-md hover:bg-indigo-700 shadow-sm"
                                    >
                                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                        </svg>
                                        Dashboard
                                    </Link>
                                    <button
                                        onClick={(e) => {
                                            e.preventDefault();
                                            e.stopPropagation();
                                            setDeleteConfirm(model.id);
                                        }}
                                        className="inline-flex items-center gap-1 px-3 py-1.5 bg-red-600 text-white text-xs font-medium rounded-md hover:bg-red-700 shadow-sm"
                                    >
                                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                        </svg>
                                        Delete
                                    </button>
                                </div>
                                </div>
                            );
                        })}
                    </div>
                )}

                {/* Stats Footer */}
                {sortedModels.length > 0 && (
                    <div className="mt-8 bg-white rounded-lg shadow-sm p-6">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                            <div>
                                <div className="text-sm text-gray-500">Total Models</div>
                                <div className="text-2xl font-bold text-gray-900">{models.length}</div>
                            </div>
                            <div>
                                <div className="text-sm text-gray-500">Regression</div>
                                <div className="text-2xl font-bold text-green-600">
                                    {models.filter(m => m.config.task === 'regression').length}
                                </div>
                            </div>
                            <div>
                                <div className="text-sm text-gray-500">Classification</div>
                                <div className="text-2xl font-bold text-yellow-600">
                                    {models.filter(m => m.config.task === 'classification').length}
                                </div>
                            </div>
                            <div>
                                <div className="text-sm text-gray-500">Avg Performance</div>
                                <div className="text-2xl font-bold text-blue-600">
                                    {(() => {
                                        const scores = models
                                            .map(m => m.config.task === 'regression' ? m.metrics?.r2 : m.metrics?.accuracy)
                                            .filter(s => s !== undefined) as number[];
                                        return scores.length > 0
                                            ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(3)
                                            : 'N/A';
                                    })()}
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Delete Confirmation Modal */}
            {deleteConfirm && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4">
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">Delete Model</h3>
                        <p className="text-gray-600 mb-6">
                            Are you sure you want to delete this model? This action cannot be undone and will remove all associated artifacts.
                        </p>
                        <div className="flex justify-end gap-3">
                            <button
                                onClick={() => setDeleteConfirm(null)}
                                disabled={deleting}
                                className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 disabled:opacity-50"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={() => handleDelete(deleteConfirm)}
                                disabled={deleting}
                                className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-md hover:bg-red-700 disabled:opacity-50 flex items-center gap-2"
                            >
                                {deleting && (
                                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                )}
                                {deleting ? 'Deleting...' : 'Delete'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
