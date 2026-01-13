"use client";

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import {
    getDatasetProfile,
    getDatasets,
    getPreprocessingPreview,
    savePreprocessingConfig,
    getPreprocessingConfig,
    Dataset,
    DatasetProfile,
    PreprocessingConfig,
    ColumnConfig,
    PreprocessedPreview,
    ImputationStrategy,
    DataTypeOverride
} from '@/app/lib/api';

const DATA_TYPES: DataTypeOverride[] = ["auto", "int64", "float64", "object", "bool", "datetime64"];
const NUMERIC_IMPUTATION: ImputationStrategy[] = ["none", "mean", "median", "mode", "constant", "drop_rows"];
const CATEGORICAL_IMPUTATION: ImputationStrategy[] = ["none", "mode", "unknown", "constant", "drop_rows"];

function getImputationOptions(dtype: string): ImputationStrategy[] {
    if (dtype.includes('int') || dtype.includes('float')) {
        return NUMERIC_IMPUTATION;
    }
    return CATEGORICAL_IMPUTATION;
}

export default function PreprocessingPage() {
    const params = useParams();
    const router = useRouter();
    const id = params.id as string;

    const [dataset, setDataset] = useState<Dataset | null>(null);
    const [profile, setProfile] = useState<DatasetProfile | null>(null);
    const [config, setConfig] = useState<PreprocessingConfig>({
        columns: {},
        apply_one_hot_encoding: true,
        apply_scaling: true
    });
    const [preview, setPreview] = useState<PreprocessedPreview | null>(null);
    const [loading, setLoading] = useState(true);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        if (!id) return;

        setLoading(true);

        Promise.all([
            getDatasets().then(list => list.find(d => d.id === id) || null),
            getDatasetProfile(id),
            getPreprocessingConfig(id)
        ])
            .then(([foundDataset, profileData, savedConfig]) => {
                setDataset(foundDataset);
                setProfile(profileData);

                // Initialize column configs from profile if not already saved
                const allColumns = Object.keys(profileData.dtypes);
                const initialColumns: Record<string, ColumnConfig> = {};

                allColumns.forEach(col => {
                    initialColumns[col] = savedConfig.columns[col] || {
                        include: true,
                        dtype_override: "auto",
                        imputation_strategy: "none",
                        constant_value: undefined
                    };
                });

                setConfig({
                    ...savedConfig,
                    columns: { ...initialColumns, ...savedConfig.columns }
                });

                setLoading(false);
            })
            .catch(err => {
                console.error(err);
                setLoading(false);
            });
    }, [id]);

    const updateColumnConfig = (col: string, updates: Partial<ColumnConfig>) => {
        setConfig(prev => ({
            ...prev,
            columns: {
                ...prev.columns,
                [col]: { ...prev.columns[col], ...updates }
            }
        }));
    };

    const handlePreview = async () => {
        setPreviewLoading(true);
        try {
            const result = await getPreprocessingPreview(id, config);
            setPreview(result);
        } catch (e: any) {
            alert("Preview failed: " + (e.response?.data?.detail || e.message));
        } finally {
            setPreviewLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            await savePreprocessingConfig(id, config);
            alert("Configuration saved!");
        } catch (e: any) {
            alert("Save failed: " + (e.response?.data?.detail || e.message));
        } finally {
            setSaving(false);
        }
    };

    if (loading) return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
    );

    const allColumns = profile ? Object.keys(profile.dtypes) : [];

    return (
        <div className="min-h-screen bg-gray-50 p-8 font-sans">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="flex justify-between items-start mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900">Configure Preprocessing</h1>
                        <p className="text-gray-500 mt-1">{dataset?.name || dataset?.filename}</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <Link href={`/datasets/${id}`} className="text-gray-600 hover:text-gray-900 font-medium">
                            &larr; Back to Details
                        </Link>
                        <button
                            onClick={handlePreview}
                            disabled={previewLoading}
                            className="px-4 py-2 border border-blue-600 text-blue-600 rounded-md hover:bg-blue-50 disabled:opacity-50"
                        >
                            {previewLoading ? "Loading..." : "Preview"}
                        </button>
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                        >
                            {saving ? "Saving..." : "Save Configuration"}
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Left: Column Configuration */}
                    <div className="bg-white shadow rounded-lg overflow-hidden">
                        <div className="px-4 py-4 border-b border-gray-100 flex justify-between items-center">
                            <h2 className="text-lg font-medium text-gray-900">Column Configuration</h2>
                            <span className="text-sm text-gray-500">{allColumns.length} columns</span>
                        </div>
                        <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gray-50 sticky top-0">
                                    <tr>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Include</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Column</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Override</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Nulls</th>
                                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Imputation</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200">
                                    {allColumns.map(col => {
                                        const colConfig = config.columns[col] || { include: true, dtype_override: "auto", imputation_strategy: "none" };
                                        const dtype = profile?.dtypes[col] || "unknown";
                                        const nullCount = profile?.null_counts[col] || 0;
                                        const imputationOptions = getImputationOptions(dtype);

                                        return (
                                            <tr key={col} className={colConfig.include ? "" : "bg-gray-50 opacity-60"}>
                                                <td className="px-4 py-3">
                                                    <input
                                                        type="checkbox"
                                                        checked={colConfig.include}
                                                        onChange={e => updateColumnConfig(col, { include: e.target.checked })}
                                                        className="h-4 w-4 text-blue-600 rounded"
                                                    />
                                                </td>
                                                <td className="px-4 py-3 text-sm font-medium text-gray-900">{col}</td>
                                                <td className="px-4 py-3">
                                                    <span className={`text-xs px-2 py-1 rounded-full ${dtype.includes('int') || dtype.includes('float')
                                                            ? 'bg-green-100 text-green-800'
                                                            : 'bg-yellow-100 text-yellow-800'
                                                        }`}>
                                                        {dtype}
                                                    </span>
                                                </td>
                                                <td className="px-4 py-3">
                                                    <select
                                                        value={colConfig.dtype_override}
                                                        onChange={e => updateColumnConfig(col, { dtype_override: e.target.value as DataTypeOverride })}
                                                        className="text-sm border-gray-300 rounded-md"
                                                        disabled={!colConfig.include}
                                                    >
                                                        {DATA_TYPES.map(t => (
                                                            <option key={t} value={t}>{t}</option>
                                                        ))}
                                                    </select>
                                                </td>
                                                <td className="px-4 py-3">
                                                    <span className={nullCount > 0 ? 'text-red-600 font-medium' : 'text-gray-500'}>
                                                        {nullCount}
                                                    </span>
                                                </td>
                                                <td className="px-4 py-3">
                                                    <select
                                                        value={colConfig.imputation_strategy}
                                                        onChange={e => updateColumnConfig(col, { imputation_strategy: e.target.value as ImputationStrategy })}
                                                        className="text-sm border-gray-300 rounded-md"
                                                        disabled={!colConfig.include || nullCount === 0}
                                                    >
                                                        {imputationOptions.map(s => (
                                                            <option key={s} value={s}>{s}</option>
                                                        ))}
                                                    </select>
                                                </td>
                                            </tr>
                                        );
                                    })}
                                </tbody>
                            </table>
                        </div>
                        {/* Global Options */}
                        <div className="px-4 py-4 border-t border-gray-100 bg-gray-50 space-y-3">
                            <h3 className="text-sm font-medium text-gray-700">Global Transformations</h3>
                            <label className="flex items-center gap-2">
                                <input
                                    type="checkbox"
                                    checked={config.apply_one_hot_encoding}
                                    onChange={e => setConfig(prev => ({ ...prev, apply_one_hot_encoding: e.target.checked }))}
                                    className="h-4 w-4 text-blue-600 rounded"
                                />
                                <span className="text-sm">One-Hot Encode categorical columns</span>
                            </label>
                            <label className="flex items-center gap-2">
                                <input
                                    type="checkbox"
                                    checked={config.apply_scaling}
                                    onChange={e => setConfig(prev => ({ ...prev, apply_scaling: e.target.checked }))}
                                    className="h-4 w-4 text-blue-600 rounded"
                                />
                                <span className="text-sm">Standard Scale numeric columns</span>
                            </label>
                        </div>
                    </div>

                    {/* Right: Preview */}
                    <div className="bg-white shadow rounded-lg overflow-hidden">
                        <div className="px-4 py-4 border-b border-gray-100">
                            <h2 className="text-lg font-medium text-gray-900">Processed Data Preview</h2>
                        </div>
                        {preview ? (
                            <>
                                <div className="px-4 py-3 bg-blue-50 border-b border-blue-100">
                                    <p className="text-sm text-blue-800">
                                        <strong>{preview.row_count}</strong> rows &bull; <strong>{preview.columns.length}</strong> columns after processing
                                    </p>
                                </div>
                                {/* Transformations Log */}
                                <div className="px-4 py-3 border-b border-gray-100 max-h-40 overflow-y-auto">
                                    <h4 className="text-xs font-medium text-gray-500 uppercase mb-2">Transformations Applied</h4>
                                    <ul className="text-xs text-gray-600 space-y-1">
                                        {preview.applied_transformations.map((t, i) => (
                                            <li key={i} className="flex items-start gap-1">
                                                <span className="text-green-500">✓</span>
                                                {t}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                                {/* Data Table */}
                                <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
                                    <table className="min-w-full divide-y divide-gray-200 text-xs">
                                        <thead className="bg-gray-50 sticky top-0">
                                            <tr>
                                                {preview.columns.map(col => (
                                                    <th key={col} className="px-3 py-2 text-left font-medium text-gray-500">{col}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {preview.data.map((row, i) => (
                                                <tr key={i}>
                                                    {preview.columns.map(col => (
                                                        <td key={col} className="px-3 py-2 whitespace-nowrap text-gray-900">
                                                            {typeof row[col] === 'number'
                                                                ? row[col].toFixed(2)
                                                                : (row[col] ?? <span className="text-gray-400">null</span>)}
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </>
                        ) : (
                            <div className="p-12 text-center text-gray-400">
                                <p>Click "Preview" to see transformed data</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
