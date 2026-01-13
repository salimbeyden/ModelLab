"use client";

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import {
    getDatasetProfile,
    getDatasets,
    getDataQuality,
    getPreparationPreview,
    saveCleaningConfig,
    saveDatasetVersion,
    Dataset,
    DatasetProfile,
    DataQualityReport,
    DataCleaningConfig,
    NumericColumnConfig,
    CategoricalColumnConfig,
    EnhancedPreview,
    ImputationStrategy,
    OutlierStrategy,
    ScalingMethod,
    EncodingMethod,
    DataTypeOverride
} from '@/app/lib/api';

type TabType = 'overview' | 'cleaning' | 'transformation' | 'preview';

const DATA_TYPE_OPTIONS: DataTypeOverride[] = ["auto", "int64", "float64", "object", "bool", "datetime64"];
const NUMERIC_IMPUTATION: ImputationStrategy[] = ["none", "mean", "median", "mode", "constant", "drop_rows"];
const CATEGORICAL_IMPUTATION: ImputationStrategy[] = ["none", "mode", "unknown", "constant", "drop_rows"];
const OUTLIER_OPTIONS: OutlierStrategy[] = ["none", "remove", "clip_iqr", "clip_zscore"];
const SCALING_OPTIONS: ScalingMethod[] = ["none", "standard", "minmax", "robust"];
const ENCODING_OPTIONS: EncodingMethod[] = ["none", "onehot", "label", "frequency"];

const DataQualityOverview = ({ quality }: { quality: DataQualityReport }) => (
    <div className="space-y-6">
        {/* Summary Cards */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-white rounded-lg shadow p-4">
                <div className="text-sm text-gray-500">Total Rows</div>
                <div className="text-2xl font-bold text-gray-900">{quality.total_rows.toLocaleString()}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
                <div className="text-sm text-gray-500">Total Columns</div>
                <div className="text-2xl font-bold text-gray-900">{quality.total_columns}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
                <div className="text-sm text-gray-500">Numeric</div>
                <div className="text-2xl font-bold text-green-600">{quality.numeric_columns.length}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
                <div className="text-sm text-gray-500">Categorical</div>
                <div className="text-2xl font-bold text-yellow-600">{quality.categorical_columns.length}</div>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
                <div className="text-sm text-gray-500">Duplicates</div>
                <div className={`text-2xl font-bold ${quality.duplicate_rows > 0 ? 'text-red-600' : 'text-gray-900'}`}>
                    {quality.duplicate_rows} ({quality.duplicate_percentage}%)
                </div>
            </div>
        </div>

        {/* Numeric Columns */}
        {quality.numeric_columns.length > 0 && (
            <div className="bg-white rounded-lg shadow">
                <div className="px-4 py-3 border-b border-gray-100">
                    <h3 className="font-medium text-gray-900">Numeric Columns ({quality.numeric_columns.length})</h3>
                </div>
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 text-sm">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Column</th>
                                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Nulls</th>
                                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Outliers</th>
                                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Unique</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {quality.numeric_columns.map(col => (
                                <tr key={col.name}>
                                    <td className="px-4 py-2 font-medium">{col.name}</td>
                                    <td className="px-4 py-2"><span className="px-2 py-0.5 rounded-full bg-green-100 text-green-800 text-xs">{col.dtype}</span></td>
                                    <td className="px-4 py-2"><span className={col.null_count > 0 ? 'text-red-600 font-medium' : ''}>{col.null_count} ({col.null_percentage}%)</span></td>
                                    <td className="px-4 py-2"><span className={col.outlier_count > 0 ? 'text-orange-600 font-medium' : ''}>{col.outlier_count} ({col.outlier_percentage}%)</span></td>
                                    <td className="px-4 py-2">{col.unique_count}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        )}

        {/* Categorical Columns */}
        {quality.categorical_columns.length > 0 && (
            <div className="bg-white rounded-lg shadow">
                <div className="px-4 py-3 border-b border-gray-100">
                    <h3 className="font-medium text-gray-900">Categorical Columns ({quality.categorical_columns.length})</h3>
                </div>
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 text-sm">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Column</th>
                                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Nulls</th>
                                <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Unique Values</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {quality.categorical_columns.map(col => (
                                <tr key={col.name}>
                                    <td className="px-4 py-2 font-medium">{col.name}</td>
                                    <td className="px-4 py-2"><span className="px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-800 text-xs">{col.dtype}</span></td>
                                    <td className="px-4 py-2"><span className={col.null_count > 0 ? 'text-red-600 font-medium' : ''}>{col.null_count} ({col.null_percentage}%)</span></td>
                                    <td className="px-4 py-2">{col.unique_count}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        )}
    </div>
);

export default function DatasetProfilePage() {
    const params = useParams();
    const router = useRouter();
    const id = params.id as string;

    const [activeTab, setActiveTab] = useState<TabType>('overview');
    const [dataset, setDataset] = useState<Dataset | null>(null);
    const [profile, setProfile] = useState<DatasetProfile | null>(null);
    const [quality, setQuality] = useState<DataQualityReport | null>(null);
    const [loading, setLoading] = useState(true);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [preview, setPreview] = useState<EnhancedPreview | null>(null);
    const [previewMode, setPreviewMode] = useState<'original' | 'processed'>('processed');
    const [showProcessedOverview, setShowProcessedOverview] = useState(false);

    // Cleaning config state
    const [config, setConfig] = useState<DataCleaningConfig>({
        remove_duplicates: false,
        numeric_columns: {},
        categorical_columns: {}
    });

    useEffect(() => {
        if (!id) return;
        loadData();
    }, [id]);

    const loadData = async () => {
        setLoading(true);
        try {
            const [foundDataset, profileData, qualityData] = await Promise.all([
                getDatasets().then(list => list.find(d => d.id === id) || null),
                getDatasetProfile(id),
                getDataQuality(id)
            ]);

            setDataset(foundDataset);
            setProfile(profileData);
            setQuality(qualityData);

            // Initialize config from quality data
            const numericCols: Record<string, NumericColumnConfig> = {};
            const categoricalCols: Record<string, CategoricalColumnConfig> = {};

            qualityData.numeric_columns.forEach(col => {
                numericCols[col.name] = {
                    include: true,
                    dtype_override: "auto",
                    imputation_strategy: "none",
                    outlier_strategy: "none",
                    scaling_method: "none"
                };
            });

            qualityData.categorical_columns.forEach(col => {
                categoricalCols[col.name] = {
                    include: true,
                    dtype_override: "auto",
                    imputation_strategy: "none",
                    encoding_method: "none"
                };
            });

            setConfig({
                remove_duplicates: false,
                numeric_columns: numericCols,
                categorical_columns: categoricalCols
            });
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handlePreview = async () => {
        setPreviewLoading(true);
        try {
            const result = await getPreparationPreview(id, config);
            setPreview(result);
            setActiveTab('preview');
        } catch (e: any) {
            alert("Preview failed: " + (e.response?.data?.detail || e.message));
        } finally {
            setPreviewLoading(false);
        }
    };

    const handleSaveVersion = async () => {
        const name = prompt("Enter a name for this new version:", `${dataset?.name} - Processed`);
        if (!name) return;

        setSaving(true);
        try {
            const newDataset = await saveDatasetVersion(id, config, name);
            if (confirm("Version created successfully! Go to new version?")) {
                router.push(`/datasets/${newDataset.id}`);
            } else {
                // Refresh to show potentially updated state if we list versions here in future
            }
        } catch (e: any) {
            alert("Save failed: " + (e.response?.data?.detail || e.message));
        } finally {
            setSaving(false);
        }
    };

    if (loading) return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
            <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
                <p className="mt-4 text-gray-500">Loading dataset...</p>
            </div>
        </div>
    );

    const tabs = [
        { id: 'overview', label: 'Overview' },
        { id: 'cleaning', label: 'Data Cleaning' },
        { id: 'transformation', label: 'Transformation' },
        { id: 'preview', label: 'Preview & Apply' }
    ];

    return (
        <div className="min-h-screen bg-gray-50 font-sans">
            {/* Header */}
            <div className="bg-white shadow">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex justify-between items-center">
                        <div>
                            <h1 className="text-2xl font-bold text-gray-900">{dataset?.name || dataset?.filename}</h1>
                            <p className="text-sm text-gray-500 mt-1">Data Preparation</p>
                        </div>
                        <div className="flex items-center gap-3">
                            <Link href="/datasets" className="text-gray-600 hover:text-gray-900">← Back</Link>

                            <button
                                onClick={handleSaveVersion}
                                disabled={saving}
                                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                            >
                                {saving ? "Saving..." : "Save as Version"}
                            </button>
                            <Link
                                href={`/runs/new?dataset=${id}`}
                                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                            >
                                Start Training
                            </Link>
                        </div>
                    </div>

                    {/* Tabs */}
                    <div className="flex mt-4 border-b border-gray-200">
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id as TabType)}
                                className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${activeTab === tab.id
                                    ? 'border-blue-600 text-blue-600'
                                    : 'border-transparent text-gray-500 hover:text-gray-700'
                                    }`}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Tab Content */}
            <div className="max-w-7xl mx-auto px-6 py-6">
                {/* OVERVIEW TAB */}
                {activeTab === 'overview' && quality && (
                    <DataQualityOverview quality={quality} />
                )}

                {/* CLEANING TAB */}
                {activeTab === 'cleaning' && quality && (
                    <div className="space-y-6">
                        {/* Column Type Conversion */}
                        <div className="bg-white rounded-lg shadow">
                            <div className="px-4 py-3 border-b border-gray-100">
                                <h3 className="font-medium text-gray-900">Column Type Conversion</h3>
                                <p className="text-sm text-gray-500 mt-1">Change column data types before other transformations</p>
                            </div>
                            <div className="overflow-x-auto max-h-64 overflow-y-auto">
                                <table className="min-w-full divide-y divide-gray-200 text-sm">
                                    <thead className="bg-gray-50 sticky top-0">
                                        <tr>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Column</th>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Current Type</th>
                                            <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Convert To</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-200">
                                        {quality.numeric_columns.map(col => {
                                            const colConfig = config.numeric_columns[col.name] || { include: true, dtype_override: "auto", imputation_strategy: "none", outlier_strategy: "none", scaling_method: "none" };
                                            return (
                                                <tr key={col.name}>
                                                    <td className="px-4 py-2 font-medium">{col.name}</td>
                                                    <td className="px-4 py-2">
                                                        <span className="px-2 py-0.5 rounded-full bg-green-100 text-green-800 text-xs">{col.dtype}</span>
                                                    </td>
                                                    <td className="px-4 py-2">
                                                        <select
                                                            value={colConfig.dtype_override || "auto"}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                numeric_columns: { ...prev.numeric_columns, [col.name]: { ...colConfig, dtype_override: e.target.value as DataTypeOverride } }
                                                            }))}
                                                            className={`text-sm border-gray-300 rounded ${colConfig.dtype_override && colConfig.dtype_override !== "auto" ? 'bg-blue-50 border-blue-300' : ''}`}
                                                            disabled={!colConfig.include}
                                                        >
                                                            {DATA_TYPE_OPTIONS.map(t => <option key={t} value={t}>{t === "auto" ? "Keep Original" : t}</option>)}
                                                        </select>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                        {quality.categorical_columns.map(col => {
                                            const colConfig = config.categorical_columns[col.name] || { include: true, dtype_override: "auto", imputation_strategy: "none", encoding_method: "none" };
                                            return (
                                                <tr key={col.name}>
                                                    <td className="px-4 py-2 font-medium">{col.name}</td>
                                                    <td className="px-4 py-2">
                                                        <span className="px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-800 text-xs">{col.dtype}</span>
                                                    </td>
                                                    <td className="px-4 py-2">
                                                        <select
                                                            value={colConfig.dtype_override || "auto"}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                categorical_columns: { ...prev.categorical_columns, [col.name]: { ...colConfig, dtype_override: e.target.value as DataTypeOverride } }
                                                            }))}
                                                            className={`text-sm border-gray-300 rounded ${colConfig.dtype_override && colConfig.dtype_override !== "auto" ? 'bg-blue-50 border-blue-300' : ''}`}
                                                            disabled={!colConfig.include}
                                                        >
                                                            {DATA_TYPE_OPTIONS.map(t => <option key={t} value={t}>{t === "auto" ? "Keep Original" : t}</option>)}
                                                        </select>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Duplicates */}
                        <div className="bg-white rounded-lg shadow p-4">
                            <label className="flex items-center gap-3">
                                <input
                                    type="checkbox"
                                    checked={config.remove_duplicates}
                                    onChange={e => setConfig(prev => ({ ...prev, remove_duplicates: e.target.checked }))}
                                    className="h-4 w-4 text-blue-600 rounded"
                                />
                                <span className="font-medium">Remove {quality.duplicate_rows} duplicate rows ({quality.duplicate_percentage}%)</span>
                            </label>
                        </div>

                        {/* Numeric Null Handling */}
                        <div className="bg-white rounded-lg shadow">
                            <div className="px-4 py-3 border-b border-gray-100">
                                <h3 className="font-medium text-gray-900">Numeric Columns - Null Handling & Outliers</h3>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200 text-sm">
                                    <thead className="bg-gray-50">
                                        <tr>
                                            <th className="px-4 py-2 text-left">Include</th>
                                            <th className="px-4 py-2 text-left">Column</th>
                                            <th className="px-4 py-2 text-left">Nulls</th>
                                            <th className="px-4 py-2 text-left">Null Handling</th>
                                            <th className="px-4 py-2 text-left">Fill Value</th>
                                            <th className="px-4 py-2 text-left">Outliers</th>
                                            <th className="px-4 py-2 text-left">Outlier Handling</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-200">
                                        {quality.numeric_columns.map(col => {
                                            const colConfig = config.numeric_columns[col.name] || { include: true, imputation_strategy: "none", outlier_strategy: "none", scaling_method: "none" };
                                            return (
                                                <tr key={col.name} className={!colConfig.include ? 'bg-gray-50 opacity-60' : ''}>
                                                    <td className="px-4 py-2">
                                                        <input
                                                            type="checkbox"
                                                            checked={colConfig.include}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                numeric_columns: { ...prev.numeric_columns, [col.name]: { ...colConfig, include: e.target.checked } }
                                                            }))}
                                                            className="h-4 w-4 text-blue-600 rounded"
                                                        />
                                                    </td>
                                                    <td className="px-4 py-2 font-medium">{col.name}</td>
                                                    <td className="px-4 py-2"><span className={col.null_count > 0 ? 'text-red-600' : ''}>{col.null_count}</span></td>
                                                    <td className="px-4 py-2">
                                                        <select
                                                            value={colConfig.imputation_strategy}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                numeric_columns: { ...prev.numeric_columns, [col.name]: { ...colConfig, imputation_strategy: e.target.value as ImputationStrategy } }
                                                            }))}
                                                            className="text-sm border-gray-300 rounded"
                                                            disabled={!colConfig.include || col.null_count === 0}
                                                        >
                                                            {NUMERIC_IMPUTATION.map(s => <option key={s} value={s}>{s}</option>)}
                                                        </select>
                                                    </td>
                                                    <td className="px-4 py-2">
                                                        <input
                                                            type="number"
                                                            placeholder="0"
                                                            value={colConfig.constant_value ?? ''}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                numeric_columns: { ...prev.numeric_columns, [col.name]: { ...colConfig, constant_value: e.target.value ? parseFloat(e.target.value) : undefined } }
                                                            }))}
                                                            className="w-20 text-sm border-gray-300 rounded px-2 py-1"
                                                            disabled={!colConfig.include || colConfig.imputation_strategy !== 'constant'}
                                                        />
                                                    </td>
                                                    <td className="px-4 py-2"><span className={col.outlier_count > 0 ? 'text-orange-600' : ''}>{col.outlier_count}</span></td>
                                                    <td className="px-4 py-2">
                                                        <select
                                                            value={colConfig.outlier_strategy}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                numeric_columns: { ...prev.numeric_columns, [col.name]: { ...colConfig, outlier_strategy: e.target.value as OutlierStrategy } }
                                                            }))}
                                                            className="text-sm border-gray-300 rounded"
                                                            disabled={!colConfig.include || col.outlier_count === 0}
                                                        >
                                                            {OUTLIER_OPTIONS.map(s => <option key={s} value={s}>{s}</option>)}
                                                        </select>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Categorical Null Handling */}
                        <div className="bg-white rounded-lg shadow">
                            <div className="px-4 py-3 border-b border-gray-100">
                                <h3 className="font-medium text-gray-900">Categorical Columns - Null Handling</h3>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200 text-sm">
                                    <thead className="bg-gray-50">
                                        <tr>
                                            <th className="px-4 py-2 text-left">Include</th>
                                            <th className="px-4 py-2 text-left">Column</th>
                                            <th className="px-4 py-2 text-left">Nulls</th>
                                            <th className="px-4 py-2 text-left">Null Handling</th>
                                            <th className="px-4 py-2 text-left">Fill Value</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-200">
                                        {quality.categorical_columns.map(col => {
                                            const colConfig = config.categorical_columns[col.name] || { include: true, imputation_strategy: "none", encoding_method: "none" };
                                            return (
                                                <tr key={col.name} className={!colConfig.include ? 'bg-gray-50 opacity-60' : ''}>
                                                    <td className="px-4 py-2">
                                                        <input
                                                            type="checkbox"
                                                            checked={colConfig.include}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                categorical_columns: { ...prev.categorical_columns, [col.name]: { ...colConfig, include: e.target.checked } }
                                                            }))}
                                                            className="h-4 w-4 text-blue-600 rounded"
                                                        />
                                                    </td>
                                                    <td className="px-4 py-2 font-medium">{col.name}</td>
                                                    <td className="px-4 py-2"><span className={col.null_count > 0 ? 'text-red-600' : ''}>{col.null_count}</span></td>
                                                    <td className="px-4 py-2">
                                                        <select
                                                            value={colConfig.imputation_strategy}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                categorical_columns: { ...prev.categorical_columns, [col.name]: { ...colConfig, imputation_strategy: e.target.value as ImputationStrategy } }
                                                            }))}
                                                            className="text-sm border-gray-300 rounded"
                                                            disabled={!colConfig.include || col.null_count === 0}
                                                        >
                                                            {CATEGORICAL_IMPUTATION.map(s => <option key={s} value={s}>{s}</option>)}
                                                        </select>
                                                    </td>
                                                    <td className="px-4 py-2">
                                                        <input
                                                            type="text"
                                                            placeholder="Unknown"
                                                            value={colConfig.constant_value ?? ''}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                categorical_columns: { ...prev.categorical_columns, [col.name]: { ...colConfig, constant_value: e.target.value || undefined } }
                                                            }))}
                                                            className="w-24 text-sm border-gray-300 rounded px-2 py-1"
                                                            disabled={!colConfig.include || colConfig.imputation_strategy !== 'constant'}
                                                        />
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}

                {/* TRANSFORMATION TAB */}
                {activeTab === 'transformation' && quality && (
                    <div className="space-y-6">
                        {/* Numeric Scaling */}
                        <div className="bg-white rounded-lg shadow">
                            <div className="px-4 py-3 border-b border-gray-100">
                                <h3 className="font-medium text-gray-900">Numeric Columns - Scaling</h3>
                                <p className="text-sm text-gray-500 mt-1">Apply normalization or standardization</p>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200 text-sm">
                                    <thead className="bg-gray-50">
                                        <tr>
                                            <th className="px-4 py-2 text-left">Column</th>
                                            <th className="px-4 py-2 text-left">Scaling Method</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-200">
                                        {quality.numeric_columns.map(col => {
                                            const colConfig = config.numeric_columns[col.name];
                                            if (!colConfig?.include) return null;
                                            return (
                                                <tr key={col.name}>
                                                    <td className="px-4 py-2 font-medium">{col.name}</td>
                                                    <td className="px-4 py-2">
                                                        <select
                                                            value={colConfig.scaling_method}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                numeric_columns: { ...prev.numeric_columns, [col.name]: { ...colConfig, scaling_method: e.target.value as ScalingMethod } }
                                                            }))}
                                                            className="text-sm border-gray-300 rounded"
                                                        >
                                                            {SCALING_OPTIONS.map(s => <option key={s} value={s}>{s === 'standard' ? 'Standard (z-score)' : s === 'minmax' ? 'MinMax (0-1)' : s === 'robust' ? 'Robust (IQR)' : s}</option>)}
                                                        </select>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Categorical Encoding */}
                        <div className="bg-white rounded-lg shadow">
                            <div className="px-4 py-3 border-b border-gray-100">
                                <h3 className="font-medium text-gray-900">Categorical Columns - Encoding</h3>
                                <p className="text-sm text-gray-500 mt-1">Convert categories to numeric values</p>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200 text-sm">
                                    <thead className="bg-gray-50">
                                        <tr>
                                            <th className="px-4 py-2 text-left">Column</th>
                                            <th className="px-4 py-2 text-left">Unique</th>
                                            <th className="px-4 py-2 text-left">Encoding Method</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-200">
                                        {quality.categorical_columns.map(col => {
                                            const colConfig = config.categorical_columns[col.name];
                                            if (!colConfig?.include) return null;
                                            return (
                                                <tr key={col.name}>
                                                    <td className="px-4 py-2 font-medium">{col.name}</td>
                                                    <td className="px-4 py-2">{col.unique_count}</td>
                                                    <td className="px-4 py-2">
                                                        <select
                                                            value={colConfig.encoding_method}
                                                            onChange={e => setConfig(prev => ({
                                                                ...prev,
                                                                categorical_columns: { ...prev.categorical_columns, [col.name]: { ...colConfig, encoding_method: e.target.value as EncodingMethod } }
                                                            }))}
                                                            className="text-sm border-gray-300 rounded"
                                                        >
                                                            {ENCODING_OPTIONS.map(s => <option key={s} value={s}>{s === 'onehot' ? 'One-Hot' : s === 'label' ? 'Label (0,1,2...)' : s === 'frequency' ? 'Frequency' : s}</option>)}
                                                        </select>
                                                    </td>
                                                </tr>
                                            );
                                        })}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}

                {/* PREVIEW TAB */}
                {activeTab === 'preview' && (
                    <div className="space-y-6">
                        {preview ? (
                            <>
                                {/* Summary */}
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <h4 className="font-medium text-blue-900 mb-1">Processed Data Stats</h4>
                                                <div className="text-sm text-blue-700 space-y-1">
                                                    <p>{preview.row_count.toLocaleString()} rows (from {preview.original_row_count.toLocaleString()})</p>
                                                    <p>{preview.columns.length} columns</p>
                                                    <p className="text-xs opacity-75">
                                                        {preview.quality_report?.numeric_columns.length || 0} numeric, {preview.quality_report?.categorical_columns.length || 0} categorical
                                                    </p>
                                                </div>
                                            </div>
                                            <button onClick={handlePreview} className="text-blue-600 hover:text-blue-800 text-sm bg-white px-3 py-1 rounded shadow-sm">
                                                Refresh
                                            </button>
                                        </div>
                                    </div>

                                    {preview.quality_report && (
                                        <div className="bg-green-50 border border-green-200 rounded-lg p-4 relative">
                                            <div className="flex justify-between items-start">
                                                <h4 className="font-medium text-green-900 mb-1">Quality Overview</h4>
                                                <button
                                                    onClick={() => setShowProcessedOverview(true)}
                                                    className="text-xs bg-green-100 hover:bg-green-200 text-green-800 px-2 py-1 rounded border border-green-300 transition-colors"
                                                >
                                                    View Details ↗
                                                </button>
                                            </div>
                                            <div className="grid grid-cols-2 gap-4 text-sm mt-2">
                                                <div>
                                                    <span className="block text-green-700 text-xs">Total Nulls</span>
                                                    <span className={`font-semibold ${(preview.quality_report.numeric_columns.reduce((a, b) => a + b.null_count, 0) +
                                                        preview.quality_report.categorical_columns.reduce((a, b) => a + b.null_count, 0)) > 0
                                                        ? 'text-red-500' : 'text-green-700'
                                                        }`}>
                                                        {preview.quality_report.numeric_columns.reduce((a, b) => a + b.null_count, 0) +
                                                            preview.quality_report.categorical_columns.reduce((a, b) => a + b.null_count, 0)}
                                                    </span>
                                                </div>
                                                <div>
                                                    <span className="block text-green-700 text-xs">Remaining Outliers</span>
                                                    <span className={`font-semibold ${preview.quality_report.numeric_columns.reduce((a, b) => a + b.outlier_count, 0) > 0
                                                        ? 'text-orange-500' : 'text-green-700'
                                                        }`}>
                                                        {preview.quality_report.numeric_columns.reduce((a, b) => a + b.outlier_count, 0)}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Transformations */}
                                <div className="bg-white rounded-lg shadow">
                                    <div className="px-4 py-3 border-b border-gray-100 flex justify-between items-center">
                                        <h3 className="font-medium text-gray-900">Applied Transformations ({preview.applied_transformations.length})</h3>
                                    </div>
                                    <div className="p-4 max-h-48 overflow-y-auto">
                                        {preview.applied_transformations.length > 0 ? (
                                            <ul className="space-y-1 text-sm">
                                                {preview.applied_transformations.map((t, i) => (
                                                    <li key={i} className="flex items-start gap-2">
                                                        <span className="text-green-500">✓</span>
                                                        <span>{t}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        ) : (
                                            <p className="text-gray-500">No transformations applied</p>
                                        )}
                                    </div>
                                </div>

                                {/* Data Table */}
                                <div className="bg-white rounded-lg shadow overflow-hidden">
                                    <div className="px-4 py-3 border-b border-gray-100 flex justify-between items-center">
                                        <h3 className="font-medium text-gray-900">Data Preview (First 5 Rows)</h3>
                                        <div className="flex bg-gray-100 rounded-lg p-1">
                                            <button
                                                onClick={() => setPreviewMode('original')}
                                                className={`px-3 py-1 text-xs font-medium rounded-md ${previewMode === 'original' ? 'bg-white shadow text-gray-900' : 'text-gray-500 hover:text-gray-700'}`}
                                            >
                                                Original
                                            </button>
                                            <button
                                                onClick={() => setPreviewMode('processed')}
                                                className={`px-3 py-1 text-xs font-medium rounded-md ${previewMode === 'processed' ? 'bg-white shadow text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
                                            >
                                                Processed
                                            </button>
                                        </div>
                                    </div>
                                    <div className="overflow-x-auto">
                                        <table className="min-w-full divide-y divide-gray-200 text-xs">
                                            <thead className="bg-gray-50">
                                                <tr>
                                                    {(previewMode === 'original' && preview.original_data && preview.original_data.length > 0
                                                        ? Object.keys(preview.original_data[0])
                                                        : preview.columns
                                                    ).map(col => (
                                                        <th key={col} className="px-3 py-2 text-left font-medium text-gray-500 whitespace-nowrap">{col}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-gray-200">
                                                {(previewMode === 'original' && preview.original_data ? preview.original_data : preview.data).map((row, i) => (
                                                    <tr key={i}>
                                                        {(previewMode === 'original' && preview.original_data && preview.original_data.length > 0
                                                            ? Object.keys(preview.original_data[0])
                                                            : preview.columns
                                                        ).map(col => {
                                                            const value = row[col];
                                                            let displayValue;

                                                            if (value === null || value === undefined) {
                                                                displayValue = <span className="text-gray-400">null</span>;
                                                            } else if (typeof value === 'boolean') {
                                                                displayValue = value ? '1' : '0';
                                                            } else if (typeof value === 'number') {
                                                                displayValue = Number.isInteger(value) ? value.toString() : value.toFixed(3);
                                                            } else {
                                                                displayValue = String(value);
                                                            }

                                                            return (
                                                                <td key={col} className="px-3 py-2 whitespace-nowrap">
                                                                    {displayValue}
                                                                </td>
                                                            );
                                                        })}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </>
                        ) : (
                            <div className="bg-white rounded-lg shadow p-12 text-center">
                                <p className="text-gray-500 mb-4">Configure your data cleaning and transformation settings, then click Preview to see the results.</p>
                                <button onClick={handlePreview} disabled={previewLoading} className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50">
                                    {previewLoading ? "Loading..." : "Generate Preview"}
                                </button>
                            </div>
                        )}
                    </div>
                )}
            </div>
            {/* PROCESSED DATA OVERVIEW MODAL */}
            {showProcessedOverview && preview?.quality_report && (
                <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
                    <div className="bg-white rounded-xl shadow-2xl w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
                        <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center bg-gray-50">
                            <div>
                                <h2 className="text-xl font-bold text-gray-900">Processed Data Overview</h2>
                                <p className="text-sm text-gray-500">Detailed quality report for the transformed dataset</p>
                            </div>
                            <button
                                onClick={() => setShowProcessedOverview(false)}
                                className="text-gray-400 hover:text-gray-600 p-2 rounded-full hover:bg-gray-100"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>
                        <div className="p-6 overflow-y-auto">
                            <DataQualityOverview quality={preview.quality_report} />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
