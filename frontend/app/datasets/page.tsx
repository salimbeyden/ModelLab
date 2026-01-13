"use client";

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation'; // Added useRouter
import { getDatasets, uploadDataset, deleteDataset, Dataset } from '@/app/lib/api';

// Helper to organize datasets into hierarchy
// Maps parent_id (or "root") to list of datasets
const groupDatasets = (datasets: Dataset[]) => {
    const groups: { [key: string]: Dataset[] } = { root: [] };
    const children: { [key: string]: Dataset[] } = {};
    const datasetIds = new Set(datasets.map(d => d.id));

    datasets.forEach(d => {
        // Only treat as child if parent exists in the list
        if (d.parent_id && datasetIds.has(d.parent_id)) {
            if (!children[d.parent_id]) children[d.parent_id] = [];
            children[d.parent_id].push(d);
        } else {
            groups.root.push(d);
        }
    });

    return { roots: groups.root, children };
};

export default function DatasetsPage() {
    const router = useRouter();
    const [datasetName, setDatasetName] = useState("");
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [uploading, setUploading] = useState(false);
    const [datasets, setDatasets] = useState<Dataset[]>([]);

    // Expand state for accordion
    const [expanded, setExpanded] = useState<Record<string, boolean>>({});

    const toggleExpand = (id: string) => {
        setExpanded(prev => ({ ...prev, [id]: !prev[id] }));
    };

    const loadDatasets = async () => {
        const data = await getDatasets();
        // Sort by date desc
        setDatasets(data.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()));
    };

    useEffect(() => {
        loadDatasets();
    }, []);

    const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files?.length) return;
        setUploading(true);
        try {
            await uploadDataset(e.target.files[0], datasetName || undefined);
            setDatasetName(""); // Reset name
            loadDatasets();
        } catch (err) {
            console.error(err);
            alert("Upload failed");
        } finally {
            setUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = "";
        }
    };

    const handleDelete = async (e: React.MouseEvent, id: string) => {
        e.preventDefault(); // Stop navigation
        e.stopPropagation();
        if (!confirm("Are you sure?")) return;
        try {
            await deleteDataset(id);
            loadDatasets();
        } catch (err) {
            alert("Delete failed");
        }
    }

    const { roots, children } = groupDatasets(datasets);

    return (
        <div className="min-h-screen bg-gray-50 p-8">
            <div className="max-w-6xl mx-auto">
                <div className="flex justify-between items-center mb-8">
                    <h1 className="text-3xl font-bold text-gray-900">Datasets</h1>
                    <div className="flex bg-white p-2 rounded-lg shadow-sm gap-2">
                        <input
                            type="text"
                            placeholder="Dataset Name (Optional)"
                            className="border rounded px-3 py-1 text-sm w-48"
                            value={datasetName}
                            onChange={(e) => setDatasetName(e.target.value)}
                        />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            disabled={uploading}
                            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
                        >
                            {uploading ? (
                                <>
                                    <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Uploading...
                                </>
                            ) : (
                                <>
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                                    </svg>
                                    Upload New Dataset
                                </>
                            )}
                        </button>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".csv,.parquet"
                            className="hidden"
                            onChange={handleUpload}
                        />
                    </div>
                </div>

                <div className="space-y-4">
                    {roots.map(dataset => {
                        const versions = children[dataset.id] || [];
                        const isExpanded = expanded[dataset.id];

                        return (
                            <div key={dataset.id} className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
                                <div className="p-6 flex items-center justify-between hover:bg-gray-50 transition-colors">
                                    <div className="flex items-center gap-4 flex-1">
                                        <div className="p-3 bg-blue-100 text-blue-600 rounded-lg">
                                            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                                            </svg>
                                        </div>
                                        <div
                                            onClick={() => router.push(`/datasets/${dataset.id}`)}
                                            className="cursor-pointer"
                                        >
                                            <h3 className="text-lg font-medium text-gray-900">{dataset.name || dataset.filename}</h3>
                                            <div className="flex gap-4 text-sm text-gray-500 mt-1">
                                                <span>{dataset.row_count.toLocaleString()} rows</span>
                                                <span>•</span>
                                                <span>{dataset.columns.length} columns</span>
                                                <span>•</span>
                                                <span>{new Date(dataset.created_at).toLocaleDateString()}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-4">
                                        {/* Version Toggle */}
                                        {versions.length > 0 && (
                                            <button
                                                onClick={() => toggleExpand(dataset.id)}
                                                className="flex items-center gap-1 text-sm text-gray-600 hover:text-blue-600 px-3 py-1 rounded-md hover:bg-gray-100"
                                            >
                                                {versions.length} Versions
                                                <svg
                                                    className={`w-4 h-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                                                    fill="none"
                                                    stroke="currentColor"
                                                    viewBox="0 0 24 24"
                                                >
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                                                </svg>
                                            </button>
                                        )}

                                        <Link
                                            href={`/datasets/${dataset.id}`}
                                            className="text-gray-400 hover:text-blue-600 transition-colors"
                                        >
                                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                            </svg>
                                        </Link>
                                        <button
                                            onClick={(e) => handleDelete(e, dataset.id)}
                                            className="text-gray-400 hover:text-red-600 transition-colors"
                                        >
                                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                            </svg>
                                        </button>
                                    </div>
                                </div>

                                {/* Versions List */}
                                {isExpanded && versions.length > 0 && (
                                    <div className="bg-gray-50 border-t border-gray-100">
                                        {versions.map(v => (
                                            <div key={v.id} className="pl-20 pr-6 py-3 flex items-center justify-between border-b border-gray-100 last:border-0 hover:bg-gray-100">
                                                <div className="flex items-center gap-3">
                                                    <span className="bg-green-100 text-green-700 text-xs px-2 py-0.5 rounded-full font-medium">Version</span>
                                                    <div
                                                        onClick={() => router.push(`/datasets/${v.id}`)}
                                                        className="cursor-pointer hover:text-blue-600"
                                                    >
                                                        <span className="font-medium text-gray-700">{v.name}</span>
                                                        <span className="text-gray-400 text-sm ml-2">({v.row_count} rows, {new Date(v.created_at).toLocaleDateString()})</span>
                                                    </div>
                                                </div>
                                                <div className="flex items-center gap-3">
                                                    <Link
                                                        href={`/datasets/${v.id}`}
                                                        className="text-sm text-blue-600 hover:text-blue-800"
                                                    >
                                                        Review & Train
                                                    </Link>
                                                    <button
                                                        onClick={(e) => handleDelete(e, v.id)}
                                                        className="text-gray-400 hover:text-red-600 transition-colors"
                                                    >
                                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                                        </svg>
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            </div>
        </div>
    );
}
