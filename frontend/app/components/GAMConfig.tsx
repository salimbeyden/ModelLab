"use client";

import React, { useState, useEffect, useMemo } from 'react';

// Types
interface SmoothTerm {
    variable: string;
    type: 's' | 'te' | 'ti' | 'linear';
    bs: 'tp' | 'cr' | 'cc' | 'ps' | 'cp' | 'ds' | 're' | 'mrf' | 'gp';
    k: number;
    by: string;
}

interface TensorTerm {
    variables: string[];
    type: 'te' | 'ti';
    k: number[];
}

interface GAMConfigProps {
    columns: string[];
    numericColumns: string[];
    categoricalColumns: string[];
    task: 'regression' | 'classification';
    onConfigChange: (config: GAMParams) => void;
    initialConfig?: GAMParams;
}

export interface GAMParams {
    formula_mode: 'auto' | 'manual';
    formula: string;
    method: 'REML' | 'GCV.Cp' | 'ML' | 'P-REML' | 'P-ML';
    family: string;
    link: string;
    select: boolean;
    gamma: number;
    smooth_terms: SmoothTerm[];
    tensor_terms: TensorTerm[];
}

// Family-Link compatibility
const FAMILY_LINKS: Record<string, string[]> = {
    gaussian: ['identity', 'log', 'inverse'],
    binomial: ['logit', 'probit', 'cloglog', 'cauchit'],
    poisson: ['log', 'identity', 'sqrt'],
    Gamma: ['inverse', 'identity', 'log'],
    nb: ['log', 'identity', 'sqrt'],
    'inverse.gaussian': ['1/mu^2', 'inverse', 'identity', 'log'],
    quasi: ['identity', 'log', 'logit'],
    quasibinomial: ['logit', 'probit', 'cloglog'],
    quasipoisson: ['log', 'identity', 'sqrt'],
};

const DEFAULT_LINKS: Record<string, string> = {
    gaussian: 'identity',
    binomial: 'logit',
    poisson: 'log',
    Gamma: 'inverse',
    nb: 'log',
    'inverse.gaussian': '1/mu^2',
    quasi: 'identity',
    quasibinomial: 'logit',
    quasipoisson: 'log',
};

const BASIS_DESCRIPTIONS: Record<string, string> = {
    tp: 'Thin Plate Spline (default, general purpose)',
    cr: 'Cubic Regression Spline (good for smooth data)',
    cc: 'Cyclic Cubic Spline (for periodic data like time of day)',
    ps: 'P-Spline (penalized B-spline)',
    cp: 'Cyclic P-Spline',
    ds: 'Duchon Spline',
    re: 'Random Effect (for hierarchical/grouped data)',
    mrf: 'Markov Random Field (for spatial data)',
    gp: 'Gaussian Process (flexible, computationally intensive)',
};

export default function GAMConfig({
    columns,
    numericColumns,
    categoricalColumns,
    task,
    onConfigChange,
    initialConfig
}: GAMConfigProps) {
    const [formulaMode, setFormulaMode] = useState<'auto' | 'manual'>(initialConfig?.formula_mode || 'auto');
    const [manualFormula, setManualFormula] = useState(initialConfig?.formula || '');
    const [method, setMethod] = useState<string>(initialConfig?.method || 'REML');
    const [family, setFamily] = useState(initialConfig?.family || (task === 'classification' ? 'binomial' : 'gaussian'));
    const [link, setLink] = useState(initialConfig?.link || '');
    const [select, setSelect] = useState(initialConfig?.select || false);
    const [gamma, setGamma] = useState(initialConfig?.gamma || 1.0);
    const [smoothTerms, setSmoothTerms] = useState<SmoothTerm[]>(initialConfig?.smooth_terms || []);
    const [tensorTerms, setTensorTerms] = useState<TensorTerm[]>(initialConfig?.tensor_terms || []);
    const [activeTab, setActiveTab] = useState<'basic' | 'smooth' | 'tensor'>('basic');

    // Update family based on task
    useEffect(() => {
        if (task === 'classification') {
            setFamily('binomial');
            setLink('logit');
        }
    }, [task]);

    // Available link functions based on selected family
    const availableLinks = useMemo(() => FAMILY_LINKS[family] || ['identity'], [family]);

    // Build the config object
    const config = useMemo((): GAMParams => ({
        formula_mode: formulaMode,
        formula: manualFormula,
        method: method as GAMParams['method'],
        family,
        link: link || DEFAULT_LINKS[family] || 'identity',
        select,
        gamma,
        smooth_terms: smoothTerms,
        tensor_terms: tensorTerms,
    }), [formulaMode, manualFormula, method, family, link, select, gamma, smoothTerms, tensorTerms]);

    // Notify parent of config changes
    useEffect(() => {
        onConfigChange(config);
    }, [config, onConfigChange]);

    // Preview formula
    const previewFormula = useMemo(() => {
        if (formulaMode === 'manual') return manualFormula;
        
        const terms: string[] = [];
        
        // Build from smooth terms or defaults
        const configuredVars = new Set(smoothTerms.map(t => t.variable));
        
        for (const col of numericColumns) {
            const smoothTerm = smoothTerms.find(t => t.variable === col);
            if (smoothTerm) {
                if (smoothTerm.type === 'linear') {
                    terms.push(col.includes(' ') ? `\`${col}\`` : col);
                } else {
                    const parts = [col.includes(' ') ? `\`${col}\`` : col];
                    if (smoothTerm.bs !== 'tp') parts.push(`bs="${smoothTerm.bs}"`);
                    if (smoothTerm.k > 0) parts.push(`k=${smoothTerm.k}`);
                    if (smoothTerm.by) parts.push(`by=${smoothTerm.by}`);
                    terms.push(`${smoothTerm.type}(${parts.join(', ')})`);
                }
            } else {
                // Default: smooth term
                terms.push(`s(${col.includes(' ') ? `\`${col}\`` : col})`);
            }
        }
        
        // Add categorical columns as linear terms
        for (const col of categoricalColumns) {
            terms.push(col.includes(' ') ? `\`${col}\`` : col);
        }
        
        // Add tensor terms
        for (const tensor of tensorTerms) {
            if (tensor.variables.length >= 2) {
                const vars = tensor.variables.map(v => v.includes(' ') ? `\`${v}\`` : v);
                const kStr = tensor.k.length ? `, k=c(${tensor.k.join(',')})` : '';
                terms.push(`${tensor.type}(${vars.join(', ')}${kStr})`);
            }
        }
        
        return `target ~ ${terms.join(' + ')}`;
    }, [formulaMode, manualFormula, numericColumns, categoricalColumns, smoothTerms, tensorTerms]);

    // Add smooth term
    const addSmoothTerm = (variable: string) => {
        if (!smoothTerms.find(t => t.variable === variable)) {
            setSmoothTerms([...smoothTerms, {
                variable,
                type: 's',
                bs: 'tp',
                k: -1,
                by: ''
            }]);
        }
    };

    // Update smooth term
    const updateSmoothTerm = (index: number, updates: Partial<SmoothTerm>) => {
        const newTerms = [...smoothTerms];
        newTerms[index] = { ...newTerms[index], ...updates };
        setSmoothTerms(newTerms);
    };

    // Remove smooth term
    const removeSmoothTerm = (index: number) => {
        setSmoothTerms(smoothTerms.filter((_, i) => i !== index));
    };

    // Add tensor term
    const addTensorTerm = () => {
        setTensorTerms([...tensorTerms, {
            variables: [],
            type: 'te',
            k: []
        }]);
    };

    // Update tensor term
    const updateTensorTerm = (index: number, updates: Partial<TensorTerm>) => {
        const newTerms = [...tensorTerms];
        newTerms[index] = { ...newTerms[index], ...updates };
        setTensorTerms(newTerms);
    };

    // Remove tensor term
    const removeTensorTerm = (index: number) => {
        setTensorTerms(tensorTerms.filter((_, i) => i !== index));
    };

    return (
        <div className="space-y-6">
            {/* Formula Preview */}
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-x-auto">
                <div className="text-gray-500 text-xs mb-1">Formula Preview:</div>
                <code>{previewFormula}</code>
            </div>

            {/* Tab Navigation */}
            <div className="border-b border-gray-200">
                <nav className="flex space-x-4">
                    {[
                        { id: 'basic', label: 'Basic Settings' },
                        { id: 'smooth', label: 'Smooth Terms' },
                        { id: 'tensor', label: 'Interactions' },
                    ].map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id as typeof activeTab)}
                            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors ${
                                activeTab === tab.id
                                    ? 'border-blue-500 text-blue-600'
                                    : 'border-transparent text-gray-500 hover:text-gray-700'
                            }`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </nav>
            </div>

            {/* Basic Settings Tab */}
            {activeTab === 'basic' && (
                <div className="space-y-6">
                    {/* Formula Mode */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Formula Mode
                        </label>
                        <div className="flex gap-4">
                            <label className="inline-flex items-center">
                                <input
                                    type="radio"
                                    value="auto"
                                    checked={formulaMode === 'auto'}
                                    onChange={() => setFormulaMode('auto')}
                                    className="form-radio text-blue-600"
                                />
                                <span className="ml-2 text-sm text-gray-700">
                                    Auto-generate (recommended)
                                </span>
                            </label>
                            <label className="inline-flex items-center">
                                <input
                                    type="radio"
                                    value="manual"
                                    checked={formulaMode === 'manual'}
                                    onChange={() => setFormulaMode('manual')}
                                    className="form-radio text-blue-600"
                                />
                                <span className="ml-2 text-sm text-gray-700">
                                    Manual formula
                                </span>
                            </label>
                        </div>
                    </div>

                    {/* Manual Formula Input */}
                    {formulaMode === 'manual' && (
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                R Formula
                            </label>
                            <textarea
                                value={manualFormula}
                                onChange={(e) => setManualFormula(e.target.value)}
                                placeholder="e.g., target ~ s(x1, bs='cr', k=10) + s(x2) + x3"
                                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm font-mono focus:ring-blue-500 focus:border-blue-500"
                                rows={3}
                            />
                            <p className="mt-1 text-xs text-gray-500">
                                Use s() for smooth terms, te() for tensor products, ti() for tensor interactions
                            </p>
                        </div>
                    )}

                    {/* Method */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Smoothing Method
                            </label>
                            <select
                                value={method}
                                onChange={(e) => setMethod(e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
                            >
                                <option value="REML">REML (Restricted Maximum Likelihood)</option>
                                <option value="GCV.Cp">GCV.Cp (Generalized Cross-Validation)</option>
                                <option value="ML">ML (Maximum Likelihood)</option>
                                <option value="P-REML">P-REML (Performance REML)</option>
                                <option value="P-ML">P-ML (Performance ML)</option>
                            </select>
                            <p className="mt-1 text-xs text-gray-500">
                                REML is most robust; GCV.Cp may work better for larger datasets
                            </p>
                        </div>

                        {/* Family (only for regression) */}
                        {task === 'regression' && (
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">
                                    Distribution Family
                                </label>
                                <select
                                    value={family}
                                    onChange={(e) => {
                                        setFamily(e.target.value);
                                        setLink(''); // Reset link when family changes
                                    }}
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
                                >
                                    <option value="gaussian">Gaussian (Normal - continuous data)</option>
                                    <option value="poisson">Poisson (Count data)</option>
                                    <option value="Gamma">Gamma (Positive continuous)</option>
                                    <option value="nb">Negative Binomial (Overdispersed counts)</option>
                                    <option value="inverse.gaussian">Inverse Gaussian</option>
                                    <option value="quasi">Quasi (custom variance)</option>
                                    <option value="quasipoisson">Quasi-Poisson</option>
                                </select>
                            </div>
                        )}
                    </div>

                    {/* Link Function */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Link Function
                            </label>
                            <select
                                value={link || DEFAULT_LINKS[family]}
                                onChange={(e) => setLink(e.target.value)}
                                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-blue-500 focus:border-blue-500"
                            >
                                {availableLinks.map(l => (
                                    <option key={l} value={l}>
                                        {l} {l === DEFAULT_LINKS[family] ? '(default)' : ''}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Gamma (smoothness) */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Smoothness Multiplier (γ): {gamma.toFixed(1)}
                            </label>
                            <input
                                type="range"
                                min="0.1"
                                max="5"
                                step="0.1"
                                value={gamma}
                                onChange={(e) => setGamma(parseFloat(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                            />
                            <p className="mt-1 text-xs text-gray-500">
                                Higher values = smoother curves (less overfitting). Default is 1.0
                            </p>
                        </div>
                    </div>

                    {/* Variable Selection */}
                    <div>
                        <label className="inline-flex items-center">
                            <input
                                type="checkbox"
                                checked={select}
                                onChange={(e) => setSelect(e.target.checked)}
                                className="form-checkbox text-blue-600 rounded"
                            />
                            <span className="ml-2 text-sm text-gray-700">
                                Enable automatic variable selection
                            </span>
                        </label>
                        <p className="mt-1 text-xs text-gray-500 ml-6">
                            Adds an extra penalty to each smooth term, allowing unimportant terms to be shrunk to zero
                        </p>
                    </div>
                </div>
            )}

            {/* Smooth Terms Tab */}
            {activeTab === 'smooth' && formulaMode === 'auto' && (
                <div className="space-y-4">
                    <p className="text-sm text-gray-600">
                        Configure individual smooth terms for each feature. By default, numeric features use thin plate splines.
                    </p>

                    {/* Quick Add */}
                    <div className="flex flex-wrap gap-2 p-3 bg-gray-50 rounded-lg">
                        <span className="text-sm text-gray-600 mr-2">Add term for:</span>
                        {numericColumns.filter(col => !smoothTerms.find(t => t.variable === col)).map(col => (
                            <button
                                key={col}
                                onClick={() => addSmoothTerm(col)}
                                className="px-2 py-1 text-xs bg-white border border-gray-300 rounded hover:bg-gray-100 transition-colors"
                            >
                                + {col}
                            </button>
                        ))}
                        {numericColumns.filter(col => !smoothTerms.find(t => t.variable === col)).length === 0 && (
                            <span className="text-xs text-gray-400 italic">All numeric columns configured</span>
                        )}
                    </div>

                    {/* Configured Terms */}
                    {smoothTerms.length > 0 ? (
                        <div className="space-y-3">
                            {smoothTerms.map((term, idx) => (
                                <div key={idx} className="p-4 bg-white border border-gray-200 rounded-lg">
                                    <div className="flex items-center justify-between mb-3">
                                        <h4 className="font-medium text-gray-900">{term.variable}</h4>
                                        <button
                                            onClick={() => removeSmoothTerm(idx)}
                                            className="text-red-500 hover:text-red-700 text-sm"
                                        >
                                            Remove
                                        </button>
                                    </div>
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                        <div>
                                            <label className="block text-xs text-gray-500 mb-1">Term Type</label>
                                            <select
                                                value={term.type}
                                                onChange={(e) => updateSmoothTerm(idx, { type: e.target.value as SmoothTerm['type'] })}
                                                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                                            >
                                                <option value="s">s() - Smooth</option>
                                                <option value="linear">Linear (no smooth)</option>
                                            </select>
                                        </div>
                                        {term.type !== 'linear' && (
                                            <>
                                                <div>
                                                    <label className="block text-xs text-gray-500 mb-1">Basis Type</label>
                                                    <select
                                                        value={term.bs}
                                                        onChange={(e) => updateSmoothTerm(idx, { bs: e.target.value as SmoothTerm['bs'] })}
                                                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                                                    >
                                                        {Object.entries(BASIS_DESCRIPTIONS).map(([key, desc]) => (
                                                            <option key={key} value={key} title={desc}>
                                                                {key} - {desc.split('(')[0].trim()}
                                                            </option>
                                                        ))}
                                                    </select>
                                                </div>
                                                <div>
                                                    <label className="block text-xs text-gray-500 mb-1">
                                                        Knots (k) {term.k === -1 ? '(auto)' : ''}
                                                    </label>
                                                    <input
                                                        type="number"
                                                        value={term.k === -1 ? '' : term.k}
                                                        onChange={(e) => updateSmoothTerm(idx, { k: e.target.value ? parseInt(e.target.value) : -1 })}
                                                        placeholder="Auto"
                                                        min={3}
                                                        max={100}
                                                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                                                    />
                                                </div>
                                                <div>
                                                    <label className="block text-xs text-gray-500 mb-1">By Variable</label>
                                                    <select
                                                        value={term.by}
                                                        onChange={(e) => updateSmoothTerm(idx, { by: e.target.value })}
                                                        className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                                                    >
                                                        <option value="">None</option>
                                                        {categoricalColumns.map(col => (
                                                            <option key={col} value={col}>{col}</option>
                                                        ))}
                                                    </select>
                                                </div>
                                            </>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-8 text-gray-400 italic">
                            No custom smooth terms configured. All numeric columns will use default s(x) with thin plate splines.
                        </div>
                    )}
                </div>
            )}

            {/* Tensor Terms Tab */}
            {activeTab === 'tensor' && formulaMode === 'auto' && (
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="text-sm font-medium text-gray-900">Tensor Product Interactions</h3>
                            <p className="text-xs text-gray-500">
                                Model interactions between multiple variables using tensor products
                            </p>
                        </div>
                        <button
                            onClick={addTensorTerm}
                            className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
                        >
                            + Add Interaction
                        </button>
                    </div>

                    {tensorTerms.length > 0 ? (
                        <div className="space-y-3">
                            {tensorTerms.map((term, idx) => (
                                <div key={idx} className="p-4 bg-white border border-gray-200 rounded-lg">
                                    <div className="flex items-center justify-between mb-3">
                                        <h4 className="font-medium text-gray-900">Interaction {idx + 1}</h4>
                                        <button
                                            onClick={() => removeTensorTerm(idx)}
                                            className="text-red-500 hover:text-red-700 text-sm"
                                        >
                                            Remove
                                        </button>
                                    </div>
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                                        <div className="md:col-span-2">
                                            <label className="block text-xs text-gray-500 mb-1">Variables (select 2+)</label>
                                            <div className="flex flex-wrap gap-2 p-2 bg-gray-50 rounded min-h-[40px]">
                                                {numericColumns.map(col => (
                                                    <label key={col} className="inline-flex items-center">
                                                        <input
                                                            type="checkbox"
                                                            checked={term.variables.includes(col)}
                                                            onChange={(e) => {
                                                                const newVars = e.target.checked
                                                                    ? [...term.variables, col]
                                                                    : term.variables.filter(v => v !== col);
                                                                updateTensorTerm(idx, { variables: newVars });
                                                            }}
                                                            className="form-checkbox text-blue-600 rounded"
                                                        />
                                                        <span className="ml-1 text-xs text-gray-700">{col}</span>
                                                    </label>
                                                ))}
                                            </div>
                                        </div>
                                        <div>
                                            <label className="block text-xs text-gray-500 mb-1">Type</label>
                                            <select
                                                value={term.type}
                                                onChange={(e) => updateTensorTerm(idx, { type: e.target.value as 'te' | 'ti' })}
                                                className="w-full px-2 py-1 border border-gray-300 rounded text-sm"
                                            >
                                                <option value="te">te() - Full Tensor (main + interaction)</option>
                                                <option value="ti">ti() - Interaction Only</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center py-8 text-gray-400 italic border-2 border-dashed border-gray-200 rounded-lg">
                            No tensor interactions configured. Click "Add Interaction" to model variable interactions.
                        </div>
                    )}

                    <div className="p-3 bg-blue-50 rounded-lg">
                        <h4 className="text-sm font-medium text-blue-900 mb-1">When to use tensor products?</h4>
                        <ul className="text-xs text-blue-800 list-disc list-inside space-y-1">
                            <li><strong>te()</strong>: Use when you expect both main effects and their interaction</li>
                            <li><strong>ti()</strong>: Use when you already have main effects and only want the pure interaction</li>
                            <li>Common use case: te(longitude, latitude) for spatial data</li>
                        </ul>
                    </div>
                </div>
            )}

            {/* Formula mode notice for non-auto */}
            {formulaMode === 'manual' && activeTab !== 'basic' && (
                <div className="text-center py-8 text-gray-400 italic">
                    Smooth and tensor term configuration is disabled in manual formula mode.
                    <br />
                    Switch to Auto mode to use the visual term builder.
                </div>
            )}
        </div>
    );
}
