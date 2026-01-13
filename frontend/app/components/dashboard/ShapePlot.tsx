"use client";

import dynamic from 'next/dynamic';
import type { ShapeFunction } from '@/app/lib/explanationApi';

// Dynamic import for Plotly (SSR issues)
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ShapePlotProps {
    shapeFunction?: ShapeFunction;
    mini?: boolean;
}

export default function ShapePlot({ shapeFunction, mini = false }: ShapePlotProps) {
    if (!shapeFunction || shapeFunction.points.length === 0) {
        return (
            <div className="flex items-center justify-center h-48 text-gray-400">
                No shape data available
            </div>
        );
    }

    const { points, feature_name, feature_type } = shapeFunction;

    // Extract x and y values
    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);
    
    // Confidence bands if available
    const hasConfidence = points.some(p => p.y_upper !== null && p.y_lower !== null);
    const yUpper = points.map(p => p.y_upper ?? p.y);
    const yLower = points.map(p => p.y_lower ?? p.y);

    const traces: any[] = [];

    // Confidence band (filled area)
    if (hasConfidence) {
        traces.push({
            x: [...xValues, ...xValues.slice().reverse()],
            y: [...yUpper, ...yLower.slice().reverse()],
            fill: 'toself',
            fillcolor: 'rgba(59, 130, 246, 0.15)',
            line: { color: 'transparent' },
            hoverinfo: 'skip',
            showlegend: false,
            type: 'scatter',
        });
    }

    // Main shape line
    traces.push({
        x: xValues,
        y: yValues,
        type: feature_type === 'categorical' ? 'bar' : 'scatter',
        mode: feature_type === 'categorical' ? undefined : 'lines',
        line: { color: '#3B82F6', width: mini ? 2 : 3 },
        marker: { color: '#3B82F6' },
        hovertemplate: feature_type === 'categorical' 
            ? '%{x}: %{y:.4f}<extra></extra>'
            : `${feature_name}: %{x:.4f}<br>Effect: %{y:.4f}<extra></extra>`,
    });

    // Zero reference line
    traces.push({
        x: feature_type === 'categorical' ? xValues : [Math.min(...(xValues as number[])), Math.max(...(xValues as number[]))],
        y: [0, 0],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#9CA3AF', width: 1, dash: 'dash' },
        hoverinfo: 'skip',
        showlegend: false,
    });

    const layout: any = {
        margin: mini 
            ? { t: 10, r: 10, b: 30, l: 40 }
            : { t: 20, r: 20, b: 60, l: 60 },
        height: mini ? 120 : 350,
        xaxis: {
            title: mini ? '' : feature_name,
            tickfont: { size: mini ? 9 : 12 },
            gridcolor: '#E5E7EB',
        },
        yaxis: {
            title: mini ? '' : 'Effect on Prediction',
            tickfont: { size: mini ? 9 : 12 },
            gridcolor: '#E5E7EB',
            zeroline: false,
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        hovermode: 'closest',
        showlegend: false,
    };

    const config = {
        displayModeBar: !mini,
        responsive: true,
        displaylogo: false,
    };

    return (
        <Plot
            data={traces}
            layout={layout}
            config={config}
            style={{ width: '100%', height: mini ? 120 : 350 }}
        />
    );
}
