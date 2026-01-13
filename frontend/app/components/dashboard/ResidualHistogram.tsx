"use client";

import dynamic from 'next/dynamic';
import type { ResidualAnalysis } from '@/app/lib/explanationApi';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ResidualHistogramProps {
    data: ResidualAnalysis;
    compact?: boolean;
}

export default function ResidualHistogram({ data, compact = false }: ResidualHistogramProps) {
    if (!data?.points || data.points.length === 0) {
        return (
            <div className="flex items-center justify-center h-48 text-gray-400">
                No residual data available
            </div>
        );
    }

    const height = compact ? 280 : 400;
    const residuals = data.points.map(p => p.residual);

    // Histogram trace
    const histogramTrace = {
        type: 'histogram' as const,
        x: residuals,
        marker: {
            color: '#6366F1',
            line: {
                color: '#4F46E5',
                width: 1,
            },
        },
        opacity: 0.75,
        name: 'Residuals',
        hovertemplate: 'Range: %{x}<br>Count: %{y}<extra></extra>',
    };

    // Add vertical line at zero
    const zeroLine = {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: [0, 0],
        y: [0, data.points.length / 5], // Approximate height
        line: {
            color: '#EF4444',
            width: 2,
            dash: 'dash' as const,
        },
        name: 'Zero',
        hoverinfo: 'skip' as const,
    };

    const layout = {
        margin: compact 
            ? { t: 20, r: 20, b: 50, l: 50 } 
            : { t: 30, r: 30, b: 60, l: 60 },
        height,
        xaxis: {
            title: compact ? undefined : { text: 'Residual (Actual - Predicted)', standoff: 10 },
            gridcolor: '#E5E7EB',
            zeroline: true,
            zerolinecolor: '#EF4444',
            zerolinewidth: 2,
        },
        yaxis: {
            title: compact ? undefined : { text: 'Frequency', standoff: 10 },
            gridcolor: '#E5E7EB',
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        showlegend: false,
        bargap: 0.05,
    };

    const config = {
        displayModeBar: false,
        responsive: true,
    };

    return (
        <div>
            {compact && (
                <div className="text-xs text-gray-500 text-center mb-1">
                    Errors centered at 0 = unbiased predictions
                </div>
            )}
            <Plot
                data={[histogramTrace, zeroLine]}
                layout={layout}
                config={config}
                style={{ width: '100%' }}
            />
            {!compact && (
                <div className="flex justify-center gap-6 mt-2 text-sm text-gray-600">
                    <span>Mean: <strong>{data.mean_residual.toFixed(4)}</strong></span>
                    <span>Std: <strong>{data.std_residual.toFixed(4)}</strong></span>
                </div>
            )}
        </div>
    );
}
