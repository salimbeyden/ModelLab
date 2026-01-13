"use client";

import dynamic from 'next/dynamic';
import type { ActualVsPredicted as ActualVsPredictedData } from '@/app/lib/explanationApi';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ActualVsPredictedProps {
    data: ActualVsPredictedData;
    compact?: boolean;
}

export default function ActualVsPredicted({ data, compact = false }: ActualVsPredictedProps) {
    if (!data?.actual || data.actual.length === 0) {
        return (
            <div className="flex items-center justify-center h-48 text-gray-400">
                No data available
            </div>
        );
    }

    const height = compact ? 280 : 400;
    const { actual, predicted, perfect_line } = data;

    // Scatter plot of actual vs predicted
    const scatterTrace = {
        type: 'scatter' as const,
        mode: 'markers' as const,
        x: actual,
        y: predicted,
        marker: {
            color: '#6366F1',
            size: compact ? 5 : 8,
            opacity: 0.6,
        },
        name: 'Predictions',
        hovertemplate: 'Actual: %{x:.3f}<br>Predicted: %{y:.3f}<extra></extra>',
    };

    // Perfect prediction line (y = x)
    const lineTrace = {
        type: 'scatter' as const,
        mode: 'lines' as const,
        x: [perfect_line.min, perfect_line.max],
        y: [perfect_line.min, perfect_line.max],
        line: {
            color: '#EF4444',
            width: 2,
            dash: 'dash' as const,
        },
        name: 'Perfect Fit',
        hoverinfo: 'skip' as const,
    };

    const layout = {
        margin: compact 
            ? { t: 20, r: 20, b: 50, l: 50 } 
            : { t: 30, r: 30, b: 60, l: 60 },
        height,
        xaxis: {
            title: compact ? undefined : { text: 'Actual Values', standoff: 10 },
            gridcolor: '#E5E7EB',
            zeroline: false,
        },
        yaxis: {
            title: compact ? undefined : { text: 'Predicted Values', standoff: 10 },
            gridcolor: '#E5E7EB',
            zeroline: false,
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        showlegend: !compact,
        legend: compact ? undefined : {
            orientation: 'h' as const,
            yanchor: 'bottom' as const,
            y: 1.02,
            xanchor: 'right' as const,
            x: 1,
        },
        hovermode: 'closest' as const,
    };

    const config = {
        displayModeBar: false,
        responsive: true,
    };

    return (
        <div>
            {compact && (
                <div className="text-xs text-gray-500 text-center mb-1">
                    Points near the red line = better predictions
                </div>
            )}
            <Plot
                data={[scatterTrace, lineTrace]}
                layout={layout}
                config={config}
                style={{ width: '100%' }}
            />
        </div>
    );
}
