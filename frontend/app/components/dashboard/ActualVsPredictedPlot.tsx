"use client";

import dynamic from 'next/dynamic';
import type { ActualVsPredicted } from '@/app/lib/explanationApi';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ActualVsPredictedPlotProps {
    data: ActualVsPredicted;
    fullSize?: boolean;
}

export default function ActualVsPredictedPlot({ data, fullSize = false }: ActualVsPredictedPlotProps) {
    const { actual, predicted, perfect_line } = data;

    const traces: any[] = [
        // Scatter points
        {
            x: actual,
            y: predicted,
            type: 'scatter',
            mode: 'markers',
            marker: {
                color: 'rgba(59, 130, 246, 0.6)',
                size: fullSize ? 8 : 6,
                line: {
                    color: 'rgba(59, 130, 246, 1)',
                    width: 1,
                },
            },
            hovertemplate: 'Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>',
            name: 'Predictions',
        },
        // Perfect prediction line (45-degree)
        {
            x: [perfect_line.min, perfect_line.max],
            y: [perfect_line.min, perfect_line.max],
            type: 'scatter',
            mode: 'lines',
            line: {
                color: '#10B981',
                width: 2,
                dash: 'dash',
            },
            hoverinfo: 'skip',
            name: 'Perfect Prediction',
        },
    ];

    const layout = {
        margin: fullSize 
            ? { t: 20, r: 20, b: 60, l: 60 }
            : { t: 10, r: 10, b: 40, l: 50 },
        height: fullSize ? 450 : 280,
        xaxis: {
            title: { text: 'Actual Values' },
            gridcolor: '#E5E7EB',
            zeroline: false,
        },
        yaxis: {
            title: { text: 'Predicted Values' },
            gridcolor: '#E5E7EB',
            zeroline: false,
            scaleanchor: 'x' as const,
            scaleratio: 1,
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        showlegend: fullSize,
        legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.8)',
        },
        hovermode: 'closest' as const,
    };

    const config = {
        displayModeBar: fullSize,
        responsive: true,
        displaylogo: false,
    };

    return (
        <Plot
            data={traces}
            layout={layout}
            config={config}
            style={{ width: '100%' }}
        />
    );
}
