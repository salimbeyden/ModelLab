"use client";

import dynamic from 'next/dynamic';
import type { ResidualAnalysis } from '@/app/lib/explanationApi';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ResidualPlotProps {
    residuals: ResidualAnalysis;
    type: 'scatter' | 'histogram';
}

export default function ResidualPlot({ residuals, type }: ResidualPlotProps) {
    const { points, mean_residual, std_residual } = residuals;

    if (type === 'scatter') {
        // Residuals vs Predicted scatter plot
        const predicted = points.map(p => p.predicted);
        const residualValues = points.map(p => p.residual);

        const traces: any[] = [
            {
                x: predicted,
                y: residualValues,
                type: 'scatter',
                mode: 'markers',
                marker: {
                    color: residualValues.map(r => r >= 0 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)'),
                    size: 6,
                },
                hovertemplate: 'Predicted: %{x:.4f}<br>Residual: %{y:.4f}<extra></extra>',
            },
            // Zero line
            {
                x: [Math.min(...predicted), Math.max(...predicted)],
                y: [0, 0],
                type: 'scatter',
                mode: 'lines',
                line: { color: '#6B7280', width: 1, dash: 'dash' },
                hoverinfo: 'skip',
            },
            // Mean residual line
            {
                x: [Math.min(...predicted), Math.max(...predicted)],
                y: [mean_residual, mean_residual],
                type: 'scatter',
                mode: 'lines',
                line: { color: '#F59E0B', width: 1.5 },
                hoverinfo: 'skip',
                name: `Mean: ${mean_residual.toFixed(4)}`,
            },
        ];

        const layout = {
            margin: { t: 10, r: 20, b: 50, l: 60 },
            height: 300,
            xaxis: {
                title: { text: 'Predicted Values' },
                gridcolor: '#E5E7EB',
            },
            yaxis: {
                title: { text: 'Residuals' },
                gridcolor: '#E5E7EB',
                zeroline: false,
            },
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            showlegend: false,
            hovermode: 'closest' as const,
        };

        return (
            <Plot
                data={traces}
                layout={layout}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
            />
        );
    }

    // Histogram
    const residualValues = points.map(p => p.residual);

    const trace = {
        x: residualValues,
        type: 'histogram' as const,
        nbinsx: 30,
        marker: {
            color: 'rgba(59, 130, 246, 0.7)',
            line: {
                color: 'rgba(59, 130, 246, 1)',
                width: 1,
            },
        },
        hovertemplate: 'Residual: %{x:.4f}<br>Count: %{y}<extra></extra>',
    };

    // Normal distribution overlay
    const normalTrace = {
        x: residualValues.sort((a, b) => a - b),
        y: residualValues.map(x => {
            const z = (x - mean_residual) / std_residual;
            return (points.length / 5) * Math.exp(-0.5 * z * z) / (std_residual * Math.sqrt(2 * Math.PI));
        }),
        type: 'scatter' as const,
        mode: 'lines' as const,
        line: { color: '#10B981', width: 2 },
        name: 'Normal Distribution',
    };

    const layout = {
        margin: { t: 10, r: 20, b: 50, l: 60 },
        height: 300,
        xaxis: {
            title: { text: 'Residual Value' },
            gridcolor: '#E5E7EB',
        },
        yaxis: {
            title: { text: 'Frequency' },
            gridcolor: '#E5E7EB',
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        showlegend: false,
        bargap: 0.05,
    };

    return (
        <Plot
            data={[trace, normalTrace]}
            layout={layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
        />
    );
}
