"use client";

import dynamic from 'next/dynamic';
import type { ICEPlot as ICEPlotType } from '@/app/lib/explanationApi';

// Dynamic import for Plotly (SSR issues)
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ICEPlotProps {
    icePlot: ICEPlotType;
    mini?: boolean;
}

export default function ICEPlot({ icePlot, mini = false }: ICEPlotProps) {
    if (!icePlot || icePlot.points.length === 0) {
        return (
            <div className="flex items-center justify-center h-48 text-gray-400">
                No ICE plot data available
            </div>
        );
    }

    const { points, feature_name, feature_type, current_value, current_prediction } = icePlot;

    // Extract x and y values
    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);

    const traces: any[] = [];

    // Main ICE curve
    traces.push({
        x: xValues,
        y: yValues,
        type: feature_type === 'categorical' ? 'bar' : 'scatter',
        mode: feature_type === 'categorical' ? undefined : 'lines',
        line: { color: '#3B82F6', width: mini ? 2 : 3 },
        marker: { color: '#3B82F6' },
        fill: feature_type === 'categorical' ? undefined : 'tozeroy',
        fillcolor: 'rgba(59, 130, 246, 0.1)',
        hovertemplate: feature_type === 'categorical' 
            ? '%{x}: %{y:.4f}<extra></extra>'
            : `${feature_name}: %{x:.4f}<br>Prediction: %{y:.4f}<extra></extra>`,
    });

    // Current value marker (vertical line for numeric, highlight for categorical)
    if (current_value !== null && current_value !== undefined) {
        if (feature_type === 'numeric' && typeof current_value === 'number') {
            // Vertical line at current value
            traces.push({
                x: [current_value, current_value],
                y: [Math.min(...yValues), Math.max(...yValues)],
                type: 'scatter',
                mode: 'lines',
                line: { color: '#EF4444', width: 2, dash: 'dash' },
                hoverinfo: 'skip',
                showlegend: false,
            });
            // Point at current prediction
            traces.push({
                x: [current_value],
                y: [current_prediction],
                type: 'scatter',
                mode: 'markers',
                marker: { color: '#EF4444', size: 12, symbol: 'circle' },
                hovertemplate: `Current: ${feature_name}=${current_value}<br>Prediction: ${current_prediction.toFixed(4)}<extra></extra>`,
                showlegend: false,
            });
        }
    }

    const layout: any = {
        margin: mini 
            ? { t: 10, r: 10, b: 30, l: 40 }
            : { t: 20, r: 20, b: 60, l: 60 },
        height: mini ? 150 : 300,
        xaxis: {
            title: mini ? '' : feature_name,
            tickfont: { size: mini ? 9 : 12 },
            gridcolor: '#E5E7EB',
        },
        yaxis: {
            title: mini ? '' : 'Predicted Value',
            tickfont: { size: mini ? 9 : 12 },
            gridcolor: '#E5E7EB',
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
            style={{ width: '100%', height: mini ? 150 : 300 }}
        />
    );
}
