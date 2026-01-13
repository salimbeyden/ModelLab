"use client";

import dynamic from 'next/dynamic';
import type { FeatureContribution } from '@/app/lib/explanationApi';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface WaterfallChartProps {
    contributions: FeatureContribution[];
}

export default function WaterfallChart({ contributions }: WaterfallChartProps) {
    if (!contributions || contributions.length === 0) {
        return (
            <div className="flex items-center justify-center h-48 text-gray-400">
                No contribution data available
            </div>
        );
    }

    // Sort by absolute contribution
    const sorted = [...contributions].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    
    // Take top 10 for readability
    const topContributions = sorted.slice(0, 10);

    // Build waterfall data
    const labels = topContributions.map(c => {
        const shortName = c.feature_name.length > 20 
            ? c.feature_name.substring(0, 18) + '...' 
            : c.feature_name;
        const value = typeof c.feature_value === 'number' 
            ? c.feature_value.toFixed(2) 
            : c.feature_value;
        return `${shortName}=${value}`;
    });

    const values = topContributions.map(c => c.contribution);
    const colors = values.map(v => v >= 0 ? '#10B981' : '#EF4444');

    // Calculate cumulative for text positioning
    let cumulative = 0;
    const textValues = values.map(v => {
        const formatted = v >= 0 ? `+${v.toFixed(3)}` : v.toFixed(3);
        cumulative += v;
        return formatted;
    });

    const trace = {
        type: 'bar' as const,
        x: labels,
        y: values,
        marker: {
            color: colors,
            line: {
                color: colors.map(c => c === '#10B981' ? '#059669' : '#DC2626'),
                width: 1,
            },
        },
        text: textValues,
        textposition: 'outside' as const,
        textfont: {
            size: 11,
            color: colors,
        },
        hovertemplate: '%{x}<br>Contribution: %{y:.4f}<extra></extra>',
    };

    const layout = {
        margin: { t: 30, r: 20, b: 100, l: 60 },
        height: 400,
        xaxis: {
            tickangle: -45,
            tickfont: { size: 10 },
        },
        yaxis: {
            title: { text: 'Contribution to Prediction' },
            gridcolor: '#E5E7EB',
            zeroline: true,
            zerolinecolor: '#6B7280',
            zerolinewidth: 2,
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        bargap: 0.3,
        showlegend: false,
    };

    const config = {
        displayModeBar: false,
        responsive: true,
    };

    return (
        <div>
            <Plot
                data={[trace]}
                layout={layout}
                config={config}
                style={{ width: '100%' }}
            />
            
            {/* Legend */}
            <div className="flex justify-center gap-6 mt-2 text-sm">
                <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded bg-green-500"></div>
                    <span className="text-gray-600">Increases prediction</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded bg-red-500"></div>
                    <span className="text-gray-600">Decreases prediction</span>
                </div>
            </div>
        </div>
    );
}
