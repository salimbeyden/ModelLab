"use client";

import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface FeatureImportanceChartProps {
    importance: Record<string, number>;
    maxFeatures?: number;
}

export default function FeatureImportanceChart({ 
    importance, 
    maxFeatures = 15 
}: FeatureImportanceChartProps) {
    // Sort by importance and take top N
    const sortedFeatures = Object.entries(importance)
        .sort((a, b) => b[1] - a[1])
        .slice(0, maxFeatures);

    const features = sortedFeatures.map(([name]) => name);
    const values = sortedFeatures.map(([, value]) => value);

    // Reverse for horizontal bar chart (highest at top)
    features.reverse();
    values.reverse();

    const trace = {
        type: 'bar' as const,
        x: values,
        y: features,
        orientation: 'h' as const,
        marker: {
            color: values.map((v, i) => {
                const intensity = (i / values.length) * 0.6 + 0.4;
                return `rgba(59, 130, 246, ${intensity})`;
            }),
            line: {
                color: 'rgba(59, 130, 246, 1)',
                width: 1,
            },
        },
        hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
    };

    const layout = {
        margin: { t: 10, r: 20, b: 40, l: 150 },
        height: Math.max(300, features.length * 25 + 60),
        xaxis: {
            title: { text: 'Importance Score' },
            gridcolor: '#E5E7EB',
        },
        yaxis: {
            tickfont: { size: 11 },
            automargin: true,
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        bargap: 0.2,
    };

    const config = {
        displayModeBar: false,
        responsive: true,
    };

    if (features.length === 0) {
        return (
            <div className="flex items-center justify-center h-48 text-gray-400">
                No feature importance data available
            </div>
        );
    }

    return (
        <Plot
            data={[trace]}
            layout={layout}
            config={config}
            style={{ width: '100%' }}
        />
    );
}
