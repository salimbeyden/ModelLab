"use client";

import dynamic from 'next/dynamic';
import type { PerClassMetrics } from '@/app/lib/explanationApi';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface PerClassMetricsChartProps {
    data: PerClassMetrics[];
    metric?: 'f1_score' | 'precision' | 'recall';
}

export default function PerClassMetricsChart({ data, metric = 'f1_score' }: PerClassMetricsChartProps) {
    if (!data || data.length === 0) {
        return (
            <div className="flex items-center justify-center h-48 text-gray-400">
                No per-class metrics available
            </div>
        );
    }

    // Sort by the selected metric
    const sorted = [...data].sort((a, b) => b[metric] - a[metric]);

    // Include sample count (n=X) in the label for compact display
    const labels = sorted.map(d => `${d.class_name} (n=${d.support})`);
    const values = sorted.map(d => d[metric]);
    
    // Color based on value
    const colors = values.map(v => {
        if (v >= 0.8) return '#10B981'; // green
        if (v >= 0.6) return '#F59E0B'; // yellow
        return '#EF4444'; // red
    });

    const trace = {
        type: 'bar' as const,
        y: labels,
        x: values,
        orientation: 'h' as const,
        marker: {
            color: colors,
            line: {
                color: colors.map(c => c === '#10B981' ? '#059669' : c === '#F59E0B' ? '#D97706' : '#DC2626'),
                width: 1,
            },
        },
        text: values.map(v => (v * 100).toFixed(1) + '%'),
        textposition: 'outside' as const,
        textfont: { size: 11 },
        hovertemplate: '%{y}<br>' + metric.replace('_', ' ') + ': %{x:.2%}<extra></extra>',
    };

    const layout = {
        margin: { t: 10, r: 60, b: 40, l: 80 },
        height: Math.max(200, data.length * 35 + 60),
        xaxis: {
            title: { text: metric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) },
            range: [0, 1.15],
            tickformat: '.0%',
            gridcolor: '#E5E7EB',
        },
        yaxis: {
            tickfont: { size: 11 },
            automargin: true,
        },
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        bargap: 0.3,
    };

    const config = {
        displayModeBar: false,
        responsive: true,
    };

    return (
        <Plot
            data={[trace]}
            layout={layout}
            config={config}
            style={{ width: '100%' }}
        />
    );
}
