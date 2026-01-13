"use client";

import type { PerformanceResponse, DashboardSummary } from '@/app/lib/explanationApi';

interface MetricsCardsProps {
    performance: PerformanceResponse;
    summary: DashboardSummary;
    detailed?: boolean;
}

export default function MetricsCards({ performance, summary, detailed = false }: MetricsCardsProps) {
    const isRegression = performance.task_type === 'regression';
    const metrics = isRegression ? performance.regression_metrics : performance.classification_metrics;

    if (!metrics) {
        return <div className="text-gray-500">No metrics available</div>;
    }

    // Define metric cards based on task type
    const cards = isRegression ? [
        { label: 'R² Score', value: performance.regression_metrics!.r2, format: 'percent', color: 'blue' },
        { label: 'RMSE', value: performance.regression_metrics!.rmse, format: 'decimal', color: 'orange' },
        { label: 'MAE', value: performance.regression_metrics!.mae, format: 'decimal', color: 'purple' },
        ...(detailed ? [
            { label: 'MAPE', value: performance.regression_metrics!.mape, format: 'percent', color: 'green' },
            { label: 'Explained Variance', value: performance.regression_metrics!.explained_variance, format: 'percent', color: 'indigo' },
            { label: 'Max Error', value: performance.regression_metrics!.max_error, format: 'decimal', color: 'red' },
        ] : []),
        { label: 'Test Samples', value: performance.regression_metrics!.n_samples, format: 'integer', color: 'gray' },
    ] : [
        { label: 'Accuracy', value: performance.classification_metrics!.accuracy, format: 'percent', color: 'blue' },
        { label: 'Precision', value: performance.classification_metrics!.precision, format: 'percent', color: 'green' },
        { label: 'Recall', value: performance.classification_metrics!.recall, format: 'percent', color: 'yellow' },
        { label: 'F1 Score', value: performance.classification_metrics!.f1_score, format: 'percent', color: 'purple' },
        ...(detailed && performance.classification_metrics!.roc_auc ? [
            { label: 'ROC-AUC', value: performance.classification_metrics!.roc_auc, format: 'percent', color: 'indigo' },
        ] : []),
        { label: 'Test Samples', value: performance.classification_metrics!.n_samples, format: 'integer', color: 'gray' },
    ];

    const formatValue = (value: number | undefined | null, format: string) => {
        if (value === undefined || value === null) return 'N/A';
        switch (format) {
            case 'percent':
                return (value * 100).toFixed(2) + '%';
            case 'decimal':
                return value.toFixed(4);
            case 'integer':
                return value.toLocaleString();
            default:
                return value.toString();
        }
    };

    const colorClasses: Record<string, string> = {
        blue: 'bg-blue-50 border-blue-200 text-blue-700',
        green: 'bg-green-50 border-green-200 text-green-700',
        orange: 'bg-orange-50 border-orange-200 text-orange-700',
        purple: 'bg-purple-50 border-purple-200 text-purple-700',
        yellow: 'bg-yellow-50 border-yellow-200 text-yellow-700',
        indigo: 'bg-indigo-50 border-indigo-200 text-indigo-700',
        red: 'bg-red-50 border-red-200 text-red-700',
        gray: 'bg-gray-50 border-gray-200 text-gray-700',
    };

    return (
        <div className={`grid gap-4 ${detailed ? 'grid-cols-2 md:grid-cols-3 lg:grid-cols-4' : 'grid-cols-2 md:grid-cols-4'}`}>
            {cards.map((card, index) => (
                <div
                    key={index}
                    className={`rounded-lg border p-4 ${colorClasses[card.color]}`}
                >
                    <div className="text-sm font-medium opacity-80">{card.label}</div>
                    <div className="text-2xl font-bold mt-1">
                        {formatValue(card.value, card.format)}
                    </div>
                </div>
            ))}
        </div>
    );
}
