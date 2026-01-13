"use client";

import type { ConfusionMatrixData } from '@/app/lib/explanationApi';

interface ConfusionMatrixProps {
    data: ConfusionMatrixData;
    compact?: boolean;
}

export default function ConfusionMatrix({ data, compact = false }: ConfusionMatrixProps) {
    const { labels, normalized_matrix, matrix } = data;

    if (!labels || labels.length === 0) {
        return (
            <div className="flex items-center justify-center h-48 text-gray-400">
                No confusion matrix data available
            </div>
        );
    }

    // Color scale from red (0) to green (1)
    const getColor = (value: number) => {
        if (value >= 0.9) return 'bg-green-600 text-white';
        if (value >= 0.7) return 'bg-green-400 text-white';
        if (value >= 0.5) return 'bg-yellow-400 text-gray-900';
        if (value >= 0.3) return 'bg-orange-400 text-white';
        return 'bg-red-400 text-white';
    };

    const cellSize = compact ? 'w-12 h-12 text-xs' : 'w-16 h-16 text-sm';
    const headerSize = compact ? 'text-xs' : 'text-sm';

    return (
        <div className="overflow-auto">
            <div className="inline-block">
                {/* Column headers */}
                <div className="flex">
                    <div className={`${cellSize} flex items-center justify-center`}></div>
                    <div className={`${cellSize} flex items-center justify-center`}></div>
                    {labels.map((label, i) => (
                        <div
                            key={`col-${i}`}
                            className={`${cellSize} flex items-center justify-center font-medium text-gray-700 ${headerSize}`}
                            title={`Predicted: ${label}`}
                        >
                            {label.length > 8 ? label.substring(0, 6) + '...' : label}
                        </div>
                    ))}
                </div>

                {/* Predicted label */}
                <div className="flex items-center justify-center text-xs text-gray-500 font-medium mb-1">
                    <span className="ml-16">← Predicted →</span>
                </div>

                {/* Matrix rows */}
                {labels.map((rowLabel, i) => (
                    <div key={`row-${i}`} className="flex">
                        {/* Row label */}
                        {i === Math.floor(labels.length / 2) && (
                            <div className={`${cellSize} flex items-center justify-center text-xs text-gray-500 font-medium`}
                                style={{ writingMode: 'vertical-rl', textOrientation: 'mixed', transform: 'rotate(180deg)' }}>
                                Actual
                            </div>
                        )}
                        {i !== Math.floor(labels.length / 2) && (
                            <div className={`${cellSize}`}></div>
                        )}
                        <div
                            className={`${cellSize} flex items-center justify-center font-medium text-gray-700 ${headerSize}`}
                            title={`Actual: ${rowLabel}`}
                        >
                            {rowLabel.length > 8 ? rowLabel.substring(0, 6) + '...' : rowLabel}
                        </div>
                        
                        {/* Cells */}
                        {normalized_matrix[i].map((value, j) => (
                            <div
                                key={`cell-${i}-${j}`}
                                className={`${cellSize} flex flex-col items-center justify-center ${getColor(value)} rounded-sm m-0.5 font-medium`}
                                title={`Actual: ${rowLabel}, Predicted: ${labels[j]}\nCount: ${matrix[i][j]}\nRate: ${(value * 100).toFixed(1)}%`}
                            >
                                <span>{(value * 100).toFixed(0)}%</span>
                                {!compact && (
                                    <span className="text-xs opacity-75">({matrix[i][j]})</span>
                                )}
                            </div>
                        ))}
                    </div>
                ))}
            </div>

            {/* Legend */}
            <div className="flex items-center justify-center gap-4 mt-4 text-xs text-gray-600">
                <div className="flex items-center gap-1">
                    <div className="w-4 h-4 bg-red-400 rounded"></div>
                    <span>Low</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-4 h-4 bg-yellow-400 rounded"></div>
                    <span>Medium</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-4 h-4 bg-green-600 rounded"></div>
                    <span>High</span>
                </div>
            </div>
        </div>
    );
}
