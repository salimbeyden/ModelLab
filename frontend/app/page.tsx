import Link from 'next/link';
import { getDatasets } from '@/app/lib/api';

export default async function Home() {
  // We can fetch data here for a quick summary stats if we want
  // For MVP, just links

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
      <main className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h1 className="text-4xl font-extrabold text-gray-900 sm:text-5xl sm:tracking-tight lg:text-6xl">
            ModelLab
          </h1>
          <p className="mt-5 max-w-xl mx-auto text-xl text-gray-500">
            Explainable AI Workbench for Dataset Preparation and Model Training.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Card 1: Datasets */}
          <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg font-medium leading-6 text-gray-900">Datasets</h3>
              <div className="mt-2 max-w-xl text-sm text-gray-500">
                <p>Upload CSV/Parquet files and profile your data before training.</p>
              </div>
              <div className="mt-5">
                <Link href="/datasets" className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none">
                  Manage Datasets
                </Link>
              </div>
            </div>
          </div>

          {/* Card 2: Models */}
          <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg font-medium leading-6 text-gray-900">Trained Models</h3>
              <div className="mt-2 max-w-xl text-sm text-gray-500">
                <p>Browse and reuse your trained models for predictions and analysis.</p>
              </div>
              <div className="mt-5">
                <Link href="/models" className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-purple-600 hover:bg-purple-700 focus:outline-none">
                  View Models
                </Link>
              </div>
            </div>
          </div>

          {/* Card 3: Runs */}
          <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow">
            <div className="px-4 py-5 sm:p-6">
              <h3 className="text-lg font-medium leading-6 text-gray-900">Training Runs</h3>
              <div className="mt-2 max-w-xl text-sm text-gray-500">
                <p>Configure model training, monitor progress, and view explanations.</p>
              </div>
              <div className="mt-5">
                <Link href="/runs/new" className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none">
                  Start New Run
                </Link>
              </div>
            </div>
          </div>
        </div>


      </main>
    </div>
  );
}
