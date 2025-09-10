import { useState, useEffect } from 'react';
import { 
  Play, 
  Pause, 
  Square, 
  Settings,
  TrendingUp,
  Shield
} from 'lucide-react';

// Types
interface SimulationConfig {
  name: string;
  duration_days: number;
  customer_count: number;
  scenario_type: 'retail' | 'high_volume' | 'fraud_detection';
  fraud_rate: number;
}

interface SimulationStatus {
  simulation_id: string;
  state: 'idle' | 'running' | 'paused' | 'completed' | 'error';
  progress: number;
  current_sim_time: number;
  agents_active: number;
  events_processed: number;
}

interface MetricData {
  name: string;
  value: number;
  unit: string;
  change?: number;
}

function App() {
  const [simulationStatus, setSimulationStatus] = useState<SimulationStatus | null>(null);
  const [config, setConfig] = useState<SimulationConfig>({
    name: 'Banking Fraud Detection Simulation',
    duration_days: 7,
    customer_count: 1000,
    scenario_type: 'fraud_detection',
    fraud_rate: 0.02
  });
  const [metrics, setMetrics] = useState<MetricData[]>([]);
  const [fraudMetrics, setFraudMetrics] = useState({
    precision: 0,
    recall: 0,
    f1_score: 0,
    accuracy: 0
  });
  const [isConfigOpen, setIsConfigOpen] = useState(false);

  // Mock API calls - replace with real API
  const mockApiCall = async (endpoint: string, method: string = 'GET', body?: any) => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    if (endpoint === '/api/simulation/start' && method === 'POST') {
      return {
        simulation_id: 'sim_' + Math.random().toString(36).substr(2, 9),
        status: 'started'
      };
    }
    
    if (endpoint.includes('/api/simulation/') && endpoint.includes('/status')) {
      return {
        simulation_id: simulationStatus?.simulation_id || 'sim_123',
        state: simulationStatus?.state || 'idle',
        progress: Math.min((simulationStatus?.progress || 0) + Math.random() * 0.1, 1),
        current_sim_time: (simulationStatus?.current_sim_time || 0) + 3600,
        agents_active: 5,
        events_processed: (simulationStatus?.events_processed || 0) + Math.floor(Math.random() * 50)
      } as SimulationStatus;
    }
    
    return {};
  };

  const startSimulation = async () => {
    try {
      const response = await mockApiCall('/api/simulation/start', 'POST', config);
      setSimulationStatus({
        simulation_id: response.simulation_id ?? 'sim_123',
        state: 'running',
        progress: 0,
        current_sim_time: 0,
        agents_active: 5,
        events_processed: 0
      });
    } catch (error) {
      console.error('Failed to start simulation:', error);
    }
  };

  const pauseSimulation = async () => {
    if (simulationStatus) {
      setSimulationStatus({
        ...simulationStatus,
        state: 'paused'
      });
    }
  };

  const stopSimulation = async () => {
    if (simulationStatus) {
      setSimulationStatus({
        ...simulationStatus,
        state: 'completed'
      });
    }
  };

  // Update simulation status periodically
  useEffect(() => {
    if (simulationStatus?.state === 'running') {
      const interval = setInterval(async () => {
        const status = await mockApiCall(`/api/simulation/${simulationStatus.simulation_id}/status`) as SimulationStatus;
        setSimulationStatus(status);
        
        // Update mock metrics
        setMetrics([
          { name: 'Transactions Processed', value: status.events_processed, unit: '', change: 12 },
          { name: 'Active Customers', value: Math.floor(config.customer_count * 0.3), unit: '', change: 2 },
          { name: 'Total Volume', value: status.events_processed * 157.32, unit: '$', change: 8 },
          { name: 'Average Response Time', value: 0.045 + Math.random() * 0.01, unit: 's', change: -5 },
          { name: 'Fraud Detected', value: Math.floor(status.events_processed * config.fraud_rate), unit: '', change: 15 },
          { name: 'False Positives', value: Math.floor(status.events_processed * 0.001), unit: '', change: -3 }
        ]);

        // Update fraud metrics
        setFraudMetrics({
          precision: 0.92 + Math.random() * 0.05,
          recall: 0.88 + Math.random() * 0.05,
          f1_score: 0.90 + Math.random() * 0.05,
          accuracy: 0.95 + Math.random() * 0.03
        });
      }, 2000);

      return () => clearInterval(interval);
    }
  }, [simulationStatus?.state, simulationStatus?.simulation_id, config.fraud_rate, config.customer_count, mockApiCall]);

  const getStatusColor = (state: string) => {
    switch (state) {
      case 'running': return 'text-green-600 bg-green-100';
      case 'paused': return 'text-yellow-600 bg-yellow-100';
      case 'completed': return 'text-blue-600 bg-blue-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatValue = (value: number, unit: string) => {
    if (unit === '$') {
      return `$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
    }
    if (unit === 's') {
      return `${value.toFixed(3)}s`;
    }
    if (Number.isInteger(value)) {
      return value.toLocaleString();
    }
    return value.toFixed(2);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Shield className="h-8 w-8 text-blue-600" />
                <h1 className="text-xl font-bold text-gray-900">Banking Simulator</h1>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setIsConfigOpen(true)}
                className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-md"
              >
                <Settings className="h-4 w-4" />
                <span>Configure</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Control Panel */}
        <div className="bg-white rounded-lg shadow mb-8">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">Simulation Control</h2>
              {simulationStatus && (
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(simulationStatus.state)}`}>
                    {simulationStatus.state.toUpperCase()}
                  </span>
                  <span className="text-sm text-gray-500">
                    ID: {simulationStatus.simulation_id.substring(0, 8)}...
                  </span>
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-4 mb-4">
              <button
                onClick={startSimulation}
                disabled={simulationStatus?.state === 'running'}
                className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play className="h-4 w-4" />
                <span>Start</span>
              </button>
              
              <button
                onClick={pauseSimulation}
                disabled={simulationStatus?.state !== 'running'}
                className="flex items-center space-x-2 px-4 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Pause className="h-4 w-4" />
                <span>Pause</span>
              </button>
              
              <button
                onClick={stopSimulation}
                disabled={!simulationStatus || simulationStatus.state === 'idle'}
                className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Square className="h-4 w-4" />
                <span>Stop</span>
              </button>
            </div>

            {simulationStatus && simulationStatus.state === 'running' && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm text-gray-600">
                  <span>Progress</span>
                  <span>{(simulationStatus.progress * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${simulationStatus.progress * 100}%` }}
                  ></div>
                </div>
                <div className="flex justify-between text-xs text-gray-500">
                  <span>Simulation Time: {Math.floor(simulationStatus.current_sim_time / 3600)}h</span>
                  <span>Active Agents: {simulationStatus.agents_active}</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Metrics Grid */}
        {metrics.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {metrics.map((metric, index) => (
              <div key={index} className="bg-white rounded-lg shadow p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">{metric.name}</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {formatValue(metric.value, metric.unit)}
                    </p>
                  </div>
                  <div className="flex items-center space-x-1">
                    {metric.change !== undefined && (
                      <>
                        {metric.change > 0 ? (
                          <TrendingUp className="h-4 w-4 text-green-500" />
                        ) : (
                          <TrendingUp className="h-4 w-4 text-red-500 rotate-180" />
                        )}
                        <span className={`text-sm font-medium ${metric.change > 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {Math.abs(metric.change)}%
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Fraud Detection Performance */}
        {simulationStatus?.state === 'running' && (
          <div className="bg-white rounded-lg shadow mb-8">
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Fraud Detection Performance</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {(fraudMetrics.precision * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">Precision</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {(fraudMetrics.recall * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">Recall</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {(fraudMetrics.f1_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">F1 Score</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-indigo-600">
                    {(fraudMetrics.accuracy * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600">Accuracy</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Configuration Modal */}
        {isConfigOpen && (
          <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4">
              <div className="p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Simulation Configuration</h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Simulation Name</label>
                    <input
                      type="text"
                      value={config.name}
                      onChange={(e) => setConfig({...config, name: e.target.value})}
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Duration (days)</label>
                    <input
                      type="number"
                      value={config.duration_days}
                      onChange={(e) => setConfig({...config, duration_days: parseInt(e.target.value)})}
                      min="1"
                      max="365"
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Customer Count</label>
                    <input
                      type="number"
                      value={config.customer_count}
                      onChange={(e) => setConfig({...config, customer_count: parseInt(e.target.value)})}
                      min="100"
                      max="10000"
                      step="100"
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Scenario Type</label>
                    <select
                      value={config.scenario_type}
                      onChange={(e) => setConfig({...config, scenario_type: e.target.value as any})}
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    >
                      <option value="retail">Retail Banking</option>
                      <option value="high_volume">High Volume</option>
                      <option value="fraud_detection">Fraud Detection</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Fraud Rate ({(config.fraud_rate * 100).toFixed(1)}%)
                    </label>
                    <input
                      type="range"
                      min="0.001"
                      max="0.1"
                      step="0.001"
                      value={config.fraud_rate}
                      onChange={(e) => setConfig({...config, fraud_rate: parseFloat(e.target.value)})}
                      className="mt-1 block w-full"
                    />
                  </div>
                </div>
                
                <div className="flex justify-end space-x-3 mt-6">
                  <button
                    onClick={() => setIsConfigOpen(false)}
                    className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => setIsConfigOpen(false)}
                    className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
                  >
                    Save Configuration
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;