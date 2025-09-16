"""
Banking Simulation Demo
Complete demonstration of the banking environment simulator
"""
import asyncio
import logging
from datetime import datetime
import json

# Fix import issues
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.simulation_engine import SimulationConfig
from environments.banking.simulation import create_banking_simulation


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def run_banking_demo():
    """Run comprehensive banking simulation demo"""
    
    print("\n" + "="*70)
    print("BANKING ENVIRONMENT SIMULATOR - DEMO")
    print("="*70)
    
    # Demo Configuration
    config = SimulationConfig(
        name="Banking Demo Simulation",
        duration_days=0.1,  # ~2.4 hours simulated time
        time_acceleration=100.0,  # 100x real-time speed
        seed=42,
        environment_type="banking"
    )
    
    print(f"\nSimulation Configuration:")
    print(f"  Name: {config.name}")
    print(f"  Duration: {config.duration_days} days ({config.duration_days * 24:.1f} hours)")
    print(f"  Acceleration: {config.time_acceleration}x")
    print(f"  Expected Runtime: ~{(config.duration_days * 24 * 3600) / config.time_acceleration:.1f} seconds")
    
    # Create simulation engine
    print(f"\n🏗️  Creating banking simulation engine...")
    engine = create_banking_simulation(config)
    
    # Environment configuration
    environment_config = {
        "scenario_type": "fraud_detection",  # Focus on fraud detection
        "scenario_params": {
            "customer_count": 200  # Manageable number for demo
        },
        "agents": [
            {
                "agent_id": "demo_fraud_detector",
                "agent_type": "mock",
                "name": "Demo Fraud Detection Agent",
                "max_concurrent_requests": 8,
                "model_config": {
                    "fraud_threshold": 0.8,
                    "amount_threshold": 1000.0,
                    "velocity_threshold": 5
                }
            }
        ]
    }
    
    print(f"\nEnvironment Configuration:")
    print(f"  Scenario: {environment_config['scenario_type']}")
    print(f"  Customers: {environment_config['scenario_params']['customer_count']}")
    print(f"  Agents: {len(environment_config['agents'])}")
    
    # Initialize simulation
    print(f"\n⚙️  Initializing simulation...")
    await engine.initialize(environment_config)
    
    print(f"✅ Simulation initialized successfully!")
    print(f"   Simulation ID: {engine.simulation_id}")
    if engine.banking_scenario:
        print(f"   Banking Scenario: {engine.banking_scenario.config.name}")
        print(f"   Customers: {len(engine.banking_scenario.customers)}")
        print(f"   Merchants: {len(engine.banking_scenario.merchants)}")
    else:
        print(f"   Banking Scenario: Not initialized")
        print(f"   Customers: N/A")
        print(f"   Merchants: N/A")
    
    # Add some manual events for demonstration
    print(f"\n📝 Adding demonstration events...")
    
    # Create some test transactions
    from core.event_dispatcher import create_transaction_event
    
    demo_transactions = [
        # Normal transactions
        {"id": "demo_001", "amount": 85.50, "customer": "demo_customer_1", "merchant": "Coffee Shop"},
        {"id": "demo_002", "amount": 156.75, "customer": "demo_customer_2", "merchant": "Grocery Store"},
        {"id": "demo_003", "amount": 45.20, "customer": "demo_customer_3", "merchant": "Gas Station"},
        
        # Suspicious transactions
        {"id": "demo_004", "amount": 2500.00, "customer": "demo_customer_1", "merchant": "Electronics Store"},
        {"id": "demo_005", "amount": 5000.00, "customer": "demo_customer_4", "merchant": "Jewelry Store"},
    ]
    
    for tx in demo_transactions:
        event = create_transaction_event(
            transaction_id=tx["id"],
            amount=tx["amount"],
            customer_id=tx["customer"],
            merchant=tx["merchant"],
            requires_fraud_check=True
        )
        
        # Add metadata to make some transactions suspicious
        if tx["amount"] > 1000:
            event.data["metadata"] = {
                "unusual_location": True,
                "velocity_flag": True,
                "is_test_fraud": True  # Mark for validation
            }
        
        await engine.event_dispatcher.dispatch(event)
    
    print(f"✅ Added {len(demo_transactions)} demonstration transactions")
    
    # Set up monitoring
    events_processed = []
    fraud_responses = []
    
    def event_monitor(event):
        events_processed.append(event)
        if event.event_type == "fraud_detection_response":
            fraud_responses.append(event)
            
        # Log interesting events
        if event.event_type in ["transaction", "fraud_detection_response", "security_alert"]:
            logger.info(f"Event: {event.event_type} - {event.data.get('transaction_id', 'N/A')}")
    
    engine.add_event_callback(event_monitor)
    
    # Add checkpoint monitoring
    def checkpoint_monitor(checkpoint):
        sim_time_hours = checkpoint['sim_time'] / 3600
        metrics = checkpoint.get('metrics', {})
        events_count = metrics.get('events_processed', {}).get('value', 0)
        
        print(f"📊 Checkpoint - Sim Time: {sim_time_hours:.1f}h, Events: {events_count}")
    
    engine.add_checkpoint_callback(checkpoint_monitor)
    
    # Run simulation
    print(f"\n🚀 Starting simulation...")
    print(f"   Expected completion in ~{(config.duration_days * 24 * 3600) / config.time_acceleration:.1f} seconds")
    print(f"   Monitoring events in real-time...")
    
    start_time = datetime.now()
    
    try:
        results = await engine.run()
        end_time = datetime.now()
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"   Runtime: {(end_time - start_time).total_seconds():.2f} seconds")
        print(f"   Final State: {results['state']}")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        return
    
    # Display results
    print(f"\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    
    # Basic statistics
    print(f"\n📈 Basic Statistics:")
    print(f"   Simulation ID: {results['simulation_id']}")
    print(f"   Duration: {results.get('duration', 0):.2f} seconds")
    print(f"   Final Simulation Time: {results.get('final_sim_time', 0):.0f} seconds")
    print(f"   Events Processed: {len(events_processed)}")
    print(f"   Fraud Responses: {len(fraud_responses)}")
    
    # Banking-specific results
    if 'banking_report' in results:
        banking_report = results['banking_report']
        
        print(f"\n🏦 Banking Analysis:")
        
        # Transaction analysis
        if 'transaction_analysis' in banking_report:
            tx_analysis = banking_report['transaction_analysis']
            print(f"   Total Transactions: {tx_analysis.get('transaction_count', 0)}")
            print(f"   Total Volume: ${tx_analysis.get('total_amount', 0):,.2f}")
            print(f"   Average Amount: ${tx_analysis.get('average_amount', 0):.2f}")
            
            # Channel distribution
            channel_dist = tx_analysis.get('channel_distribution', {})
            if channel_dist:
                print(f"   Channel Distribution:")
                for channel, count in channel_dist.items():
                    print(f"     {channel}: {count}")
        
        # Fraud detection performance
        if 'fraud_detection' in banking_report:
            fraud_perf = banking_report['fraud_detection']
            print(f"\n🛡️  Fraud Detection Performance:")
            print(f"   Precision: {fraud_perf.get('precision', 0):.1%}")
            print(f"   Recall: {fraud_perf.get('recall', 0):.1%}")
            print(f"   F1 Score: {fraud_perf.get('f1_score', 0):.1%}")
            print(f"   Accuracy: {fraud_perf.get('accuracy', 0):.1%}")
            print(f"   True Positives: {fraud_perf.get('true_positives', 0)}")
            print(f"   False Positives: {fraud_perf.get('false_positives', 0)}")
            print(f"   False Negatives: {fraud_perf.get('false_negatives', 0)}")
        
        # Customer insights
        if 'customer_insights' in banking_report:
            customer_insights = banking_report['customer_insights']
            print(f"\n👥 Customer Insights:")
            print(f"   Total Customers: {customer_insights.get('total_customers', 0)}")
            print(f"   Active Customers: {customer_insights.get('active_customers', 0)}")
            print(f"   Avg Transactions/Customer: {customer_insights.get('avg_transactions_per_customer', 0):.1f}")
    
    # Fraud summary
    if 'fraud_summary' in results:
        fraud_summary = results['fraud_summary']
        print(f"\n🔍 Fraud Detection Summary:")
        for key, value in fraud_summary.items():
            if key in ['precision', 'recall', 'f1_score', 'accuracy']:
                print(f"   {key.title()}: {value}")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value}")
    
    # Agent performance
    if 'agent_performance' in results:
        agent_perf = results['agent_performance']
        print(f"\n🤖 Agent Performance:")
        print(f"   Total Requests: {agent_perf.get('total_requests', 0)}")
        print(f"   Success Rate: {agent_perf.get('success_rate', 0):.1%}")
        print(f"   Total Failures: {agent_perf.get('total_failures', 0)}")
    
    # Event summary
    if 'event_summary' in results:
        event_summary = results['event_summary']
        print(f"\n📨 Event Processing Summary:")
        print(f"   Total Events: {event_summary.get('total_events', 0)}")
        print(f"   Successful Events: {event_summary.get('successful_events', 0)}")
        print(f"   Failed Events: {event_summary.get('failed_events', 0)}")
        
        most_common = event_summary.get('most_common_events', [])
        if most_common:
            print(f"   Most Common Event Types:")
            for event_type, count in most_common[:5]:
                print(f"     {event_type}: {count}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"banking_demo_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results file: {e}")
    
    print(f"\n" + "="*50)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    return results


async def run_performance_test():
    """Run performance test with different customer counts"""
    
    print(f"\n🚀 PERFORMANCE TEST")
    print(f"="*30)
    
    customer_counts = [100, 500, 1000]
    results = []
    
    for count in customer_counts:
        print(f"\nTesting with {count} customers...")
        
        config = SimulationConfig(
            name=f"Performance Test {count}",
            duration_days=0.02,  # Short duration
            time_acceleration=500.0,  # Fast execution
            environment_type="banking"
        )
        
        engine = create_banking_simulation(config)
        
        environment_config = {
            "scenario_type": "high_volume",
            "scenario_params": {"customer_count": count},
            "agents": [
                {
                    "agent_id": f"perf_agent_{count}",
                    "agent_type": "mock",
                    "name": f"Performance Agent {count}",
                    "max_concurrent_requests": min(10, count // 20)
                }
            ]
        }
        
        start_time = datetime.now()
        
        try:
            await engine.initialize(environment_config)
            simulation_results = await engine.run()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract metrics
            banking_report = simulation_results.get('banking_report', {})
            tx_analysis = banking_report.get('transaction_analysis', {})
            tx_count = tx_analysis.get('transaction_count', 0)
            throughput = tx_count / duration if duration > 0 else 0
            
            results.append({
                'customers': count,
                'duration': duration,
                'transactions': tx_count,
                'throughput': throughput
            })
            
            print(f"  ✅ {count} customers: {duration:.2f}s, {tx_count} transactions, {throughput:.1f} tx/sec")
            
        except Exception as e:
            print(f"  ❌ {count} customers: Failed - {e}")
            results.append({
                'customers': count,
                'duration': None,
                'transactions': 0,
                'throughput': 0,
                'error': str(e)
            })
    
    print(f"\n📊 Performance Summary:")
    for result in results:
        if result.get('error'):
            print(f"  {result['customers']} customers: FAILED - {result['error']}")
        else:
            print(f"  {result['customers']} customers: {result['throughput']:.1f} tx/sec")
    
    return results


if __name__ == "__main__":
    print("Banking Environment Simulator - Demo")
    print("====================================")
    
    choice = input("\nSelect demo type:\n1. Full Banking Demo\n2. Performance Test\n3. Both\nChoice (1-3): ")
    
    if choice == "1":
        asyncio.run(run_banking_demo())
    elif choice == "2":
        asyncio.run(run_performance_test())
    elif choice == "3":
        asyncio.run(run_banking_demo())
        asyncio.run(run_performance_test())
    else:
        print("Running full demo...")
        asyncio.run(run_banking_demo())