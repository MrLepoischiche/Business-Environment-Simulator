"""
Example usage of the Business Environment Simulator Core Framework
Demonstrates basic simulation setup and execution
"""
import asyncio
import logging
from datetime import datetime

from core import (
    create_simulation_engine,
    SimulationConfig,
    BusinessEvent,
    EventPriority,
    create_transaction_event,
    MetricDefinition,
    MetricType
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def run_basic_simulation():
    """Run a basic banking simulation example"""
    
    # 1. Create simulation configuration
    config = SimulationConfig(
        name="Basic Banking Simulation",
        duration_days=1,  # 1 day simulation
        time_acceleration=10.0,  # 10x real-time speed
        seed=42,
        environment_type="banking"
    )
    
    # 2. Create simulation engine
    engine = create_simulation_engine(config)
    
    # 3. Define environment configuration
    environment_config = {
        "agents": [
            {
                "agent_id": "fraud_detector_1",
                "agent_type": "mock",  # Using mock agent for demo
                "name": "Primary Fraud Detector",
                "model_config": {
                    "rules": {
                        "high_amount_threshold": 1000,
                        "fraud_probability": 0.05
                    }
                },
                "max_concurrent_requests": 5,
                "timeout_seconds": 10
            }
        ],
        "customers": {
            "count": 1000,
            "transaction_rate": 2.0  # transactions per hour per customer
        },
        "business_rules": {
            "fraud_threshold": 0.8,
            "max_daily_amount": 10000
        }
    }
    
    # 4. Initialize simulation
    logger.info("Initializing simulation...")
    await engine.initialize(environment_config)
    
    # 5. Set up event callbacks for monitoring
    def on_event(event: BusinessEvent):
        if event.event_type in ["transaction", "fraud_detection_response"]:
            logger.info(f"Event: {event.event_type} - {event.data}")
    
    engine.add_event_callback(on_event)
    
    # 6. Set up checkpoint callbacks
    def on_checkpoint(checkpoint):
        logger.info(f"Checkpoint at sim time {checkpoint['sim_time']}: "
                   f"{checkpoint['metrics']['events_processed']['value']} events processed")
    
    engine.add_checkpoint_callback(on_checkpoint)
    
    # 7. Generate some initial events to kickstart the simulation
    await generate_sample_events(engine)
    
    # 8. Run the simulation
    logger.info("Starting simulation...")
    results = await engine.run()
    
    # 9. Display results
    print_simulation_results(results)
    
    return results


async def generate_sample_events(engine):
    """Generate sample events to drive the simulation"""
    
    # Generate some sample transactions
    sample_transactions = [
        {"transaction_id": "tx_001", "amount": 150.00, "customer_id": "cust_001", "merchant": "Grocery Store"},
        {"transaction_id": "tx_002", "amount": 1500.00, "customer_id": "cust_002", "merchant": "Electronics Store"},
        {"transaction_id": "tx_003", "amount": 50.00, "customer_id": "cust_003", "merchant": "Coffee Shop"},
        {"transaction_id": "tx_004", "amount": 2500.00, "customer_id": "cust_004", "merchant": "Jewelry Store"},
        {"transaction_id": "tx_005", "amount": 75.00, "customer_id": "cust_005", "merchant": "Gas Station"}
    ]
    
    # Dispatch transaction events
    for tx in sample_transactions:
        event = create_transaction_event(
            transaction_id=tx["transaction_id"],
            amount=tx["amount"],
            customer_id=tx["customer_id"],
            merchant=tx["merchant"],
            requires_fraud_check=True
        )
        
        await engine.event_dispatcher.dispatch(event)
        logger.info(f"Generated transaction event: {tx['transaction_id']} - ${tx['amount']}")
        
    # Generate a customer interaction event
    customer_event = BusinessEvent(
        event_type="customer_interaction",
        data={
            "customer_id": "cust_001", 
            "interaction_type": "inquiry",
            "channel": "phone",
            "subject": "Account balance question"
        },
        priority=EventPriority.NORMAL
    )
    
    await engine.event_dispatcher.dispatch(customer_event)
    
    # Generate a system monitoring event
    system_event = BusinessEvent(
        event_type="system_health",
        data={
            "component": "fraud_detection_service",
            "status": "healthy",
            "response_time_ms": 45,
            "cpu_usage": 0.25
        },
        priority=EventPriority.LOW
    )
    
    await engine.event_dispatcher.dispatch(system_event)


def print_simulation_results(results):
    """Print formatted simulation results"""
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
    print(f"Simulation ID: {results['simulation_id']}")
    print(f"Configuration: {results['config']['name']}")
    print(f"Duration: {results['duration']:.2f} seconds")
    print(f"Final State: {results['state']}")
    print(f"Simulation Time: {results['final_sim_time']:.2f}")
    
    print("\n--- METRICS SUMMARY ---")
    metrics = results.get('metrics', {})
    
    for metric_name, metric_data in metrics.items():
        if metric_data.get('statistics') and metric_data['statistics'].get('count', 0) > 0:
            stats = metric_data['statistics']
            print(f"{metric_name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Total: {stats['sum']:.2f}")
            print(f"  Average: {stats['avg']:.2f}")
            print(f"  Min/Max: {stats['min']:.2f} / {stats['max']:.2f}")
    
    print("\n--- AGENT PERFORMANCE ---")
    agent_perf = results.get('agent_performance', {})
    print(f"Total Requests: {agent_perf.get('total_requests', 0)}")
    print(f"Total Failures: {agent_perf.get('total_failures', 0)}")
    print(f"Success Rate: {agent_perf.get('success_rate', 0):.2%}")
    
    print("\n--- EVENT SUMMARY ---")
    event_summary = results.get('event_summary', {})
    print(f"Total Events: {event_summary.get('total_events', 0)}")
    print(f"Successful: {event_summary.get('successful_events', 0)}")
    print(f"Failed: {event_summary.get('failed_events', 0)}")
    
    if event_summary.get('most_common_events'):
        print("Most Common Events:")
        for event_type, count in event_summary['most_common_events']:
            print(f"  {event_type}: {count}")
    
    print("\n--- CHECKPOINTS ---")
    checkpoints = results.get('checkpoints', [])
    print(f"Total Checkpoints: {len(checkpoints)}")
    
    if checkpoints:
        last_checkpoint = checkpoints[-1]
        print(f"Final Checkpoint Time: {last_checkpoint['sim_time']}")
        print(f"Events at End: {last_checkpoint['event_count']}")
    
    print("\n" + "="*60)


async def run_extended_simulation():
    """Run a more complex simulation with custom metrics"""
    
    config = SimulationConfig(
        name="Extended Banking Simulation",
        duration_days=7,  # 1 week simulation
        time_acceleration=100.0,  # 100x speed for faster testing
        checkpoint_interval=300,  # 5-minute checkpoints
        environment_type="banking"
    )
    
    engine = create_simulation_engine(config)
    
    # More complex agent configuration
    environment_config = {
        "agents": [
            {
                "agent_id": "fraud_detector_primary",
                "agent_type": "mock",
                "name": "Primary Fraud Detector",
                "max_concurrent_requests": 10,
                "model_config": {"fraud_threshold": 0.85}
            },
            {
                "agent_id": "risk_assessor",
                "agent_type": "mock", 
                "name": "Risk Assessment Agent",
                "max_concurrent_requests": 5,
                "model_config": {"risk_categories": ["low", "medium", "high"]}
            }
        ]
    }
    
    await engine.initialize(environment_config)
    # Add custom business metrics
    engine.metrics_collector.register_metric(MetricDefinition(
        name="high_value_transactions",
        metric_type=MetricType.COUNTER,
        description="Transactions over $1000",
        unit="count"
    ))
    
    # Generate more diverse events
    await generate_extended_events(engine)
    
    logger.info("Starting extended simulation...")
    results = await engine.run()
    
    print_simulation_results(results)
    return results


async def generate_extended_events(engine):
    """Generate a more diverse set of events"""
    
    # Generate transactions with different patterns
    import random
    
    customers = [f"cust_{i:03d}" for i in range(1, 101)]  # 100 customers
    merchants = ["Grocery Store", "Gas Station", "Restaurant", "ATM", "Online Store", 
                "Department Store", "Electronics", "Pharmacy", "Coffee Shop"]
    
    # Generate 50 random transactions
    for i in range(50):
        amount = random.uniform(10, 2000)
        customer = random.choice(customers)
        merchant = random.choice(merchants)
        
        # Make some transactions more suspicious
        if random.random() < 0.1:  # 10% chance of suspicious transaction
            amount = random.uniform(1500, 5000)  # Higher amounts
            
        event = create_transaction_event(
            transaction_id=f"tx_{i:03d}",
            amount=amount,
            customer_id=customer,
            merchant=merchant
        )
        
        await engine.event_dispatcher.dispatch(event)
        
        # Add small delay to spread events over time
        await asyncio.sleep(0.01)
    
    logger.info("Generated 50 diverse transaction events")


if __name__ == "__main__":
    print("Business Environment Simulator - Core Framework Demo")
    print("====================================================")
    
    # Run basic simulation
    print("\n1. Running Basic Simulation...")
    asyncio.run(run_basic_simulation())
    
    # Wait a moment
    print("\nWaiting 3 seconds before extended simulation...")
    import time
    time.sleep(3)
    
    # Run extended simulation
    print("\n2. Running Extended Simulation...")
    asyncio.run(run_extended_simulation())
    
    print("\nDemo completed successfully!")