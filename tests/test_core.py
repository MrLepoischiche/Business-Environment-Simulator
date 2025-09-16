"""
Unit tests for core simulation framework
Tests all core components: simulation engine, agents, events, metrics
"""
import pytest
import asyncio
import simpy
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from core import (
    SimulationEngine,
    SimulationConfig, 
    SimulationState,
    AgentManager,
    AgentConfig,
    AIAgentInterface,
    MockAIAgent,
    EventDispatcher,
    BusinessEvent,
    EventPriority,
    MetricsCollector,
    MetricDefinition,
    MetricType,
    create_transaction_event
)


class TestSimulationEngine:
    """Test cases for SimulationEngine"""
    
    @pytest.fixture
    def config(self):
        return SimulationConfig(
            name="Test Simulation",
            duration_days=1,
            time_acceleration=100.0,
            seed=42,
            environment_type="test"
        )
    
    @pytest.fixture
    def engine(self, config):
        return SimulationEngine(config)
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization"""
        environment_config = {
            "agents": [
                {
                    "agent_id": "test_agent",
                    "agent_type": "mock",
                    "name": "Test Agent",
                    "max_concurrent_requests": 5
                }
            ]
        }
        
        await engine.initialize(environment_config)
        
        assert engine.state == SimulationState.IDLE
        assert engine.simulation_id is not None
        assert engine.agent_manager is not None
        assert engine.event_dispatcher is not None
        assert engine.metrics_collector is not None
        
    @pytest.mark.asyncio
    async def test_engine_run_simulation(self, engine):
        """Test running a basic simulation"""
        environment_config = {
            "agents": [
                {
                    "agent_id": "test_agent",
                    "agent_type": "mock", 
                    "name": "Test Agent",
                    "max_concurrent_requests": 1
                }
            ]
        }
        
        await engine.initialize(environment_config)
        
        # Add some test events
        test_event = BusinessEvent(
            event_type="test_event",
            data={"test": "data"},
            requires_ai_processing=True
        )
        await engine.event_dispatcher.dispatch(test_event)
        
        # Run short simulation
        engine.config.duration_days = 0.01  # Very short for testing
        results = await engine.run()
        
        assert results["state"] == SimulationState.COMPLETED.value
        assert results["simulation_id"] == engine.simulation_id
        assert "metrics" in results
        assert "agent_performance" in results
        
    @pytest.mark.asyncio 
    async def test_engine_pause_resume(self, engine):
        """Test pausing and resuming simulation"""
        environment_config = {"agents": []}
        await engine.initialize(environment_config)
        
        # Test pause
        await engine.pause()
        assert engine.state == SimulationState.PAUSED
        
        # Test resume
        await engine.resume()
        assert engine.state == SimulationState.RUNNING
        
    @pytest.mark.asyncio
    async def test_engine_stop(self, engine):
        """Test stopping simulation"""
        environment_config = {"agents": []}
        await engine.initialize(environment_config)
        
        await engine.stop()
        assert engine.state == SimulationState.COMPLETED
        
    def test_engine_get_status(self, engine):
        """Test getting engine status"""
        status = engine.get_status()
        
        assert "simulation_id" in status
        assert "state" in status
        assert "current_sim_time" in status
        assert "progress" in status


class TestAgentManager:
    """Test cases for AgentManager"""
    
    @pytest.fixture
    def env(self):
        return simpy.Environment()
    
    @pytest.fixture
    def event_dispatcher(self, env):
        return EventDispatcher(env)
    
    @pytest.fixture
    def agent_manager(self, env, event_dispatcher):
        return AgentManager(env, event_dispatcher)
    
    @pytest.mark.asyncio
    async def test_agent_manager_initialization(self, agent_manager):
        """Test agent manager initialization"""
        agent_configs = [
            {
                "agent_id": "test_agent_1",
                "agent_type": "mock",
                "name": "Test Agent 1",
                "max_concurrent_requests": 3
            },
            {
                "agent_id": "test_agent_2", 
                "agent_type": "mock",
                "name": "Test Agent 2",
                "max_concurrent_requests": 2
            }
        ]
        
        await agent_manager.initialize(agent_configs)
        
        assert len(agent_manager.agent_pools) == 2
        assert "mock" in agent_manager.agent_pools
        assert agent_manager.get_active_agent_count() > 0
        
    @pytest.mark.asyncio
    async def test_mock_agent_functionality(self):
        """Test MockAIAgent functionality"""
        config = AgentConfig(
            agent_id="test_mock",
            agent_type="mock",
            name="Test Mock Agent",
            model_config={"test": "config"}
        )
        
        agent = MockAIAgent("test_mock", config)
        await agent.initialize()
        
        assert agent.state.value == "active"
        
        # Test event processing
        event = BusinessEvent(
            event_type="fraud_detection",
            data={
                "transaction_id": "tx_123",
                "amount": 500.0
            }
        )
        
        response = await agent.process_event(event)
        
        assert "decision" in response
        assert "confidence" in response
        assert "reasoning" in response
        
        await agent.cleanup()
        assert agent.state.value == "stopped"
        
    @pytest.mark.asyncio
    async def test_agent_performance_tracking(self, agent_manager):
        """Test agent performance metrics tracking"""
        agent_configs = [
            {
                "agent_id": "perf_test_agent",
                "agent_type": "mock", 
                "name": "Performance Test Agent",
                "max_concurrent_requests": 1
            }
        ]
        
        await agent_manager.initialize(agent_configs)
        
        # Process some events
        for i in range(5):
            event = BusinessEvent(
                event_type="test_event",
                data={"test_id": i},
                requires_ai_processing=True
            )
            await agent_manager._process_with_agent(event)
            
        # Check performance metrics
        performance = agent_manager.get_performance_metrics()
        
        assert performance["total_requests"] >= 5
        assert "agent_pools" in performance
        assert "success_rate" in performance


class TestEventDispatcher:
    """Test cases for EventDispatcher"""
    
    @pytest.fixture
    def env(self):
        return simpy.Environment()
    
    @pytest.fixture
    def dispatcher(self, env):
        return EventDispatcher(env)
    
    @pytest.mark.asyncio
    async def test_event_creation_and_dispatch(self, dispatcher):
        """Test creating and dispatching events"""
        received_events = []
        
        def event_handler(event):
            received_events.append(event)
            
        # Subscribe to events
        dispatcher.subscribe("test_event", event_handler)
        
        # Create and dispatch event
        event = BusinessEvent(
            event_type="test_event",
            data={"message": "test"},
            priority=EventPriority.NORMAL
        )
        
        success = await dispatcher.dispatch(event)
        assert success
        
        # Wait a bit for event processing
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        assert received_events[0].event_type == "test_event"
        assert received_events[0].data["message"] == "test"
        
    @pytest.mark.asyncio
    async def test_event_filtering(self, dispatcher):
        """Test event filtering functionality"""
        high_priority_events = []
        all_events = []
        
        def high_priority_handler(event):
            high_priority_events.append(event)
            
        def all_events_handler(event):
            all_events.append(event)
            
        # Subscribe with different filters
        dispatcher.subscribe(
            "test_event",
            high_priority_handler,
            priorities={EventPriority.HIGH, EventPriority.CRITICAL}
        )
        dispatcher.subscribe("test_event", all_events_handler)
        
        # Dispatch events with different priorities
        events = [
            BusinessEvent("test_event", {"id": 1}, EventPriority.LOW),
            BusinessEvent("test_event", {"id": 2}, EventPriority.NORMAL),
            BusinessEvent("test_event", {"id": 3}, EventPriority.HIGH),
            BusinessEvent("test_event", {"id": 4}, EventPriority.CRITICAL)
        ]
        
        for event in events:
            await dispatcher.dispatch(event)
            
        await asyncio.sleep(0.1)  # Wait for processing
        
        # Check filtering worked
        assert len(all_events) == 4
        assert len(high_priority_events) == 2  # Only HIGH and CRITICAL
        
    def test_transaction_event_helper(self):
        """Test transaction event creation helper"""
        event = create_transaction_event(
            transaction_id="tx_123",
            amount=1500.0,
            customer_id="cust_456",
            merchant="Test Store",
            requires_fraud_check=True
        )
        
        assert event.event_type == "transaction"
        assert event.data["transaction_id"] == "tx_123"
        assert event.data["amount"] == 1500.0
        assert event.requires_ai_processing == True
        assert event.priority == EventPriority.HIGH  # Large amount
        
    def test_event_subscription_management(self, dispatcher):
        """Test subscription management"""
        def handler(event):
            pass
            
        # Test subscription
        sub_id = dispatcher.subscribe("test_event", handler)
        assert sub_id in dispatcher.subscriptions
        
        # Test unsubscription
        success = dispatcher.unsubscribe(sub_id)
        assert success
        assert sub_id not in dispatcher.subscriptions
        
        # Test pause/resume
        sub_id = dispatcher.subscribe("test_event", handler)
        
        assert dispatcher.pause_subscription(sub_id)
        assert not dispatcher.subscriptions[sub_id].active
        
        assert dispatcher.resume_subscription(sub_id)
        assert dispatcher.subscriptions[sub_id].active


class TestMetricsCollector:
    """Test cases for MetricsCollector"""
    
    @pytest.fixture
    def collector(self):
        return MetricsCollector("test_simulation")
    
    @pytest.mark.asyncio
    async def test_metrics_initialization(self, collector):
        """Test metrics collector initialization"""
        await collector.initialize()
        
        # Check that built-in metrics are registered
        assert "events_processed" in collector.metrics
        assert "events_failed" in collector.metrics
        assert "transactions_processed" in collector.metrics
        
    def test_counter_metrics(self, collector):
        """Test counter metric functionality"""
        # Test increment
        collector.increment_counter("events_processed", 5)
        
        metric = collector.metrics["events_processed"]
        assert metric.get_current_value() == 5.0
        
        # Test multiple increments
        collector.increment_counter("events_processed", 3)
        assert metric.get_current_value() == 8.0
        
    def test_gauge_metrics(self, collector):
        """Test gauge metric functionality"""
        # Test set
        collector.set_gauge("active_agents", 10)
        
        metric = collector.metrics["active_agents"]
        assert metric.get_current_value() == 10.0
        
        # Test increment/decrement
        metric.increment(2)
        assert metric.get_current_value() == 12.0
        
        metric.decrement(5)
        assert metric.get_current_value() == 7.0
        
    def test_histogram_metrics(self, collector):
        """Test histogram metric functionality"""
        # Add some observations
        values = [0.1, 0.5, 1.0, 2.5, 0.3, 0.8, 1.5]
        
        for value in values:
            collector.observe_histogram("response_time", value)
            
        metric = collector.metrics["response_time"]
        stats = metric.get_statistics()
        
        assert stats["count"] == len(values)
        assert abs(stats["avg"] - (sum(values) / len(values))) < 0.001
        assert stats["min"] == min(values)
        assert stats["max"] == max(values)
        
    def test_custom_metric_registration(self, collector):
        """Test registering custom metrics"""
        custom_metric = MetricDefinition(
            name="custom_counter",
            metric_type=MetricType.COUNTER,
            description="Custom test counter",
            unit="items"
        )
        
        collector.register_metric(custom_metric)
        
        assert "custom_counter" in collector.metrics
        assert collector.metrics["custom_counter"].definition.description == "Custom test counter"
        
    @pytest.mark.asyncio
    async def test_metrics_summary(self, collector):
        """Test getting metrics summary"""
        await collector.initialize()
        
        # Add some data
        collector.increment_counter("events_processed", 100)
        collector.set_gauge("active_agents", 5)
        collector.observe_histogram("response_time", 0.2)