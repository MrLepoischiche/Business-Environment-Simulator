"""
Core framework for Business Environment Simulator
Provides simulation engine, agent management, event handling, and metrics collection
"""

from .simulation_engine import (
    SimulationEngine,
    SimulationConfig,
    SimulationState,
    create_simulation_engine
)

from .agent_manager import (
    AgentManager,
    AIAgentInterface,
    AgentConfig,
    AgentState,
    AgentPool,
    MockAIAgent
)

from .event_dispatcher import (
    EventDispatcher,
    BusinessEvent,
    EventPriority,
    EventFilter,
    EventSubscription,
    create_transaction_event,
    create_customer_interaction_event,
    create_system_event
)

from .metrics_collector import (
    MetricsCollector,
    MetricDefinition,
    MetricType,
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    TimerMetric,
    TimerContext
)

__version__ = "0.1.0"
__author__ = "Business Environment Simulator"

__all__ = [
    # Simulation Engine
    "SimulationEngine",
    "SimulationConfig", 
    "SimulationState",
    "create_simulation_engine",
    
    # Agent Management
    "AgentManager",
    "AIAgentInterface",
    "AgentConfig",
    "AgentState", 
    "AgentPool",
    "MockAIAgent",
    
    # Event System
    "EventDispatcher",
    "BusinessEvent",
    "EventPriority",
    "EventFilter",
    "EventSubscription",
    "create_transaction_event",
    "create_customer_interaction_event", 
    "create_system_event",
    
    # Metrics
    "MetricsCollector",
    "MetricDefinition",
    "MetricType",
    "CounterMetric",
    "GaugeMetric", 
    "HistogramMetric",
    "TimerMetric",
    "TimerContext"
]