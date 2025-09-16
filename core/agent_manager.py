"""
Agent Manager for handling AI agents in simulation environment
Manages agent lifecycle, communication, and performance tracking
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import json
from uuid import uuid4
import simpy

from .event_dispatcher import EventDispatcher, BusinessEvent


class AgentState(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    agent_id: str
    agent_type: str
    name: str
    model_path: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    max_concurrent_requests: int = 10
    timeout_seconds: int = 30
    retry_count: int = 3
    

class AIAgentInterface(ABC):
    """Abstract base class for all AI agents in the simulation"""
    
    def __init__(self, agent_id: str, config: AgentConfig):
        self.agent_id = agent_id
        self.config = config
        self.state = AgentState.INACTIVE
        self.performance_metrics = {
            "requests_processed": 0,
            "requests_failed": 0,
            "total_response_time": 0.0,
            "average_response_time": 0.0
        }
        self.logger = logging.getLogger(f"agent.{agent_id}")
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent (load models, etc.)"""
        pass
        
    @abstractmethod
    async def process_event(self, event: BusinessEvent) -> Dict[str, Any]:
        """Process a business event and return decision/response"""
        pass
        
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        pass
        
    async def health_check(self) -> bool:
        """Check if agent is healthy and responsive"""
        try:
            return self.state in [AgentState.ACTIVE, AgentState.BUSY]
        except Exception:
            return False
            
    def update_metrics(self, response_time: float, success: bool) -> None:
        """Update performance metrics"""
        if success:
            self.performance_metrics["requests_processed"] += 1
            self.performance_metrics["total_response_time"] += response_time
            self.performance_metrics["average_response_time"] = (
                self.performance_metrics["total_response_time"] / 
                self.performance_metrics["requests_processed"]
            )
        else:
            self.performance_metrics["requests_failed"] += 1


class MockAIAgent(AIAgentInterface):
    """Mock AI agent for testing purposes"""
    
    def __init__(self, agent_id: str, config: AgentConfig):
        super().__init__(agent_id, config)
        self.decision_rules = config.model_config.get("rules", {}) if config.model_config else {}
        
    async def initialize(self) -> None:
        """Initialize mock agent"""
        await asyncio.sleep(0.1)  # Simulate initialization time
        self.state = AgentState.ACTIVE
        self.logger.info(f"Mock agent {self.agent_id} initialized")
        
    async def process_event(self, event: BusinessEvent) -> Dict[str, Any]:
        """Process event with mock logic"""
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Simple mock decision based on event type
        if event.event_type == "fraud_detection":
            fraud_score = hash(str(event.data.get("transaction_id", ""))) % 100 / 100.0
            return {
                "decision": "reject" if fraud_score > 0.95 else "approve",
                "confidence": fraud_score,
                "reasoning": f"Mock fraud score: {fraud_score:.2f}"
            }
        
        return {"decision": "processed", "confidence": 0.8}
        
    async def cleanup(self) -> None:
        """Cleanup mock agent"""
        self.state = AgentState.STOPPED
        self.logger.info(f"Mock agent {self.agent_id} cleaned up")


class AgentPool:
    """Manages a pool of agents of the same type"""
    
    def __init__(self, agent_type: str, agent_class: Type[AIAgentInterface], pool_size: int):
        self.agent_type = agent_type
        self.agent_class = agent_class
        self.pool_size = pool_size
        self.agents: List[AIAgentInterface] = []
        self.available_agents: List[AIAgentInterface] = []
        self.busy_agents: List[AIAgentInterface] = []
        self.logger = logging.getLogger(f"pool.{agent_type}")
        
    async def initialize(self, base_config: AgentConfig) -> None:
        """Initialize all agents in the pool"""
        self.logger.info(f"Initializing pool of {self.pool_size} {self.agent_type} agents")
        
        for i in range(self.pool_size):
            agent_config = AgentConfig(
                agent_id=f"{base_config.agent_id}_{i}",
                agent_type=base_config.agent_type,
                name=f"{base_config.name}_{i}",
                model_path=base_config.model_path,
                model_config=base_config.model_config,
                max_concurrent_requests=base_config.max_concurrent_requests,
                timeout_seconds=base_config.timeout_seconds,
                retry_count=base_config.retry_count
            )
            
            agent = self.agent_class(agent_config.agent_id, agent_config)
            await agent.initialize()
            
            self.agents.append(agent)
            self.available_agents.append(agent)
            
        self.logger.info(f"Pool initialized with {len(self.agents)} agents")
        
    async def get_available_agent(self) -> Optional[AIAgentInterface]:
        """Get an available agent from the pool"""
        if self.available_agents:
            agent = self.available_agents.pop(0)
            self.busy_agents.append(agent)
            agent.state = AgentState.BUSY
            return agent
        return None
        
    async def return_agent(self, agent: AIAgentInterface) -> None:
        """Return agent to available pool"""
        if agent in self.busy_agents:
            self.busy_agents.remove(agent)
            self.available_agents.append(agent)
            agent.state = AgentState.ACTIVE
            
    async def cleanup(self) -> None:
        """Cleanup all agents in pool"""
        self.logger.info(f"Cleaning up {len(self.agents)} agents")
        for agent in self.agents:
            await agent.cleanup()
        self.agents.clear()
        self.available_agents.clear()
        self.busy_agents.clear()


class AgentManager:
    """
    Central manager for all AI agents in the simulation.
    Handles agent pools, load balancing, and communication.
    """
    
    def __init__(self, simpy_env: simpy.Environment, event_dispatcher: EventDispatcher):
        self.env = simpy_env
        self.event_dispatcher = event_dispatcher
        self.agent_pools: Dict[str, AgentPool] = {}
        self.agent_types: Dict[str, Type[AIAgentInterface]] = {
            "mock": MockAIAgent  # Default mock agent
        }
        self.logger = logging.getLogger("agent_manager")
        
        # Performance tracking
        self.total_requests = 0
        self.total_failures = 0
        self.request_queue_size = 0
        
    async def initialize(self, agent_configs: List[Dict[str, Any]]) -> None:
        """Initialize agent manager with configuration"""
        self.logger.info(f"Initializing agent manager with {len(agent_configs)} agent types")
        
        for config_dict in agent_configs:
            config = AgentConfig(
                agent_id=config_dict["agent_id"],
                agent_type=config_dict["agent_type"],
                name=config_dict["name"],
                model_path=config_dict.get("model_path"),
                model_config=config_dict.get("model_config", {}),
                max_concurrent_requests=config_dict.get("max_concurrent_requests", 10),
                timeout_seconds=config_dict.get("timeout_seconds", 30),
                retry_count=config_dict.get("retry_count", 3)
            )
            
            await self._create_agent_pool(config)
            
        # Subscribe to relevant events
        def event_callback(event: BusinessEvent):
            asyncio.create_task(self._handle_event(event))
        self.event_dispatcher.subscribe("*", event_callback)
        
        self.logger.info("Agent manager initialized successfully")
        
    async def _create_agent_pool(self, config: AgentConfig) -> None:
        """Create and initialize an agent pool"""
        agent_class = self.agent_types.get(config.agent_type, MockAIAgent)
        pool_size = config.max_concurrent_requests
        
        pool = AgentPool(config.agent_type, agent_class, pool_size)
        await pool.initialize(config)
        
        self.agent_pools[config.agent_type] = pool
        self.logger.info(f"Created pool for {config.agent_type} with {pool_size} agents")
        
    async def _handle_event(self, event: BusinessEvent) -> None:
        """Handle incoming business events"""
        if event.requires_ai_processing:
            await self._process_with_agent(event)
            
    async def _process_with_agent(self, event: BusinessEvent) -> None:
        """Process event with appropriate AI agent"""
        agent_type = event.data.get("agent_type", "mock")
        
        if agent_type not in self.agent_pools:
            self.logger.warning(f"No agent pool found for type: {agent_type}")
            return
            
        pool = self.agent_pools[agent_type]
        agent = await pool.get_available_agent()
        
        if not agent:
            self.logger.warning(f"No available agents for type: {agent_type}")
            self.total_failures += 1
            return
            
        try:
            self.total_requests += 1
            start_time = asyncio.get_event_loop().time()
            
            # Process event with agent
            result = await agent.process_event(event)
            
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            # Update metrics
            agent.update_metrics(response_time, True)
            
            # Create response event
            response_event = BusinessEvent(
                event_type=f"{event.event_type}_response",
                data={
                    "original_event_id": event.event_id,
                    "agent_response": result,
                    "processing_time": response_time,
                    "agent_id": agent.agent_id
                },
                timestamp=end_time
            )
            
            # Dispatch response
            await self.event_dispatcher.dispatch(response_event)
            
        except Exception as e:
            self.logger.error(f"Agent processing failed: {e}")
            agent.update_metrics(0, False)
            self.total_failures += 1
            
        finally:
            await pool.return_agent(agent)
            
    def register_agent_type(self, agent_type: str, agent_class: Type[AIAgentInterface]) -> None:
        """Register a new agent type"""
        self.agent_types[agent_type] = agent_class
        self.logger.info(f"Registered agent type: {agent_type}")
        
    def get_active_agent_count(self) -> int:
        """Get total number of active agents"""
        total = 0
        for pool in self.agent_pools.values():
            total += len([a for a in pool.agents if a.state == AgentState.ACTIVE])
        return total
        
    def get_agent_states(self) -> Dict[str, Any]:
        """Get current state of all agents"""
        states = {}
        for agent_type, pool in self.agent_pools.items():
            states[agent_type] = {
                "total_agents": len(pool.agents),
                "available": len(pool.available_agents),
                "busy": len(pool.busy_agents),
                "states": [agent.state.value for agent in pool.agents]
            }
        return states
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get aggregate performance metrics"""
        metrics = {
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "success_rate": (self.total_requests - self.total_failures) / max(self.total_requests, 1),
            "agent_pools": {}
        }
        
        for agent_type, pool in self.agent_pools.items():
            pool_metrics = {
                "agent_count": len(pool.agents),
                "individual_metrics": [agent.performance_metrics for agent in pool.agents]
            }
            metrics["agent_pools"][agent_type] = pool_metrics
            
        return metrics
        
    async def cleanup(self) -> None:
        """Cleanup all agent pools"""
        self.logger.info("Cleaning up agent manager")
        for pool in self.agent_pools.values():
            await pool.cleanup()
        self.agent_pools.clear()
        self.logger.info("Agent manager cleanup completed")