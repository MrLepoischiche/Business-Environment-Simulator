"""
Core Simulation Engine for Business Environment Simulator
Handles discrete event simulation with agent integration
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import simpy
from uuid import uuid4

from .event_dispatcher import EventDispatcher, BusinessEvent
from .agent_manager import AgentManager
from .metrics_collector import MetricsCollector


class SimulationState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimulationConfig:
    """Configuration for simulation runs"""
    name: str
    duration_days: float  # Changed from int to float to allow fractional days
    time_acceleration: float = 1.0  # Real-time multiplier
    seed: Optional[int] = None
    max_events_per_second: int = 1000
    checkpoint_interval: int = 3600  # seconds
    environment_type: str = "banking"
    

class SimulationEngine:
    """
    Main simulation engine using SimPy for discrete event simulation.
    Coordinates agents, environments, and event processing.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.simulation_id = str(uuid4())
        self.state = SimulationState.IDLE
        
        # SimPy environment
        self.env = simpy.Environment()
        
        # Core components
        self.event_dispatcher = EventDispatcher(self.env)
        self.agent_manager = AgentManager(self.env, self.event_dispatcher)
        self.metrics_collector = MetricsCollector(self.simulation_id)
        
        # Runtime state
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.current_sim_time: float = 0
        self.checkpoints: List[Dict] = []
        
        # Callbacks
        self.on_event_callbacks: List[Callable] = []
        self.on_checkpoint_callbacks: List[Callable] = []
        
        # Setup logging
        self.logger = logging.getLogger(f"simulation.{self.simulation_id}")
        
    async def initialize(self, environment_config: Dict[str, Any]) -> None:
        """Initialize simulation with environment configuration"""
        try:
            self.logger.info(f"Initializing simulation: {self.config.name}")
            
            # Initialize components
            await self.agent_manager.initialize(environment_config.get("agents", []))
            await self.metrics_collector.initialize()
            
            # Setup environment-specific processes
            self._setup_environment_processes(environment_config)
            
            # Setup checkpointing
            if self.config.checkpoint_interval > 0:
                self.env.process(self._checkpoint_process())
                
            self.state = SimulationState.IDLE
            self.logger.info("Simulation initialized successfully")
            
        except Exception as e:
            self.state = SimulationState.ERROR
            self.logger.error(f"Failed to initialize simulation: {e}")
            raise
            
    def _setup_environment_processes(self, config: Dict[str, Any]) -> None:
        """Setup environment-specific background processes"""
        # This will be extended by specific environments
        pass
        
    async def run(self) -> Dict[str, Any]:
        """Run the complete simulation"""
        try:
            self.state = SimulationState.RUNNING
            self.start_time = datetime.utcnow()
            
            self.logger.info(f"Starting simulation run for {self.config.duration_days} days")
            
            # Convert days to simulation time units
            sim_duration = self.config.duration_days * 24 * 3600
            
            # Run simulation
            await self._run_simulation(sim_duration)
            
            self.end_time = datetime.utcnow()
            self.state = SimulationState.COMPLETED
            
            # Collect final results
            results = await self._collect_results()
            
            self.logger.info(f"Simulation completed in {self.end_time - self.start_time}")
            return results
            
        except Exception as e:
            self.state = SimulationState.ERROR
            self.logger.error(f"Simulation failed: {e}")
            raise
            
    async def _run_simulation(self, duration: float) -> None:
        """Execute the SimPy simulation"""
        # Setup simulation end condition
        def simulation_end():
            yield self.env.timeout(duration)
            
        # Start the simulation process
        self.env.process(simulation_end())
        
        # Run until completion or error
        try:
            self.env.run()
        except Exception as e:
            self.logger.error(f"SimPy simulation error: {e}")
            raise
            
    async def pause(self) -> None:
        """Pause the running simulation"""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            self.logger.info("Simulation paused")
            
    async def resume(self) -> None:
        """Resume paused simulation"""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            self.logger.info("Simulation resumed")
            
    async def stop(self) -> None:
        """Stop the simulation gracefully"""
        self.state = SimulationState.COMPLETED
        await self.agent_manager.cleanup()
        self.logger.info("Simulation stopped")
        
    def _checkpoint_process(self):
        """Background process for creating checkpoints"""
        while True:
            yield self.env.timeout(self.config.checkpoint_interval)
            
            if self.state == SimulationState.RUNNING:
                checkpoint = self._create_checkpoint()
                self.checkpoints.append(checkpoint)
                
                # Trigger callbacks
                for callback in self.on_checkpoint_callbacks:
                    try:
                        callback(checkpoint)
                    except Exception as e:
                        self.logger.warning(f"Checkpoint callback failed: {e}")
                        
    def _create_checkpoint(self) -> Dict[str, Any]:
        """Create a simulation checkpoint"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "sim_time": self.env.now,
            "metrics": self.metrics_collector.get_current_metrics(),
            "agent_states": self.agent_manager.get_agent_states(),
            "event_count": self.event_dispatcher.get_event_count()
        }
        
    async def _collect_results(self) -> Dict[str, Any]:
        """Collect final simulation results"""
        return {
            "simulation_id": self.simulation_id,
            "config": self.config.__dict__,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
            "final_sim_time": self.env.now,
            "state": self.state.value,
            "metrics": await self.metrics_collector.get_final_metrics(),
            "agent_performance": self.agent_manager.get_performance_metrics(),
            "checkpoints": self.checkpoints,
            "event_summary": self.event_dispatcher.get_summary()
        }
        
    def add_event_callback(self, callback: Callable[[BusinessEvent], None]) -> None:
        """Add callback for event processing"""
        self.on_event_callbacks.append(callback)
        self.event_dispatcher.add_callback(callback)
        
    def add_checkpoint_callback(self, callback: Callable[[Dict], None]) -> None:
        """Add callback for checkpoint creation"""
        self.on_checkpoint_callbacks.append(callback)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        return {
            "simulation_id": self.simulation_id,
            "state": self.state.value,
            "current_sim_time": self.env.now if hasattr(self.env, 'now') else 0,
            "progress": min(1.0, self.env.now / (self.config.duration_days * 24 * 3600)) if hasattr(self.env, 'now') else 0,
            "agents_active": self.agent_manager.get_active_agent_count(),
            "events_processed": self.event_dispatcher.get_event_count(),
            "metrics_count": len(self.metrics_collector.get_current_metrics())
        }


# Factory function for creating simulation engines
def create_simulation_engine(config: SimulationConfig) -> SimulationEngine:
    """Factory function to create properly configured simulation engine"""
    return SimulationEngine(config)