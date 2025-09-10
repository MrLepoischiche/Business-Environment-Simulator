"""
FastAPI application for Banking Simulation API
Provides REST endpoints for simulation control and monitoring
"""
import asyncio
import logging
import sys
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

# Add parent directory to Python path to find core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from core import SimulationConfig, SimulationState, create_simulation_engine
from environments.banking import create_scenario


# Pydantic models for API
class SimulationConfigRequest(BaseModel):
    name: str = Field(..., description="Simulation name")
    duration_days: float = Field(7.0, ge=0.001, le=365.0, description="Simulation duration in days (supports fractional days)")
    customer_count: int = Field(1000, ge=100, le=10000, description="Number of customers")
    scenario_type: str = Field("retail", description="Scenario type: retail, high_volume, fraud_detection")
    fraud_rate: float = Field(0.01, ge=0.001, le=0.1, description="Fraud rate percentage")
    time_acceleration: float = Field(10.0, ge=1.0, le=1000.0, description="Time acceleration factor")


class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    message: str


class SimulationStatusResponse(BaseModel):
    simulation_id: str
    state: str
    progress: float
    current_sim_time: float
    agents_active: int
    events_processed: int
    start_time: Optional[str] = None


class MetricsResponse(BaseModel):
    simulation_id: str
    timestamp: str
    metrics: Dict[str, Any]
    fraud_metrics: Optional[Dict[str, float]] = None


# Global simulation manager
class SimulationManager:
    def __init__(self):
        self.simulations: Dict[str, Any] = {}
        self.logger = logging.getLogger("simulation_manager")
        
    async def create_simulation(self, config_request: SimulationConfigRequest) -> str:
        """Create a new simulation instance"""
        simulation_id = str(uuid.uuid4())
        
        # Create simulation config
        config = SimulationConfig(
            name=config_request.name,
            duration_days=config_request.duration_days,
            time_acceleration=config_request.time_acceleration,
            seed=42,
            environment_type="banking"
        )
        
        # Create banking simulation engine
        from environments.banking.simulation import create_banking_simulation
        engine = create_banking_simulation(config)
        
        # Store simulation info
        self.simulations[simulation_id] = {
            "engine": engine,
            "config": config_request,
            "created_at": datetime.utcnow(),
            "task": None
        }
        
        self.logger.info(f"Created simulation {simulation_id}")
        return simulation_id
        
    async def start_simulation(self, simulation_id: str) -> None:
        """Start a simulation"""
        if simulation_id not in self.simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")
            
        sim_info = self.simulations[simulation_id]
        engine = sim_info["engine"]
        config_request = sim_info["config"]
        
        # Prepare environment configuration
        environment_config = {
            "scenario_type": config_request.scenario_type,
            "scenario_params": {
                "customer_count": config_request.customer_count
            },
            "agents": [
                {
                    "agent_id": "fraud_detector_primary",
                    "agent_type": "banking_fraud",  # Custom banking fraud agent
                    "name": "Primary Fraud Detector",
                    "max_concurrent_requests": 10,
                    "model_config": {
                        "fraud_threshold": 0.8,
                        "amount_threshold": 1000.0,
                        "velocity_threshold": 5
                    }
                }
            ]
        }
        
        # Initialize and start simulation in background
        async def run_simulation():
            try:
                await engine.initialize(environment_config)
                results = await engine.run()
                sim_info["results"] = results
                sim_info["completed_at"] = datetime.utcnow()
                self.logger.info(f"Simulation {simulation_id} completed")
            except Exception as e:
                self.logger.error(f"Simulation {simulation_id} failed: {e}")
                sim_info["error"] = str(e)
                
        # Start the simulation task
        sim_info["task"] = asyncio.create_task(run_simulation())
        sim_info["started_at"] = datetime.utcnow()
        
        self.logger.info(f"Started simulation {simulation_id}")
        
    async def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get simulation status"""
        if simulation_id not in self.simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")
            
        sim_info = self.simulations[simulation_id]
        engine = sim_info["engine"]
        
        # Get status from engine
        status = engine.get_status()
        
        return {
            "simulation_id": simulation_id,
            "state": status["state"],
            "progress": status["progress"],
            "current_sim_time": status["current_sim_time"],
            "agents_active": status["agents_active"],
            "events_processed": status["events_processed"],
            "start_time": sim_info.get("started_at").isoformat() if sim_info.get("started_at") else None
        }
        
    async def get_simulation_metrics(self, simulation_id: str) -> Dict[str, Any]:
        """Get simulation metrics"""
        if simulation_id not in self.simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")
            
        sim_info = self.simulations[simulation_id]
        engine = sim_info["engine"]
        
        # Get current metrics
        metrics = engine.metrics_collector.get_current_metrics()
        
        # Get fraud metrics if available
        fraud_metrics = None
        if hasattr(engine, 'banking_metrics'):
            fraud_metrics = engine.banking_metrics.get_fraud_summary()
            
        return {
            "simulation_id": simulation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "fraud_metrics": fraud_metrics
        }
        
    async def pause_simulation(self, simulation_id: str) -> None:
        """Pause a simulation"""
        if simulation_id not in self.simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")
            
        engine = self.simulations[simulation_id]["engine"]
        await engine.pause()
        
    async def stop_simulation(self, simulation_id: str) -> None:
        """Stop a simulation"""
        if simulation_id not in self.simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")
            
        sim_info = self.simulations[simulation_id]
        engine = sim_info["engine"]
        
        await engine.stop()
        
        # Cancel background task if running
        if sim_info["task"] and not sim_info["task"].done():
            sim_info["task"].cancel()
            
    async def get_simulation_report(self, simulation_id: str) -> Dict[str, Any]:
        """Get complete simulation report"""
        if simulation_id not in self.simulations:
            raise HTTPException(status_code=404, detail="Simulation not found")
            
        sim_info = self.simulations[simulation_id]
        engine = sim_info["engine"]
        
        if hasattr(engine, 'get_banking_report'):
            return await engine.get_banking_report()
        else:
            # Fallback to basic results
            return engine.get_status()
            
    def list_simulations(self) -> List[Dict[str, Any]]:
        """List all simulations"""
        result = []
        for sim_id, sim_info in self.simulations.items():
            engine = sim_info["engine"]
            status = engine.get_status()
            
            result.append({
                "simulation_id": sim_id,
                "name": sim_info["config"].name,
                "scenario_type": sim_info["config"].scenario_type,
                "state": status["state"],
                "created_at": sim_info["created_at"].isoformat(),
                "started_at": sim_info.get("started_at").isoformat() if sim_info.get("started_at") else None
            })
            
        return result


# Global manager instance
simulation_manager = SimulationManager()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("api")
    logger.info("Banking Simulation API starting up")
    
    yield
    
    # Shutdown
    logger.info("Banking Simulation API shutting down")
    # Clean up any running simulations
    for sim_id in list(simulation_manager.simulations.keys()):
        try:
            await simulation_manager.stop_simulation(sim_id)
        except Exception as e:
            logger.error(f"Error stopping simulation {sim_id}: {e}")


# Create FastAPI app
app = FastAPI(
    title="Banking Simulation API",
    description="REST API for banking environment simulation and fraud detection testing",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# Simulation endpoints
@app.post("/api/simulation/create", response_model=SimulationResponse)
async def create_simulation(config: SimulationConfigRequest):
    """Create a new simulation"""
    try:
        simulation_id = await simulation_manager.create_simulation(config)
        return SimulationResponse(
            simulation_id=simulation_id,
            status="created",
            message="Simulation created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create simulation: {str(e)}")


@app.post("/api/simulation/{simulation_id}/start", response_model=SimulationResponse)
async def start_simulation(simulation_id: str, background_tasks: BackgroundTasks):
    """Start a simulation"""
    try:
        await simulation_manager.start_simulation(simulation_id)
        return SimulationResponse(
            simulation_id=simulation_id,
            status="started",
            message="Simulation started successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start simulation: {str(e)}")


@app.get("/api/simulation/{simulation_id}/status", response_model=SimulationStatusResponse)
async def get_simulation_status(simulation_id: str):
    """Get simulation status"""
    try:
        status = await simulation_manager.get_simulation_status(simulation_id)
        return SimulationStatusResponse(**status)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get simulation status: {str(e)}")


@app.get("/api/simulation/{simulation_id}/metrics", response_model=MetricsResponse)
async def get_simulation_metrics(simulation_id: str):
    """Get simulation metrics"""
    try:
        metrics = await simulation_manager.get_simulation_metrics(simulation_id)
        return MetricsResponse(**metrics)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get simulation metrics: {str(e)}")


@app.post("/api/simulation/{simulation_id}/pause")
async def pause_simulation(simulation_id: str):
    """Pause a simulation"""
    try:
        await simulation_manager.pause_simulation(simulation_id)
        return {"message": "Simulation paused successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pause simulation: {str(e)}")


@app.post("/api/simulation/{simulation_id}/stop")
async def stop_simulation(simulation_id: str):
    """Stop a simulation"""
    try:
        await simulation_manager.stop_simulation(simulation_id)
        return {"message": "Simulation stopped successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop simulation: {str(e)}")


@app.get("/api/simulation/{simulation_id}/report")
async def get_simulation_report(simulation_id: str):
    """Get complete simulation report"""
    try:
        report = await simulation_manager.get_simulation_report(simulation_id)
        return report
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get simulation report: {str(e)}")


@app.get("/api/simulations")
async def list_simulations():
    """List all simulations"""
    try:
        simulations = simulation_manager.list_simulations()
        return {"simulations": simulations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list simulations: {str(e)}")


# Scenario endpoints
@app.get("/api/scenarios")
async def list_scenarios():
    """List available scenarios"""
    return {
        "scenarios": [
            {
                "type": "retail",
                "name": "Retail Banking Scenario",
                "description": "Standard retail banking with normal transaction patterns"
            },
            {
                "type": "high_volume",
                "name": "High Volume Scenario", 
                "description": "High-volume banking scenario for stress testing"
            },
            {
                "type": "fraud_detection",
                "name": "Fraud Detection Scenario",
                "description": "Scenario focused on fraud detection with higher fraud rates"
            }
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),  # Par défaut localhost
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level="info"
    )
