"""
Banking Simulation Integration
Integrates banking environment with core simulation framework
"""
import asyncio
import logging
import simpy
import statistics
from collections import deque, defaultdict
from typing import Deque, DefaultDict, Dict, Any, Optional
from datetime import datetime

from core.simulation_engine import SimulationEngine, SimulationConfig
from core.event_dispatcher import BusinessEvent
from core.agent_manager import AIAgentInterface, AgentConfig
from .scenarios import create_scenario, BankingScenario
from .metrics import create_banking_metrics_collector, BankingMetricsCollector
from .entities import CustomerFactory, TransactionGenerator


class BankingSimulationEngine(SimulationEngine):
    """Enhanced simulation engine for banking environments"""
    
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
        
        # Banking-specific components
        self.banking_scenario: Optional[BankingScenario] = None
        self.banking_metrics: Optional[BankingMetricsCollector] = None
        
        # Override metrics collector with banking version
        self.metrics_collector = create_banking_metrics_collector(self.simulation_id)
        self.banking_metrics = self.metrics_collector
        
        self.logger = logging.getLogger(f"banking_simulation.{self.simulation_id}")
        
    async def initialize(self, environment_config: Dict[str, Any]) -> None:
        """Initialize banking simulation with environment configuration"""
        try:
            self.logger.info("Initializing banking simulation environment")
            
            # Initialize base simulation
            await super().initialize(environment_config)
            
            # Create banking scenario
            scenario_type = environment_config.get("scenario_type", "retail")
            scenario_params = environment_config.get("scenario_params", {})
            
            self.banking_scenario = create_scenario(
                scenario_type=scenario_type,
                env=self.env,
                **scenario_params
            )
            
            # Initialize scenario
            await self.banking_scenario.initialize(self.event_dispatcher)
            
            # Setup banking event handlers
            self._setup_banking_event_handlers()
            
            # Start scenario processes
            self.banking_scenario.start_scenario_processes()
            
            self.logger.info("Banking simulation environment initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize banking simulation: {e}")
            raise
            
    def _setup_banking_event_handlers(self) -> None:
        """Setup event handlers for banking events"""
        # Subscribe to transaction events
        self.event_dispatcher.subscribe(
            ["transaction", "fraud_detection_response", "customer_interaction", 
             "system_outage", "system_recovery", "daily_summary"],
            self._sync_handle_banking_event
        )
        
    def _sync_handle_banking_event(self, event: BusinessEvent) -> None:
        """Synchronous wrapper for handling banking events"""
        try:
            # Schedule the async handler to run in the event loop
            asyncio.create_task(self._handle_banking_event(event))
        except Exception as e:
            self.logger.error(f"Error in sync banking event handler: {e}")
        
    async def _handle_banking_event(self, event: BusinessEvent) -> None:
        """Handle banking-specific events"""
        try:
            # Process event with banking metrics collector
            if self.banking_metrics:
                await self.banking_metrics.process_transaction_event(event)
            
            # Log important events
            if event.event_type == "fraud_detection_response":
                await self._log_fraud_response(event)
            elif event.event_type == "system_outage":
                self.logger.warning(f"System outage detected: {event.data}")
            elif event.event_type == "daily_summary":
                await self._log_daily_summary(event)
                
        except Exception as e:
            self.logger.error(f"Error handling banking event {event.event_id}: {e}")
            
    async def _log_fraud_response(self, event: BusinessEvent) -> None:
        """Log fraud detection responses"""
        data = event.data
        agent_response = data.get("agent_response", {})
        decision = agent_response.get("decision", "unknown")
        confidence = agent_response.get("confidence", 0.0)
        processing_time = data.get("processing_time", 0.0)
        
        original_data = data.get("original_event_data", {})
        amount = original_data.get("amount", 0.0)
        is_test_fraud = original_data.get("is_test_fraud", False)
        
        if decision == "reject":
            self.logger.info(f"Fraud detected: ${amount:.2f} transaction rejected "
                           f"(confidence: {confidence:.2f}, time: {processing_time:.3f}s)")
            if not is_test_fraud:
                self.logger.warning("False positive: Legitimate transaction rejected")
        elif is_test_fraud:
            self.logger.warning(f"Fraud missed: ${amount:.2f} fraudulent transaction approved "
                              f"(confidence: {confidence:.2f})")
            
    async def _log_daily_summary(self, event: BusinessEvent) -> None:
        """Log daily summary information"""
        data = event.data
        day = data.get("simulation_day", 0)
        transactions = data.get("transactions_generated", 0)
        fraud_transactions = data.get("fraud_transactions", 0)
        fraud_rate = data.get("fraud_rate", 0.0)
        
        self.logger.info(f"Day {day} Summary: {transactions} transactions, "
                        f"{fraud_transactions} fraudulent ({fraud_rate:.1%})")
        
    async def get_banking_report(self) -> Dict[str, Any]:
        """Get comprehensive banking simulation report"""
        # Get base simulation results
        base_results = await self._collect_results()
        
        # Get banking-specific report
        banking_report = {}
        if self.banking_metrics:
            banking_report = await self.banking_metrics.generate_banking_report()
        
        # Get scenario statistics
        scenario_stats = (self.banking_scenario.get_scenario_statistics() 
                         if self.banking_scenario else {})
        
        # Combine all reports
        full_report = {
            **base_results,
            "banking_report": banking_report,
            "scenario_statistics": scenario_stats,
            "fraud_summary": self.banking_metrics.get_fraud_summary() if self.banking_metrics else {},
            "environment_type": "banking",
            "scenario_type": self.banking_scenario.config.name if self.banking_scenario else "unknown"
        }
        
        return full_report


class BankingFraudAgent(AIAgentInterface):
    """Enhanced fraud detection agent for banking simulation"""
    
    def __init__(self, agent_id: str, config: AgentConfig):
        super().__init__(agent_id, config)
        
        # Fraud detection parameters
        self.fraud_threshold = config.model_config.get("fraud_threshold", 0.8) if config.model_config else 0.8
        self.amount_threshold = config.model_config.get("amount_threshold", 1000.0) if config.model_config else 1000.0
        self.velocity_threshold = config.model_config.get("velocity_threshold", 5) if config.model_config else 5  # transactions per hour
        
        # Customer transaction history (simplified for simulation)
        self.customer_history = {}
        
    async def initialize(self) -> None:
        """Initialize fraud detection agent"""
        await asyncio.sleep(0.1)
        from core.agent_manager import AgentState
        self.state = AgentState.ACTIVE
        self.logger.info(f"Banking fraud agent {self.agent_id} initialized")
        
    async def process_event(self, event) -> Dict[str, Any]:
        """Process transaction event for fraud detection"""
        data = event.data
        
        # Extract transaction details
        customer_id = data.get("customer_id")
        amount = data.get("amount", 0.0)
        transaction_type = data.get("transaction_type", "unknown")
        channel = data.get("channel", "unknown")
        location = data.get("location", "")
        metadata = data.get("metadata", {})
        
        # Calculate fraud score
        fraud_score = await self._calculate_fraud_score(
            customer_id or "unknown", amount, transaction_type, channel, location, metadata
        )
        
        # Make decision
        decision = "reject" if fraud_score > self.fraud_threshold else "approve"
        
        # Generate reasoning
        reasons = self._generate_reasoning(fraud_score, amount, transaction_type, metadata)
        
        # Update customer history
        if customer_id:
            self._update_customer_history(customer_id, amount, datetime.now())
        
        return {
            "decision": decision,
            "confidence": fraud_score,
            "reasoning": reasons,
            "fraud_score": fraud_score,
            "threshold_used": self.fraud_threshold,
            "factors_considered": [
                "amount", "velocity", "location", "time", "channel", "customer_history"
            ]
        }
        
    async def _calculate_fraud_score(self, customer_id: str, amount: float, 
                                   transaction_type: str, channel: str, 
                                   location: str, metadata: Dict) -> float:
        """Calculate fraud risk score for transaction"""
        score = 0.0
        
        # Amount-based risk
        if amount > self.amount_threshold:
            score += 0.3
        if amount > 5000:
            score += 0.2
            
        # Velocity check
        velocity_score = self._check_velocity(customer_id)
        score += velocity_score * 0.25
        
        # Location risk
        if "foreign" in location.lower() or "unusual" in location.lower():
            score += 0.2
            
        # Time-based risk
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Night transactions
            score += 0.15
            
        # Channel risk
        if channel in ["atm", "online"]:
            score += 0.05
            
        # Metadata indicators
        pattern_indicators = metadata.get("pattern_indicators", [])
        if pattern_indicators:
            score += len(pattern_indicators) * 0.1
            
        # Test fraud override (for validation)
        if metadata.get("is_test_fraud", False):
            score = max(score, 0.85)  # Ensure test fraud is usually detected
            
        return min(score, 1.0)  # Cap at 1.0
        
    def _check_velocity(self, customer_id: str) -> float:
        """Check transaction velocity for customer"""
        if customer_id not in self.customer_history:
            return 0.0
            
        history = self.customer_history[customer_id]
        current_time = datetime.now()
        
        # Count transactions in last hour
        recent_transactions = [
            tx_time for tx_time in history["transaction_times"]
            if (current_time - tx_time).total_seconds() < 3600
        ]
        
        # Velocity risk score
        velocity = len(recent_transactions)
        if velocity > self.velocity_threshold:
            return min(velocity / self.velocity_threshold, 1.0)
        return 0.0
        
    def _update_customer_history(self, customer_id: str, amount: float, timestamp: datetime) -> None:
        """Update customer transaction history"""
        if customer_id not in self.customer_history:
            self.customer_history[customer_id] = {
                "transaction_times": [],
                "amounts": [],
                "total_amount": 0.0,
                "transaction_count": 0
            }
            
        history = self.customer_history[customer_id]
        history["transaction_times"].append(timestamp)
        history["amounts"].append(amount)
        history["total_amount"] += amount
        history["transaction_count"] += 1
        
        # Keep only last 100 transactions per customer
        if len(history["transaction_times"]) > 100:
            history["transaction_times"] = history["transaction_times"][-100:]
            history["amounts"] = history["amounts"][-100:]
            
    def _generate_reasoning(self, fraud_score: float, amount: float, 
                          transaction_type: str, metadata: Dict) -> str:
        """Generate human-readable reasoning for the decision"""
        reasons = []
        
        if fraud_score > 0.8:
            reasons.append("High fraud risk detected")
        elif fraud_score > 0.5:
            reasons.append("Medium fraud risk")
        else:
            reasons.append("Low fraud risk")
            
        if amount > 1000:
            reasons.append(f"Large transaction amount: ${amount:.2f}")
            
        pattern_indicators = metadata.get("pattern_indicators", [])
        if pattern_indicators:
            reasons.append(f"Suspicious patterns: {', '.join(pattern_indicators)}")
            
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            reasons.append("Transaction during unusual hours")
            
        return "; ".join(reasons)
        
    async def cleanup(self) -> None:
        """Cleanup fraud detection agent"""
        self.customer_history.clear()
        from core.agent_manager import AgentState
        self.state = AgentState.STOPPED
        self.logger.info(f"Banking fraud agent {self.agent_id} cleaned up")


# Factory functions
def create_banking_simulation(config: SimulationConfig) -> BankingSimulationEngine:
    """Create a banking simulation engine"""
    return BankingSimulationEngine(config)


def create_fraud_agent(agent_id: str, config: AgentConfig) -> BankingFraudAgent:
    """Create a banking fraud detection agent"""
    return BankingFraudAgent(agent_id, config)