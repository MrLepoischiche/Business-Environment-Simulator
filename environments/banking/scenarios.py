"""
Banking Simulation Scenarios - Complete Implementation
Defines various banking scenarios and their execution logic
"""
import asyncio
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import simpy

from core.event_dispatcher import EventDispatcher, BusinessEvent, EventPriority, create_transaction_event
from .entities import (
    Customer, CustomerFactory, TransactionGenerator, 
    Merchant, Transaction, TransactionType, Channel
)


@dataclass
class ScenarioConfig:
    """Configuration for banking scenarios"""
    name: str
    duration_days: float  # Changed from int to float to allow fractional days
    customer_count: int
    transaction_rate: float  # transactions per hour per customer
    fraud_rate: float = 0.01  # percentage of fraudulent transactions
    peak_hours: Optional[List[int]] = None  # hours with increased activity
    special_events: Optional[List[Dict[str, Any]]] = None  # special events during simulation

    def __post_init__(self):
        if self.peak_hours is None:
            self.peak_hours = [9, 10, 11, 12, 13, 17, 18, 19]
        if self.special_events is None:
            self.special_events = []


class BankingScenario:
    """Base class for banking simulation scenarios"""
    
    def __init__(self, config: ScenarioConfig, env: simpy.Environment):
        self.config = config
        self.env = env
        self.logger = logging.getLogger(f"scenario.{config.name}")
        
        # Initialize entities
        self.customers: List[Customer] = []
        self.merchants: List[Merchant] = []
        self.transaction_generator: Optional[TransactionGenerator] = None
        
        # Event dispatcher reference (set during initialization)
        self.event_dispatcher = None
        
        # Statistics
        self.transactions_generated = 0
        self.fraud_transactions = 0
        
    async def initialize(self, event_dispatcher: EventDispatcher) -> None:
        """Initialize the scenario with required components"""
        self.event_dispatcher = event_dispatcher
        
        self.logger.info(f"Initializing scenario: {self.config.name}")
        
        # Create customers
        await self._create_customers()
        
        # Create merchants
        await self._create_merchants()
        
        # Initialize transaction generator
        self.transaction_generator = TransactionGenerator(self.customers, self.merchants)
        
        self.logger.info(f"Scenario initialized: {len(self.customers)} customers, {len(self.merchants)} merchants")
        
    async def _create_customers(self) -> None:
        """Create customer population"""
        self.logger.info(f"Creating {self.config.customer_count} customers")
        
        # Customer type distribution
        customer_types = []
        for _ in range(int(self.config.customer_count * 0.8)):  # 80% individual
            customer_types.append("individual")
        for _ in range(int(self.config.customer_count * 0.15)):  # 15% business
            customer_types.append("business")
        for _ in range(int(self.config.customer_count * 0.05)):  # 5% premium
            customer_types.append("premium")
        
        # Adjust if we don't have exact count
        while len(customer_types) < self.config.customer_count:
            customer_types.append("individual")
        
        random.shuffle(customer_types)
        
        # Create customers
        for i, ctype in enumerate(customer_types):
            from .entities import CustomerType
            
            if ctype == "individual":
                customer = CustomerFactory.create_customer(CustomerType.INDIVIDUAL)
            elif ctype == "business":
                customer = CustomerFactory.create_customer(CustomerType.BUSINESS)
            else:
                customer = CustomerFactory.create_customer(CustomerType.PREMIUM)
            
            self.customers.append(customer)
            
            # Small delay to prevent blocking
            if i % 100 == 0:
                await asyncio.sleep(0.01)
                
    async def _create_merchants(self) -> None:
        """Create merchant population"""
        merchant_count = min(50, max(10, self.config.customer_count // 20))  # 1 merchant per 20 customers
        
        self.logger.info(f"Creating {merchant_count} merchants")
        
        for i in range(merchant_count):
            merchant = CustomerFactory.create_merchant()
            self.merchants.append(merchant)
            
    def start_scenario_processes(self) -> None:
        """Start all scenario processes"""
        self.logger.info("Starting scenario processes")
        
        # Start transaction generation process
        self.env.process(self._transaction_generation_process())
        
        # Start special events if configured
        if self.config.special_events:
            for event in self.config.special_events:
                self.env.process(self._special_event_process(event))
                
        # Start daily cycle process
        self.env.process(self._daily_cycle_process())
        
    def _transaction_generation_process(self):
        """Main process for generating transactions"""
        while True:
            try:
                # Calculate current hour of day (simulated)
                sim_hours = self.env.now / 3600  # Convert seconds to hours
                current_hour = int(sim_hours % 24)
                
                # Adjust transaction rate based on peak hours
                rate_multiplier = 1.0
                if self.config.peak_hours and current_hour in self.config.peak_hours:
                    rate_multiplier = 2.5
                elif current_hour < 6 or current_hour > 22:  # Night hours
                    rate_multiplier = 0.2
                
                # Calculate transactions to generate this cycle
                base_rate = self.config.transaction_rate / 3600  # per second
                adjusted_rate = base_rate * rate_multiplier * len(self.customers)
                
                # Generate transactions
                if adjusted_rate > 0:
                    # Poisson distribution for realistic timing
                    next_transaction_time = random.expovariate(adjusted_rate)
                    yield self.env.timeout(next_transaction_time)
                    
                    # Generate transaction
                    yield self.env.process(self._generate_single_transaction())
                    
                else:
                    # Wait 1 minute during very low activity periods
                    yield self.env.timeout(60)
                    
            except Exception as e:
                self.logger.error(f"Error in transaction generation: {e}")
                yield self.env.timeout(60)  # Wait before retrying
                
    def _generate_single_transaction(self):
        """Generate a single transaction"""
        try:
            # Check if transaction generator is initialized
            if not self.transaction_generator:
                self.logger.warning("Transaction generator not initialized, skipping transaction generation")
                yield self.env.timeout(0.001)
                return
                
            # Decide if this should be fraudulent
            is_fraud = random.random() < self.config.fraud_rate
            
            # Select customer based on activity patterns
            customer = self._select_active_customer()
            if not customer:
                return
                
            # Generate transaction
            transaction = self.transaction_generator.generate_transaction(
                customer=customer,
                force_suspicious=is_fraud
            )
            
            # Create business event
            event = create_transaction_event(
                transaction_id=transaction.transaction_id,
                amount=transaction.amount,
                customer_id=transaction.customer_id,
                merchant=transaction.merchant.name if transaction.merchant else "N/A",
                requires_fraud_check=True
            )
            
            # Add additional transaction data
            event.data.update({
                "account_id": transaction.account_id,
                "transaction_type": transaction.transaction_type.value,
                "channel": transaction.channel.value,
                "location": transaction.location,
                "timestamp": transaction.timestamp.isoformat(),
                "metadata": transaction.metadata,
                "is_test_fraud": is_fraud  # For validation purposes
            })
            
            # Dispatch event - create coroutine and run it
            if self.event_dispatcher:
                asyncio.create_task(self.event_dispatcher.dispatch(event))
            else:
                self.logger.warning("Event dispatcher not initialized, skipping event dispatch")
            
            # Update statistics
            self.transactions_generated += 1
            if is_fraud:
                self.fraud_transactions += 1
                
            # Log periodically
            if self.transactions_generated % 100 == 0:
                self.logger.info(f"Generated {self.transactions_generated} transactions "
                               f"({self.fraud_transactions} fraudulent)")
                
        except Exception as e:
            self.logger.error(f"Error generating transaction: {e}")
            
        yield self.env.timeout(0.001)  # Small delay to prevent blocking
            
    def _select_active_customer(self) -> Optional[Customer]:
        """Select a customer based on their activity patterns"""
        # Simple selection for now - could be enhanced with more sophisticated logic
        current_hour = int((self.env.now / 3600) % 24)
        
        # Filter customers who are likely to be active at this hour
        active_customers = [
            customer for customer in self.customers
            if current_hour in customer.profile.peak_hours or random.random() < 0.1
        ]
        
        if not active_customers:
            active_customers = random.sample(self.customers, min(10, len(self.customers)))
            
        return random.choice(active_customers) if active_customers else None
        
    def _special_event_process(self, event_config: Dict[str, Any]):
        """Process for handling special events"""
        event_type = event_config.get("type", "unknown")
        trigger_time = event_config.get("trigger_time", 0)  # in simulation seconds
        duration = event_config.get("duration", 3600)  # 1 hour default
        
        # Wait until trigger time
        yield self.env.timeout(trigger_time)
        
        self.logger.info(f"Starting special event: {event_type}")
        
        if event_type == "fraud_spike":
            yield from self._fraud_spike_event(duration, event_config.get("intensity", 5.0))
        elif event_type == "system_outage":
            yield from self._system_outage_event(duration, event_config.get("affected_channels", ["online"]))
        elif event_type == "merchant_compromise":
            yield from self._merchant_compromise_event(duration, event_config.get("merchant_name", ""))
            
    def _fraud_spike_event(self, duration: float, intensity: float):
        """Simulate a fraud spike event"""
        original_fraud_rate = self.config.fraud_rate
        self.config.fraud_rate *= intensity
        
        # Create system alert event
        alert_event = BusinessEvent(
            event_type="security_alert",
            data={
                "alert_type": "fraud_spike_detected",
                "intensity": intensity,
                "estimated_duration": duration,
                "timestamp": datetime.now().isoformat()
            },
            priority=EventPriority.HIGH,
            requires_ai_processing=True
        )
        if self.event_dispatcher:
            asyncio.create_task(self.event_dispatcher.dispatch(alert_event))
        else:
            self.logger.warning("Event dispatcher not initialized, skipping event dispatch")

        yield self.env.timeout(duration)
        
        # Restore original fraud rate
        self.config.fraud_rate = original_fraud_rate
        
        self.logger.info(f"Fraud spike event ended. Fraud rate restored to {original_fraud_rate}")
        
    def _system_outage_event(self, duration: float, affected_channels: List[str]):
        """Simulate a system outage affecting specific channels"""
        # Create outage event
        outage_event = BusinessEvent(
            event_type="system_outage",
            data={
                "affected_channels": affected_channels,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "severity": "high" if len(affected_channels) > 2 else "medium"
            },
            priority=EventPriority.CRITICAL,
            requires_ai_processing=False
        )

        if self.event_dispatcher:
            asyncio.create_task(self.event_dispatcher.dispatch(outage_event))
        else:
            self.logger.warning("Event dispatcher not initialized, skipping event dispatch")

        self.logger.info(f"System outage started affecting channels: {affected_channels}")
        
        yield self.env.timeout(duration)
        
        # Recovery event
        recovery_event = BusinessEvent(
            event_type="system_recovery",
            data={
                "recovered_channels": affected_channels,
                "outage_duration": duration,
                "timestamp": datetime.now().isoformat()
            },
            priority=EventPriority.HIGH
        )

        if self.event_dispatcher:
            asyncio.create_task(self.event_dispatcher.dispatch(recovery_event))
        else:
            self.logger.warning("Event dispatcher not initialized, skipping event dispatch")

        self.logger.info(f"System recovery completed for channels: {affected_channels}")

    def _merchant_compromise_event(self, duration: float, merchant_name: Optional[str] = None):
        """Simulate a merchant compromise event"""
        # Select merchant
        if merchant_name:
            merchant = next((m for m in self.merchants if m.name == merchant_name), None)
        else:
            merchant = random.choice(self.merchants)
            
        if not merchant:
            yield self.env.timeout(0.001)  # Small delay
            return
            
        # Create compromise alert
        compromise_event = BusinessEvent(
            event_type="merchant_compromise",
            data={
                "merchant_id": merchant.merchant_id,
                "merchant_name": merchant.name,
                "category": merchant.category,
                "risk_level": "critical",
                "timestamp": datetime.now().isoformat()
            },
            priority=EventPriority.CRITICAL,
            requires_ai_processing=True
        )

        if self.event_dispatcher:
            asyncio.create_task(self.event_dispatcher.dispatch(compromise_event))
        else:
            self.logger.warning("Event dispatcher not initialized, skipping event dispatch")

        # Temporarily increase fraud rate for this merchant's transactions
        original_risk = merchant.risk_level
        merchant.risk_level = "critical"
        
        self.logger.info(f"Merchant compromise event started for {merchant.name}")
        
        yield self.env.timeout(duration)
        
        # Restore merchant risk level
        merchant.risk_level = original_risk
        
        self.logger.info(f"Merchant compromise event ended for {merchant.name}")
        
    def _daily_cycle_process(self):
        """Process that handles daily business cycles"""
        while True:
            try:
                # Wait for start of new day (24 hours in simulation)
                yield self.env.timeout(24 * 3600)
                
                # Generate daily summary event
                daily_summary = BusinessEvent(
                    event_type="daily_summary",
                    data={
                        "simulation_day": int(self.env.now / (24 * 3600)),
                        "transactions_generated": self.transactions_generated,
                        "fraud_transactions": self.fraud_transactions,
                        "fraud_rate": self.fraud_transactions / max(self.transactions_generated, 1),
                        "timestamp": datetime.now().isoformat()
                    },
                    priority=EventPriority.NORMAL
                )

                if self.event_dispatcher:
                    asyncio.create_task(self.event_dispatcher.dispatch(daily_summary))
                else:
                    self.logger.warning("Event dispatcher not initialized, skipping event dispatch")

                # Reset daily counters if needed
                # self.daily_transaction_count = 0
                
            except Exception as e:
                self.logger.error(f"Error in daily cycle: {e}")
                yield self.env.timeout(3600)  # Wait 1 hour before retrying
            
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """Get current scenario statistics"""
        return {
            "scenario_name": self.config.name,
            "customers": len(self.customers),
            "merchants": len(self.merchants),
            "transactions_generated": self.transactions_generated,
            "fraud_transactions": self.fraud_transactions,
            "fraud_rate": self.fraud_transactions / max(self.transactions_generated, 1),
            "simulation_time": self.env.now,
            "simulation_days": self.env.now / (24 * 3600)
        }


class RetailBankingScenario(BankingScenario):
    """Standard retail banking scenario"""
    
    def __init__(self, env: simpy.Environment, customer_count: int = 1000):
        config = ScenarioConfig(
            name="Retail Banking Scenario",
            duration_days=30,
            customer_count=customer_count,
            transaction_rate=2.0,  # 2 transactions per hour per customer
            fraud_rate=0.005,  # 0.5% fraud rate
            peak_hours=[9, 10, 11, 12, 13, 17, 18, 19],  # Business hours and evening
            special_events=[
                {
                    "type": "fraud_spike",
                    "trigger_time": 7 * 24 * 3600,  # Day 7
                    "duration": 4 * 3600,  # 4 hours
                    "intensity": 10.0  # 10x normal fraud rate
                }
            ]
        )
        super().__init__(config, env)


class HighVolumeScenario(BankingScenario):
    """High-volume banking scenario for stress testing"""
    
    def __init__(self, env: simpy.Environment, customer_count: int = 5000):
        config = ScenarioConfig(
            name="High Volume Banking Scenario",
            duration_days=7,
            customer_count=customer_count,
            transaction_rate=10.0,  # 10 transactions per hour per customer
            fraud_rate=0.01,  # 1% fraud rate
            peak_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            special_events=[
                {
                    "type": "system_outage",
                    "trigger_time": 2 * 24 * 3600,  # Day 2
                    "duration": 30 * 60,  # 30 minutes
                    "affected_channels": ["online", "mobile"]
                },
                {
                    "type": "merchant_compromise", 
                    "trigger_time": 4 * 24 * 3600,  # Day 4
                    "duration": 2 * 3600,  # 2 hours
                    "merchant_name": "Amazon"
                }
            ]
        )
        super().__init__(config, env)


class FraudDetectionScenario(BankingScenario):
    """Scenario focused on fraud detection testing"""
    
    def __init__(self, env: simpy.Environment, customer_count: int = 2000):
        config = ScenarioConfig(
            name="Fraud Detection Scenario",
            duration_days=14,
            customer_count=customer_count,
            transaction_rate=3.0,
            fraud_rate=0.05,  # Higher fraud rate for testing
            peak_hours=[10, 11, 12, 14, 15, 16, 17, 18],
            special_events=[
                {
                    "type": "fraud_spike",
                    "trigger_time": 3 * 24 * 3600,  # Day 3
                    "duration": 6 * 3600,  # 6 hours
                    "intensity": 20.0  # 20x normal fraud rate
                },
                {
                    "type": "fraud_spike",
                    "trigger_time": 10 * 24 * 3600,  # Day 10
                    "duration": 2 * 3600,  # 2 hours
                    "intensity": 15.0
                }
            ]
        )
        super().__init__(config, env)
        
    def _generate_single_transaction(self):
        """Override to add more sophisticated fraud patterns"""
        try:
            # Check if transaction generator is initialized
            if not self.transaction_generator:
                self.logger.warning("Transaction generator not initialized, skipping transaction generation")
                yield self.env.timeout(0.001)
                return
                
            # Enhanced fraud detection with patterns
            is_fraud = self._should_generate_fraud()
            fraud_pattern = self._select_fraud_pattern() if is_fraud else None
            
            customer = self._select_active_customer()
            if not customer:
                return
                
            transaction = self.transaction_generator.generate_transaction(
                customer=customer,
                force_suspicious=is_fraud
            )
            
            # Apply fraud pattern if applicable
            if fraud_pattern:
                self._apply_fraud_pattern(transaction, fraud_pattern)
                
            # Create enhanced event
            event = create_transaction_event(
                transaction_id=transaction.transaction_id,
                amount=transaction.amount,
                customer_id=transaction.customer_id,
                merchant=transaction.merchant.name if transaction.merchant else "N/A",
                requires_fraud_check=True
            )
            
            # Enhanced event data
            event.data.update({
                "account_id": transaction.account_id,
                "transaction_type": transaction.transaction_type.value,
                "channel": transaction.channel.value,
                "location": transaction.location,
                "timestamp": transaction.timestamp.isoformat(),
                "metadata": transaction.metadata,
                "is_test_fraud": is_fraud,
                "fraud_pattern": fraud_pattern if fraud_pattern else None,
                "agent_type": "fraud_detection"  # Specify which agent should handle this
            })
            
            if self.event_dispatcher:
                asyncio.create_task(self.event_dispatcher.dispatch(event))
            else:
                self.logger.warning("Event dispatcher not available, skipping event dispatch")

            self.transactions_generated += 1
            if is_fraud:
                self.fraud_transactions += 1
                
        except Exception as e:
            self.logger.error(f"Error in fraud detection scenario transaction generation: {e}")
        
        yield self.env.timeout(0.001)  # Small delay
            
    def _should_generate_fraud(self) -> bool:
        """Enhanced fraud decision logic"""
        base_rate = self.config.fraud_rate
        
        # Increase fraud rate during night hours
        current_hour = int((self.env.now / 3600) % 24)
        if current_hour < 6 or current_hour > 22:
            base_rate *= 2
            
        return random.random() < base_rate
        
    def _select_fraud_pattern(self) -> str:
        """Select type of fraud pattern to apply"""
        patterns = [
            "card_testing",      # Small amounts to test stolen cards
            "account_takeover",  # Unusual location/device
            "synthetic_identity", # New account with suspicious activity
            "bust_out",          # Rapid spending increase
            "velocity_abuse"     # Many transactions in short time
        ]
        
        return random.choice(patterns)
        
    def _apply_fraud_pattern(self, transaction: Transaction, pattern: str) -> None:
        """Apply specific fraud pattern to transaction"""
        if pattern == "card_testing":
            transaction.amount = round(random.uniform(1.0, 25.0), 2)  # Small test amounts
            transaction.metadata["pattern_indicators"] = ["small_amount", "unusual_merchant"]
            
        elif pattern == "account_takeover":
            transaction.metadata["pattern_indicators"] = ["new_device", "unusual_location", "unusual_time"]
            transaction.location = "Foreign Country"
            
        elif pattern == "synthetic_identity":
            transaction.metadata["pattern_indicators"] = ["new_customer", "high_initial_activity"]
            
        elif pattern == "bust_out":
            transaction.amount *= 3  # Much higher than normal
            transaction.metadata["pattern_indicators"] = ["spending_spike", "approaching_limit"]
            
        elif pattern == "velocity_abuse":
            transaction.metadata["pattern_indicators"] = ["high_frequency", "multiple_merchants", "short_timespan"]


def create_scenario(scenario_type: str, env: simpy.Environment, **kwargs) -> BankingScenario:
    """Factory function to create banking scenarios"""
    
    if scenario_type == "retail":
        return RetailBankingScenario(env, **kwargs)
    elif scenario_type == "high_volume":
        return HighVolumeScenario(env, **kwargs)
    elif scenario_type == "fraud_detection":
        return FraudDetectionScenario(env, **kwargs)
    else:
        # Default to retail
        return RetailBankingScenario(env, **kwargs)