"""
Tests for Banking Environment
Tests banking entities, scenarios, metrics, and integration
"""
import pytest
import asyncio
import simpy
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from environments.banking import (
    Customer, Account, Transaction, Merchant,
    CustomerFactory, TransactionGenerator,
    CustomerType, AccountType, TransactionType, TransactionStatus, Channel,
    BankingScenario, RetailBankingScenario, ScenarioConfig,
    BankingMetricsCollector, FraudMetrics,
    create_scenario, create_banking_metrics_collector
)
from core import BusinessEvent, EventPriority, EventDispatcher


class TestBankingEntities:
    """Test banking entity classes"""
    
    def test_customer_creation(self):
        """Test customer entity creation"""
        from environments.banking.entities import Address, CustomerProfile
        
        address = Address(
            street="123 Main St",
            city="TestCity", 
            state="TC",
            zip_code="12345"
        )
        
        profile = CustomerProfile(
            risk_score=0.3,
            avg_monthly_transactions=50,
            avg_transaction_amount=200.0,
            preferred_channels=[Channel.ONLINE, Channel.MOBILE],
            peak_hours=[9, 12, 17]
        )
        
        customer = Customer(
            customer_id="test_123",
            first_name="John",
            last_name="Doe",
            email="john.doe@email.com",
            phone="555-1234",
            address=address,
            customer_type=CustomerType.INDIVIDUAL,
            date_of_birth=datetime.now() - timedelta(days=30*365),
            registration_date=datetime.now() - timedelta(days=365),
            profile=profile
        )
        
        assert customer.customer_id == "test_123"
        assert customer.full_name == "John Doe"
        assert customer.age == 30
        assert len(customer.accounts) == 0
        
        # Test serialization
        customer_dict = customer.to_dict()
        assert customer_dict["customer_id"] == "test_123"
        assert customer_dict["customer_type"] == "individual"
        
    def test_account_operations(self):
        """Test account operations"""
        account = Account(
            account_id="acc_123",
            customer_id="cust_456",
            account_type=AccountType.CHECKING,
            balance=1000.0,
            credit_limit=500.0
        )
        
        assert account.available_balance == 1500.0
        
        # Test debit
        assert account.can_debit(800.0)
        assert account.debit(800.0)
        assert account.balance == 200.0
        
        # Test credit
        account.credit(300.0)
        assert account.balance == 500.0
        
        # Test overdraft protection
        assert not account.can_debit(2100.0)  # Exceeds available balance
        
    def test_transaction_entity(self):
        """Test transaction entity"""
        merchant = Merchant(
            merchant_id="mer_123",
            name="Test Store",
            category="retail",
            mcc_code="5411"
        )
        
        transaction = Transaction(
            transaction_id="tx_789",
            account_id="acc_123",
            customer_id="cust_456",
            transaction_type=TransactionType.DEBIT,
            amount=250.0,
            description="Purchase at Test Store",
            merchant=merchant,
            channel=Channel.POS
        )
        
        assert transaction.status == TransactionStatus.PENDING
        
        # Test fraud marking
        transaction.mark_as_fraud(0.95, ["high_amount", "unusual_location"])
        assert transaction.is_suspicious
        assert transaction.fraud_score == 0.95
        assert transaction.status == TransactionStatus.REJECTED
        
        # Test approval
        transaction.status = TransactionStatus.PENDING
        transaction.approve()
        assert transaction.status == TransactionStatus.APPROVED
        
    def test_customer_factory(self):
        """Test customer factory functionality"""
        # Test individual customer
        customer = CustomerFactory.create_customer(CustomerType.INDIVIDUAL)
        
        assert customer.customer_id is not None
        assert customer.customer_type == CustomerType.INDIVIDUAL
        assert len(customer.accounts) >= 1
        assert customer.profile.risk_score >= 0.0
        
        # Test business customer
        business_customer = CustomerFactory.create_customer(CustomerType.BUSINESS)
        
        assert business_customer.customer_type == CustomerType.BUSINESS
        assert business_customer.profile.avg_monthly_transactions > customer.profile.avg_monthly_transactions
        
        # Test premium customer
        premium_customer = CustomerFactory.create_customer(CustomerType.PREMIUM)
        
        assert premium_customer.customer_type == CustomerType.PREMIUM
        assert premium_customer.profile.risk_score <= 0.3  # Lower risk
        
    def test_merchant_factory(self):
        """Test merchant creation"""
        merchant = CustomerFactory.create_merchant()
        
        assert merchant.merchant_id is not None
        assert merchant.name is not None
        assert merchant.category is not None
        assert merchant.mcc_code is not None
        assert merchant.risk_level in ["low", "medium", "high"]
        
    def test_transaction_generator(self):
        """Test transaction generation"""
        # Create test data
        customers = [CustomerFactory.create_customer() for _ in range(3)]
        merchants = [CustomerFactory.create_merchant() for _ in range(2)]
        
        generator = TransactionGenerator(customers, merchants)
        
        # Generate normal transaction
        transaction = generator.generate_transaction()
        
        assert transaction.transaction_id is not None
        assert transaction.customer_id in [c.customer_id for c in customers]
        assert transaction.amount > 0
        assert transaction.merchant is not None
        
        # Generate suspicious transaction
        customer = customers[0]
        suspicious_tx = generator.generate_transaction(customer, force_suspicious=True)
        
        assert suspicious_tx.customer_id == customer.customer_id
        assert suspicious_tx.amount >= customer.profile.avg_transaction_amount * 5
        

class TestBankingScenarios:
    """Test banking scenario implementations"""
    
    @pytest.fixture
    def env(self):
        return simpy.Environment()
    
    @pytest.fixture
    def event_dispatcher(self, env):
        return EventDispatcher(env)
    
    def test_scenario_config(self):
        """Test scenario configuration"""
        config = ScenarioConfig(
            name="Test Scenario",
            duration_days=5,
            customer_count=500,
            transaction_rate=1.5,
            fraud_rate=0.02,
            peak_hours=[9, 12, 17],
            special_events=[
                {
                    "type": "fraud_spike",
                    "trigger_time": 24 * 3600,  # Day 1
                    "duration": 3600,  # 1 hour
                    "intensity": 5.0
                }
            ]
        )
        
        assert config.name == "Test Scenario"
        assert config.customer_count == 500
        assert config.fraud_rate == 0.02
        assert len(config.special_events or []) == 1
        
    @pytest.mark.asyncio
    async def test_banking_scenario_initialization(self, env, event_dispatcher):
        """Test banking scenario initialization"""
        config = ScenarioConfig(
            name="Test Banking Scenario",
            duration_days=1,
            customer_count=100,
            transaction_rate=2.0
        )
        
        scenario = BankingScenario(config, env)
        await scenario.initialize(event_dispatcher)
        
        assert len(scenario.customers) == 100
        assert len(scenario.merchants) > 0
        assert scenario.transaction_generator is not None
        
    @pytest.mark.asyncio
    async def test_retail_banking_scenario(self, env):
        """Test retail banking scenario"""
        scenario = RetailBankingScenario(env, customer_count=50)
        
        assert scenario.config.name == "Retail Banking Scenario"
        assert scenario.config.customer_count == 50
        assert scenario.config.transaction_rate == 2.0
        assert len(scenario.config.special_events or []) > 0
        
    def test_scenario_factory(self, env):
        """Test scenario factory function"""
        # Test retail scenario
        retail = create_scenario("retail", env, customer_count=200)
        assert isinstance(retail, RetailBankingScenario)
        assert retail.config.customer_count == 200
        
        # Test high volume scenario
        high_vol = create_scenario("high_volume", env, customer_count=1000)
        assert high_vol.__class__.__name__ == "HighVolumeScenario"
        
        # Test fraud detection scenario
        fraud = create_scenario("fraud_detection", env, customer_count=500)
        assert fraud.__class__.__name__ == "FraudDetectionScenario"
        
        # Test default scenario
        default = create_scenario("unknown", env)
        assert isinstance(default, RetailBankingScenario)


class TestBankingMetrics:
    """Test banking metrics collector"""
    
    @pytest.fixture
    def collector(self):
        return create_banking_metrics_collector("test_banking_sim")
    
    @pytest.mark.asyncio
    async def test_banking_metrics_initialization(self, collector):
        """Test banking metrics collector initialization"""
        await collector.initialize()
        
        # Check banking-specific metrics exist
        assert "transaction_volume" in collector.metrics
        assert "fraud_detection_precision" in collector.metrics
        assert "false_positive_rate" in collector.metrics
        assert "customer_satisfaction_score" in collector.metrics
        
    def test_fraud_metrics_tracking(self, collector):
        """Test fraud detection metrics tracking"""
        # Initially all zeros
        assert collector.fraud_metrics.true_positives == 0
        assert collector.fraud_metrics.precision == 0.0
        
        # Simulate fraud detection results
        collector.fraud_metrics.true_positives = 85
        collector.fraud_metrics.false_positives = 15
        collector.fraud_metrics.true_negatives = 900
        collector.fraud_metrics.false_negatives = 10
        
        # Check calculated metrics
        assert collector.fraud_metrics.precision == 0.85  # 85/(85+15)
        assert collector.fraud_metrics.recall == 85/95  # 85/(85+10)
        assert collector.fraud_metrics.accuracy == 985/1010  # (85+900)/1010
        assert collector.fraud_metrics.f1_score > 0.8
        
    @pytest.mark.asyncio
    async def test_transaction_event_processing(self, collector):
        """Test processing transaction events"""
        await collector.initialize()
        
        # Create transaction event
        transaction_event = BusinessEvent(
            event_type="transaction",
            data={
                "amount": 1500.0,
                "transaction_type": "debit",
                "channel": "online",
                "customer_id": "test_customer"
            }
        )
        
        await collector.process_transaction_event(transaction_event)
        
        # Check metrics were updated
        assert collector.metrics["transactions_processed"].get_current_value() == 1
        assert collector.metrics["large_transaction_count"].get_current_value() == 1
        assert len(collector.transaction_amounts) == 1
        
    @pytest.mark.asyncio
    async def test_fraud_response_processing(self, collector):
        """Test processing fraud detection responses"""
        await collector.initialize()
        
        # Simulate fraud detection response
        fraud_response_event = BusinessEvent(
            event_type="fraud_detection_response",
            data={
                "agent_response": {
                    "decision": "reject",
                    "confidence": 0.92
                },
                "processing_time": 0.045,
                "original_event_data": {
                    "amount": 2000.0,
                    "is_test_fraud": True  # This was actually fraud
                }
            }
        )
        
        await collector.process_fraud_response(fraud_response_event)
        
        # Check fraud metrics were updated correctly (true positive)
        assert collector.fraud_metrics.true_positives == 1
        assert collector.fraud_metrics.false_positives == 0
        
        # Test false positive case
        false_positive_event = BusinessEvent(
            event_type="fraud_detection_response",
            data={
                "agent_response": {
                    "decision": "reject",
                    "confidence": 0.88
                },
                "processing_time": 0.032,
                "original_event_data": {
                    "amount": 500.0,
                    "is_test_fraud": False  # This was legitimate
                }
            }
        )
        
        await collector.process_fraud_response(false_positive_event)
        
        # Check false positive was recorded
        assert collector.fraud_metrics.true_positives == 1
        assert collector.fraud_metrics.false_positives == 1
        
    @pytest.mark.asyncio
    async def test_banking_report_generation(self, collector):
        """Test comprehensive banking report generation"""
        await collector.initialize()
        
        # Add some sample data
        collector.fraud_metrics.true_positives = 45
        collector.fraud_metrics.false_positives = 5
        collector.fraud_metrics.true_negatives = 950
        collector.fraud_metrics.false_negatives = 8
        
        collector.transaction_amounts.extend([100, 250, 500, 1200, 75, 300])
        collector.channel_metrics.update({"online": 150, "mobile": 80, "atm": 30})
        
        # Generate report
        report = await collector.generate_banking_report()
        
        assert "fraud_detection" in report
        assert "transaction_analysis" in report
        assert "customer_insights" in report
        assert "operational_summary" in report
        
        # Check fraud detection section
        fraud_section = report["fraud_detection"]
        assert fraud_section["precision"] == 0.9  # 45/(45+5)
        assert fraud_section["recall"] > 0.8
        assert fraud_section["true_positives"] == 45
        
        # Check transaction analysis
        tx_analysis = report["transaction_analysis"]
        assert tx_analysis["transaction_count"] == 6
        assert tx_analysis["total_amount"] == sum([100, 250, 500, 1200, 75, 300])
        assert "channel_distribution" in tx_analysis
        
    def test_fraud_summary(self, collector):
        """Test fraud detection summary"""
        collector.fraud_metrics.true_positives = 92
        collector.fraud_metrics.false_positives = 8
        collector.fraud_metrics.true_negatives = 900
        collector.fraud_metrics.false_negatives = 12
        
        summary = collector.get_fraud_summary()
        
        assert "precision" in summary
        assert "recall" in summary
        assert "f1_score" in summary
        assert "accuracy" in summary
        assert summary["correctly_detected"] == 92
        assert summary["missed_fraud"] == 12
        assert summary["false_alarms"] == 8


class TestBankingIntegration:
    """Integration tests for complete banking simulation"""
    
    @pytest.mark.asyncio
    async def test_complete_simulation_flow(self):
        """Test complete banking simulation flow"""
        from environments.banking.simulation import create_banking_simulation
        from core import SimulationConfig
        
        config = SimulationConfig(
            name="Integration Test Simulation",
            duration_days=0.01,  # Very short
            time_acceleration=100.0,
            environment_type="banking"
        )
        
        engine = create_banking_simulation(config)
        
        environment_config = {
            "scenario_type": "fraud_detection",
            "scenario_params": {"customer_count": 20},
            "agents": [
                {
                    "agent_id": "integration_test_agent",
                    "agent_type": "mock",
                    "name": "Integration Test Agent",
                    "max_concurrent_requests": 3,
                    "model_config": {"fraud_threshold": 0.8}
                }
            ]
        }
        
        # Initialize and run
        await engine.initialize(environment_config)
        results = await engine.run()
        
        # Verify results structure
        assert "banking_report" in results
        assert "fraud_summary" in results
        assert "scenario_statistics" in results
        
        banking_report = results["banking_report"]
        assert "fraud_detection" in banking_report
        assert "transaction_analysis" in banking_report
        
    @pytest.mark.asyncio
    async def test_scenario_event_generation(self):
        """Test that scenarios generate realistic events"""
        from environments.banking.scenarios import FraudDetectionScenario
        
        env = simpy.Environment()
        event_dispatcher = EventDispatcher(env)
        
        scenario = FraudDetectionScenario(env, customer_count=10)
        await scenario.initialize(event_dispatcher)
        
        # Track generated events
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
            
        event_dispatcher.subscribe("transaction", event_handler)
        
        # Start scenario processes
        scenario.start_scenario_processes()
        
        # Run simulation for a short time
        env.run(until=100)  # 100 time units
        
        # Should have generated some events
        assert len(events_received) > 0
        
        # Check event structure
        for event in events_received:
            assert event.event_type == "transaction"
            assert "amount" in event.data
            assert "customer_id" in event.data
            assert "transaction_id" in event.data
            
    def test_metrics_integration_with_events(self):
        """Test that metrics properly integrate with event processing"""
        collector = create_banking_metrics_collector("integration_test")
        
        # Process various banking events
        events = [
            BusinessEvent("transaction", {
                "amount": 100.0, "transaction_type": "debit", 
                "channel": "online", "customer_id": "cust_1"
            }),
            BusinessEvent("fraud_detection_response", {
                "agent_response": {"decision": "approve", "confidence": 0.3},
                "processing_time": 0.025,
                "original_event_data": {"amount": 100.0, "is_test_fraud": False}
            }),
            BusinessEvent("customer_interaction", {
                "customer_id": "cust_1", "interaction_type": "inquiry"
            })
        ]
        
        # Process events
        for event in events:
            asyncio.run(collector.process_transaction_event(event))
            
        # Verify metrics were updated
        transactions_metric = collector.metrics.get("transactions_processed")
        if transactions_metric:
            assert getattr(transactions_metric, 'value', 0) >= 1
        assert collector.fraud_metrics.true_negatives >= 1  # Legitimate transaction approved
        
    @pytest.mark.asyncio
    async def test_error_handling_in_simulation(self):
        """Test error handling in banking simulation"""
        from environments.banking.simulation import create_banking_simulation
        from core import SimulationConfig
        
        config = SimulationConfig(
            name="Error Test Simulation",
            duration_days=0.001,
            environment_type="banking"
        )
        
        engine = create_banking_simulation(config)
        
        # Test with invalid configuration
        invalid_config = {
            "scenario_type": "invalid_scenario",  # Invalid scenario type
            "scenario_params": {"customer_count": -1},  # Invalid parameter
            "agents": []
        }
        
        # Should handle gracefully or raise appropriate exception
        try:
            await engine.initialize(invalid_config)
            # If it doesn't raise an exception, it should fallback to defaults
            assert engine.banking_scenario is not None
        except Exception as e:
            # Should be a meaningful error message
            assert isinstance(e, (ValueError, TypeError))


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "--cov=environments.banking",
        "--cov-report=html",
        "--cov-report=term"
    ])