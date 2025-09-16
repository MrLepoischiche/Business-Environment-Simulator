"""
Banking Environment Package
Complete banking simulation environment with entities, scenarios, and metrics
"""

from .entities import (
    Customer,
    Account,
    Transaction,
    Merchant,
    CustomerFactory,
    TransactionGenerator,
    CustomerType,
    AccountType,
    TransactionType,
    TransactionStatus,
    Channel,
    Address,
    CustomerProfile
)

from .scenarios import (
    BankingScenario,
    RetailBankingScenario,
    HighVolumeScenario,
    FraudDetectionScenario,
    ScenarioConfig,
    create_scenario
)

from .metrics import (
    BankingMetricsCollector,
    FraudMetrics,
    create_banking_metrics_collector
)

__version__ = "0.1.0"
__author__ = "Banking Environment Simulator"

__all__ = [
    # Entities
    "Customer",
    "Account", 
    "Transaction",
    "Merchant",
    "CustomerFactory",
    "TransactionGenerator",
    "CustomerType",
    "AccountType",
    "TransactionType",
    "TransactionStatus",
    "Channel",
    "Address",
    "CustomerProfile",
    
    # Scenarios
    "BankingScenario",
    "RetailBankingScenario",
    "HighVolumeScenario", 
    "FraudDetectionScenario",
    "ScenarioConfig",
    "create_scenario",
    
    # Metrics
    "BankingMetricsCollector",
    "FraudMetrics",
    "create_banking_metrics_collector"
]


# Configuration for banking environment
DEFAULT_BANKING_CONFIG = {
    "name": "Banking Environment",
    "version": "0.1.0",
    "supported_scenarios": [
        "retail",
        "high_volume", 
        "fraud_detection"
    ],
    "default_agents": [
        {
            "agent_id": "fraud_detector",
            "agent_type": "mock",
            "name": "Fraud Detection Agent",
            "max_concurrent_requests": 10
        }
    ],
    "metrics": {
        "fraud_detection_threshold": 0.8,
        "large_transaction_threshold": 1000.0,
        "high_risk_amount_threshold": 5000.0
    }
}