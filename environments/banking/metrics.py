"""
Banking-specific metrics and KPIs
Defines business metrics relevant to banking simulations
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque

from core.metrics_collector import MetricsCollector, MetricDefinition, MetricType
from core.event_dispatcher import BusinessEvent


@dataclass
class FraudMetrics:
    """Fraud detection performance metrics"""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    @property
    def precision(self) -> float:
        """Fraud detection precision"""
        total_positive = self.true_positives + self.false_positives
        return self.true_positives / total_positive if total_positive > 0 else 0.0
    
    @property
    def recall(self) -> float:
        """Fraud detection recall"""
        total_actual_positive = self.true_positives + self.false_negatives
        return self.true_positives / total_actual_positive if total_actual_positive > 0 else 0.0
    
    @property
    def f1_score(self) -> float:
        """F1 score for fraud detection"""
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy"""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        correct = self.true_positives + self.true_negatives
        return correct / total if total > 0 else 0.0


class BankingMetricsCollector(MetricsCollector):
    """Enhanced metrics collector for banking simulations"""
    
    def __init__(self, simulation_id: str):
        super().__init__(simulation_id)
        
        # Banking-specific state
        self.fraud_metrics = FraudMetrics()
        self.transaction_amounts = deque(maxlen=10000)
        self.customer_metrics = defaultdict(dict)
        self.channel_metrics = defaultdict(int)
        self.hourly_transaction_counts = defaultdict(int)
        
        # Register banking-specific metrics
        self._register_banking_metrics()
        
        self.logger = logging.getLogger(f"banking_metrics.{simulation_id}")
        
    def _register_banking_metrics(self) -> None:
        """Register banking-specific metrics"""
        banking_metrics = [
            # Transaction metrics
            MetricDefinition("transaction_volume", MetricType.COUNTER, "Total transaction volume", "currency"),
            MetricDefinition("transaction_count_by_type", MetricType.COUNTER, "Transactions by type"),
            MetricDefinition("transaction_count_by_channel", MetricType.COUNTER, "Transactions by channel"),
            MetricDefinition("avg_transaction_amount", MetricType.GAUGE, "Average transaction amount", "currency"),
            MetricDefinition("large_transaction_count", MetricType.COUNTER, "Transactions over $1000"),
            
            # Fraud detection metrics
            MetricDefinition("fraud_detection_precision", MetricType.GAUGE, "Fraud detection precision", "percentage"),
            MetricDefinition("fraud_detection_recall", MetricType.GAUGE, "Fraud detection recall", "percentage"),
            MetricDefinition("fraud_detection_f1", MetricType.GAUGE, "Fraud detection F1 score", "percentage"),
            MetricDefinition("fraud_detection_accuracy", MetricType.GAUGE, "Fraud detection accuracy", "percentage"),
            MetricDefinition("false_positive_rate", MetricType.GAUGE, "False positive rate", "percentage"),
            MetricDefinition("fraud_blocked_amount", MetricType.COUNTER, "Amount blocked due to fraud", "currency"),
            
            # Customer metrics
            MetricDefinition("active_customers", MetricType.GAUGE, "Number of active customers"),
            MetricDefinition("customer_transaction_frequency", MetricType.HISTOGRAM, "Customer transaction frequency", "transactions/day"),
            MetricDefinition("new_customers", MetricType.COUNTER, "New customer registrations"),
            
            # Operational metrics
            MetricDefinition("system_availability", MetricType.GAUGE, "System availability", "percentage"),
            MetricDefinition("channel_uptime", MetricType.GAUGE, "Channel uptime", "percentage"),
            MetricDefinition("peak_hour_load", MetricType.GAUGE, "Peak hour transaction load", "transactions/hour"),
            MetricDefinition("fraud_investigation_queue", MetricType.GAUGE, "Fraud cases awaiting investigation"),
            
            # Risk metrics
            MetricDefinition("portfolio_risk_score", MetricType.GAUGE, "Overall portfolio risk score", "score"),
            MetricDefinition("high_risk_transactions", MetricType.COUNTER, "High risk transactions"),
            MetricDefinition("compliance_violations", MetricType.COUNTER, "Compliance violations detected"),
            
            # Performance metrics
            MetricDefinition("fraud_detection_latency", MetricType.TIMER, "Fraud detection response time", "seconds"),
            MetricDefinition("transaction_processing_time", MetricType.TIMER, "Transaction processing time", "seconds"),
            MetricDefinition("customer_satisfaction_score", MetricType.GAUGE, "Customer satisfaction", "score")
        ]
        
        for definition in banking_metrics:
            self.register_metric(definition)
            
    async def process_transaction_event(self, event: BusinessEvent) -> None:
        """Process transaction-related events for metrics"""
        if event.event_type == "transaction":
            await self._process_transaction_metrics(event)
        elif event.event_type == "fraud_detection_response":
            await self._process_fraud_response(event)
        elif event.event_type.startswith("customer_"):
            await self._process_customer_metrics(event)
        elif event.event_type.startswith("system_"):
            await self._process_system_metrics(event)
            
    async def _process_transaction_metrics(self, event: BusinessEvent) -> None:
        """Process transaction metrics"""
        data = event.data
        amount = data.get("amount", 0.0)
        tx_type = data.get("transaction_type", "unknown")
        channel = data.get("channel", "unknown")
        customer_id = data.get("customer_id")
        
        # Basic transaction metrics
        self.increment_counter("transactions_processed")
        self.observe_histogram("transaction_amount", amount)
        self.increment_counter("transaction_volume", amount)
        
        # Transaction type metrics
        self.increment_counter("transaction_count_by_type", labels={"type": tx_type})
        
        # Channel metrics
        self.increment_counter("transaction_count_by_channel", labels={"channel": channel})
        self.channel_metrics[channel] += 1
        
        # Large transaction tracking
        if amount > 1000:
            self.increment_counter("large_transaction_count")
            
        # Track amounts for average calculation
        self.transaction_amounts.append(amount)
        if len(self.transaction_amounts) >= 100:  # Update average every 100 transactions
            avg_amount = sum(self.transaction_amounts) / len(self.transaction_amounts)
            self.set_gauge("avg_transaction_amount", avg_amount)
            
        # Customer activity tracking
        if customer_id:
            self.customer_metrics[customer_id]["last_transaction"] = datetime.now()
            self.customer_metrics[customer_id]["transaction_count"] = (
                self.customer_metrics[customer_id].get("transaction_count", 0) + 1
            )
            
        # Hourly transaction tracking
        current_hour = datetime.now().hour
        self.hourly_transaction_counts[current_hour] += 1
        
        # Update peak hour metrics
        max_hourly = max(self.hourly_transaction_counts.values()) if self.hourly_transaction_counts else 0
        self.set_gauge("peak_hour_load", max_hourly)
        
    async def _process_fraud_response(self, event: BusinessEvent) -> None:
        """Process fraud detection response metrics"""
        data = event.data
        agent_response = data.get("agent_response", {})
        original_event_data = data.get("original_event_data", {})
        
        decision = agent_response.get("decision", "unknown")
        confidence = agent_response.get("confidence", 0.0)
        processing_time = data.get("processing_time", 0.0)
        
        # Check if this was a test fraud transaction
        is_actual_fraud = original_event_data.get("is_test_fraud", False)
        
        # Update fraud metrics based on decision and actual fraud status
        if is_actual_fraud and decision == "reject":
            self.fraud_metrics.true_positives += 1
        elif is_actual_fraud and decision == "approve":
            self.fraud_metrics.false_negatives += 1
        elif not is_actual_fraud and decision == "reject":
            self.fraud_metrics.false_positives += 1
        elif not is_actual_fraud and decision == "approve":
            self.fraud_metrics.true_negatives += 1
            
        # Update fraud detection performance metrics
        self.set_gauge("fraud_detection_precision", self.fraud_metrics.precision * 100)
        self.set_gauge("fraud_detection_recall", self.fraud_metrics.recall * 100)
        self.set_gauge("fraud_detection_f1", self.fraud_metrics.f1_score * 100)
        self.set_gauge("fraud_detection_accuracy", self.fraud_metrics.accuracy * 100)
        
        # False positive rate
        total_negative = self.fraud_metrics.true_negatives + self.fraud_metrics.false_positives
        fpr = (self.fraud_metrics.false_positives / total_negative * 100) if total_negative > 0 else 0
        self.set_gauge("false_positive_rate", fpr)
        
        # Track blocked amount for fraud
        if decision == "reject":
            amount = original_event_data.get("amount", 0.0)
            self.increment_counter("fraud_blocked_amount", amount)
            
        # Processing time
        self.observe_histogram("fraud_detection_latency", processing_time)
        
        # Log significant events
        if is_actual_fraud and decision == "approve":
            self.logger.warning(f"False negative: Fraud transaction approved (confidence: {confidence})")
        elif not is_actual_fraud and decision == "reject":
            self.logger.warning(f"False positive: Legitimate transaction rejected (confidence: {confidence})")
            
    async def _process_customer_metrics(self, event: BusinessEvent) -> None:
        """Process customer-related metrics"""
        if event.event_type == "customer_registration":
            self.increment_counter("new_customers")
        elif event.event_type == "customer_interaction":
            data = event.data
            interaction_type = data.get("interaction_type", "unknown")
            
            # Track customer satisfaction based on interaction type
            satisfaction_scores = {
                "complaint": 2.0,
                "inquiry": 3.5,
                "support": 3.0,
                "compliment": 5.0
            }
            
            score = satisfaction_scores.get(interaction_type, 3.0)
            self.observe_histogram("customer_satisfaction_score", score)
            
        # Update active customers count
        active_count = len([
            cid for cid, metrics in self.customer_metrics.items()
            if metrics.get("last_transaction") and 
            (datetime.now() - metrics["last_transaction"]).days < 30
        ])
        self.set_gauge("active_customers", active_count)
        
    async def _process_system_metrics(self, event: BusinessEvent) -> None:
        """Process system-related metrics"""
        if event.event_type == "system_outage":
            data = event.data
            affected_channels = data.get("affected_channels", [])
            
            # Reduce system availability
            current_availability = 100.0  # Default value
            if "system_availability" in self.metrics:
                metric = self.metrics["system_availability"]
                current_availability = getattr(metric, 'value', 100.0)
            reduction = len(affected_channels) * 10  # 10% per affected channel
            new_availability = max(0, current_availability - reduction)
            self.set_gauge("system_availability", new_availability)
            
        elif event.event_type == "system_recovery":
            # Restore full availability
            self.set_gauge("system_availability", 100.0)
            
    async def generate_banking_report(self) -> Dict[str, Any]:
        """Generate comprehensive banking metrics report"""
        current_time = datetime.now()
        
        # Get base metrics
        base_report = await self.get_final_metrics()
        
        # Add banking-specific analysis
        banking_report = {
            "report_timestamp": current_time.isoformat(),
            "simulation_id": self.simulation_id,
            "base_metrics": base_report,
            
            # Fraud detection performance
            "fraud_detection": {
                "precision": self.fraud_metrics.precision,
                "recall": self.fraud_metrics.recall,
                "f1_score": self.fraud_metrics.f1_score,
                "accuracy": self.fraud_metrics.accuracy,
                "true_positives": self.fraud_metrics.true_positives,
                "false_positives": self.fraud_metrics.false_positives,
                "true_negatives": self.fraud_metrics.true_negatives,
                "false_negatives": self.fraud_metrics.false_negatives
            },
            
            # Transaction analysis
            "transaction_analysis": {
                "total_amount": sum(self.transaction_amounts),
                "average_amount": sum(self.transaction_amounts) / len(self.transaction_amounts) if self.transaction_amounts else 0,
                "transaction_count": len(self.transaction_amounts),
                "channel_distribution": dict(self.channel_metrics),
                "hourly_distribution": dict(self.hourly_transaction_counts)
            },
            
            # Customer insights
            "customer_insights": {
                "total_customers": len(self.customer_metrics),
                "active_customers": len([
                    cid for cid, metrics in self.customer_metrics.items()
                    if metrics.get("last_transaction") and 
                    (current_time - metrics["last_transaction"]).days < 7
                ]),
                "avg_transactions_per_customer": sum(
                    metrics.get("transaction_count", 0) 
                    for metrics in self.customer_metrics.values()
                ) / max(len(self.customer_metrics), 1)
            },
            
            # Operational summary
            "operational_summary": {
                "system_availability": getattr(self.metrics.get("system_availability"), 'value', 100) if "system_availability" in self.metrics else 100,
                "peak_transaction_hour": max(self.hourly_transaction_counts.items(), key=lambda x: x[1])[0] if self.hourly_transaction_counts else None,
                "busiest_channel": max(self.channel_metrics.items(), key=lambda x: x[1])[0] if self.channel_metrics else None
            }
        }
        
        return banking_report
        
    def reset_fraud_metrics(self) -> None:
        """Reset fraud detection metrics (useful for testing)"""
        self.fraud_metrics = FraudMetrics()
        
    def get_fraud_summary(self) -> Dict[str, Any]:
        """Get quick fraud detection summary"""
        return {
            "precision": f"{self.fraud_metrics.precision:.1%}",
            "recall": f"{self.fraud_metrics.recall:.1%}",
            "f1_score": f"{self.fraud_metrics.f1_score:.1%}",
            "accuracy": f"{self.fraud_metrics.accuracy:.1%}",
            "total_fraud_events": self.fraud_metrics.true_positives + self.fraud_metrics.false_negatives,
            "correctly_detected": self.fraud_metrics.true_positives,
            "missed_fraud": self.fraud_metrics.false_negatives,
            "false_alarms": self.fraud_metrics.false_positives
        }


def create_banking_metrics_collector(simulation_id: str) -> BankingMetricsCollector:
    """Factory function to create banking metrics collector"""
    return BankingMetricsCollector(simulation_id)