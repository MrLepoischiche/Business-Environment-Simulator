"""
Metrics Collector for simulation performance and business metrics
Handles real-time collection, aggregation, and reporting of KPIs
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics
from enum import Enum
from uuid import uuid4


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    

@dataclass
class MetricValue:
    """Individual metric measurement"""
    timestamp: float
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    

class Metric:
    """Base metric class"""
    
    def __init__(self, definition: MetricDefinition):
        self.definition = definition
        self.values: deque = deque(maxlen=10000)  # Keep last 10k values
        self.current_value: Optional[float] = None

    def record(self, value: Union[int, float], labels: Optional[Dict[str, str]] = None,
              timestamp: Optional[float] = None) -> None:
        """Record a metric value"""
        if timestamp is None:
            timestamp = datetime.utcnow().timestamp()
            
        metric_value = MetricValue(
            timestamp=timestamp,
            value=float(value),
            labels=labels or {}
        )
        
        self.values.append(metric_value)
        self.current_value = float(value)
        
    def get_current_value(self) -> Optional[float]:
        """Get current metric value"""
        return self.current_value
        
    def get_values_in_window(self, window_seconds: int) -> List[MetricValue]:
        """Get values within time window"""
        if not self.values:
            return []
            
        cutoff_time = datetime.utcnow().timestamp() - window_seconds
        return [v for v in self.values if v.timestamp >= cutoff_time]
        
    def get_statistics(self, window_seconds: Optional[int] = None) -> Dict[str, float]:
        """Get statistical summary of metric"""
        values = self.get_values_in_window(window_seconds) if window_seconds else list(self.values)
        
        if not values:
            return {}
            
        numeric_values = [v.value for v in values]
        
        return {
            "count": len(numeric_values),
            "sum": sum(numeric_values),
            "avg": statistics.mean(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "median": statistics.median(numeric_values),
            "std_dev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0.0,
            "latest": numeric_values[-1] if numeric_values else 0.0
        }


class CounterMetric(Metric):
    """Counter metric - always increasing"""
    
    def __init__(self, definition: MetricDefinition):
        super().__init__(definition)
        self.total_count = 0.0

    def increment(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment counter"""
        self.total_count += amount
        self.record(self.total_count, labels)
        
    def reset(self) -> None:
        """Reset counter to zero"""
        self.total_count = 0.0
        self.record(0.0)


class GaugeMetric(Metric):
    """Gauge metric - can go up and down"""

    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set gauge value"""
        self.record(value, labels)

    def increment(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment gauge"""
        new_value = (self.current_value or 0) + amount
        self.record(new_value, labels)

    def decrement(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement gauge"""
        new_value = (self.current_value or 0) - amount
        self.record(new_value, labels)


class HistogramMetric(Metric):
    """Histogram metric for distribution tracking"""

    def __init__(self, definition: MetricDefinition, buckets: Optional[List[float]] = None):
        super().__init__(definition)
        self.buckets = buckets or [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        self.bucket_counts = {bucket: 0 for bucket in self.buckets}

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value"""
        self.record(value, labels)
        
        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                self.bucket_counts[bucket] += 1
                
    def get_histogram_data(self) -> Dict[str, Any]:
        """Get histogram distribution data"""
        values = [v.value for v in self.values]
        if not values:
            return {"buckets": self.bucket_counts, "total_count": 0}
            
        return {
            "buckets": self.bucket_counts.copy(),
            "total_count": len(values),
            "percentiles": {
                "p50": statistics.median(values),
                "p90": self._percentile(values, 0.9),
                "p95": self._percentile(values, 0.95),
                "p99": self._percentile(values, 0.99)
            } if len(values) >= 10 else {}
        }
        
    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile"""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]


class TimerMetric(HistogramMetric):
    """Timer metric for measuring durations"""
    
    def __init__(self, definition: MetricDefinition):
        # Use time-appropriate buckets (in seconds)
        buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
        super().__init__(definition, buckets)
        
    def time_function(self, func: Callable, *args, **kwargs) -> Any:
        """Time a function call"""
        start_time = datetime.utcnow().timestamp()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = datetime.utcnow().timestamp() - start_time
            self.observe(duration)
            
    async def time_async_function(self, func: Callable, *args, **kwargs) -> Any:
        """Time an async function call"""
        start_time = datetime.utcnow().timestamp()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = datetime.utcnow().timestamp() - start_time
            self.observe(duration)


class MetricsCollector:
    """
    Central metrics collection system for the simulation.
    Handles business and technical metrics with real-time aggregation.
    """
    
    def __init__(self, simulation_id: str):
        self.simulation_id = simulation_id
        self.metrics: Dict[str, Metric] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.logger = logging.getLogger(f"metrics.{simulation_id}")
        
        # Built-in business metrics
        self._register_builtin_metrics()
        
        # Aggregation settings
        self.aggregation_interval = 60  # seconds
        self.retention_hours = 24
        
        # Background tasks
        self._aggregation_task = None
        self._cleanup_task = None
        
    def _register_builtin_metrics(self) -> None:
        """Register built-in metrics for all simulations"""
        builtin_metrics = [
            # System metrics
            MetricDefinition("events_processed", MetricType.COUNTER, "Total events processed"),
            MetricDefinition("events_failed", MetricType.COUNTER, "Total events failed"),
            MetricDefinition("event_processing_time", MetricType.TIMER, "Event processing duration", "seconds"),
            MetricDefinition("active_agents", MetricType.GAUGE, "Number of active agents"),
            MetricDefinition("queue_size", MetricType.GAUGE, "Event queue size"),
            
            # Business metrics (banking example)
            MetricDefinition("transactions_processed", MetricType.COUNTER, "Total transactions processed"),
            MetricDefinition("fraud_detected", MetricType.COUNTER, "Fraud cases detected"),
            MetricDefinition("false_positives", MetricType.COUNTER, "False positive fraud alerts"),
            MetricDefinition("transaction_amount", MetricType.HISTOGRAM, "Transaction amounts", "currency"),
            MetricDefinition("customer_satisfaction", MetricType.GAUGE, "Customer satisfaction score", "score"),
            
            # Performance metrics
            MetricDefinition("response_time", MetricType.TIMER, "System response time", "seconds"),
            MetricDefinition("throughput", MetricType.GAUGE, "System throughput", "requests/second"),
            MetricDefinition("error_rate", MetricType.GAUGE, "Error rate", "percentage"),
        ]
        
        for definition in builtin_metrics:
            self.register_metric(definition)
            
    async def initialize(self) -> None:
        """Initialize metrics collector"""
        self.logger.info(f"Initializing metrics collector for simulation {self.simulation_id}")
        
        # Start background tasks
        self._aggregation_task = asyncio.create_task(self._aggregation_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Metrics collector initialized")
        
    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a new metric definition"""
        self.metric_definitions[definition.name] = definition
        
        # Create appropriate metric instance
        if definition.metric_type == MetricType.COUNTER:
            metric = CounterMetric(definition)
        elif definition.metric_type == MetricType.GAUGE:
            metric = GaugeMetric(definition)
        elif definition.metric_type == MetricType.HISTOGRAM:
            metric = HistogramMetric(definition)
        elif definition.metric_type == MetricType.TIMER:
            metric = TimerMetric(definition)
        else:
            metric = Metric(definition)
            
        self.metrics[definition.name] = metric
        self.logger.debug(f"Registered metric: {definition.name} ({definition.metric_type.value})")

    def increment_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric"""
        metric = self.metrics.get(name)
        if isinstance(metric, CounterMetric):
            metric.increment(amount, labels)
        else:
            self.logger.warning(f"Counter metric not found or not a CounterMetric: {name}")

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value"""
        metric = self.metrics.get(name)
        if isinstance(metric, GaugeMetric):
            metric.set(value, labels)
        else:
            self.logger.warning(f"Gauge metric not found or not a GaugeMetric: {name}")

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a histogram metric value"""
        metric = self.metrics.get(name)
        if isinstance(metric, HistogramMetric):
            metric.observe(value, labels)
        else:
            self.logger.warning(f"Histogram metric not found or not a HistogramMetric: {name}")

    def time_operation(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return TimerContext(self, name, labels)

    def record_business_metric(self, metric_type: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a business-specific metric"""
        metric_name = f"business_{metric_type}"
        
        # Auto-register if not exists
        if metric_name not in self.metrics:
            definition = MetricDefinition(
                name=metric_name,
                metric_type=MetricType.GAUGE,
                description=f"Business metric: {metric_type}",
                unit=metadata.get("unit", "") if metadata else ""
            )
            self.register_metric(definition)
        
        metric = self.metrics.get(metric_name)
        if isinstance(metric, GaugeMetric):
            labels = metadata.get("labels", {}) if metadata else {}
            metric.set(value, labels)
        else:
            self.logger.warning(f"Metric '{metric_name}' is not a GaugeMetric and cannot be set.")
            
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current values of all metrics"""
        current = {}
        for name, metric in self.metrics.items():
            current[name] = {
                "value": metric.get_current_value(),
                "type": metric.definition.metric_type.value,
                "unit": metric.definition.unit,
                "description": metric.definition.description
            }
        return current
        
    async def get_final_metrics(self) -> Dict[str, Any]:
        """Get comprehensive final metrics report"""
        final_metrics = {}
        
        for name, metric in self.metrics.items():
            stats = metric.get_statistics()
            
            final_metrics[name] = {
                "definition": {
                    "type": metric.definition.metric_type.value,
                    "unit": metric.definition.unit,
                    "description": metric.definition.description
                },
                "statistics": stats,
                "current_value": metric.get_current_value()
            }
            
            # Add histogram-specific data
            if isinstance(metric, HistogramMetric):
                final_metrics[name]["histogram"] = metric.get_histogram_data()
                
        return final_metrics
        
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of key metrics for reporting"""
        return {
            "simulation_id": self.simulation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "key_metrics": {
                "events_processed": self.metrics["events_processed"].get_current_value() if "events_processed" in self.metrics else 0,
                "events_failed": self.metrics["events_failed"].get_current_value() if "events_failed" in self.metrics else 0,
                "active_agents": self.metrics["active_agents"].get_current_value() if "active_agents" in self.metrics else 0,
                "transactions_processed": self.metrics["transactions_processed"].get_current_value() if "transactions_processed" in self.metrics else 0,
                "fraud_detected": self.metrics["fraud_detected"].get_current_value() if "fraud_detected" in self.metrics else 0,
            },
            "performance": {
                "avg_response_time": self._get_avg_metric("response_time"),
                "error_rate": self.metrics["error_rate"].get_current_value() if "error_rate" in self.metrics else 0,
                "throughput": self.metrics["throughput"].get_current_value() if "throughput" in self.metrics else 0,
            }
        }
        
    def _get_avg_metric(self, name: str, window_seconds: int = 300) -> float:
        """Get average value of a metric over time window"""
        if name not in self.metrics:
            return 0.0
            
        stats = self.metrics[name].get_statistics(window_seconds)
        return stats.get("avg", 0.0)
        
    async def _aggregation_loop(self) -> None:
        """Background loop for metric aggregation"""
        while True:
            try:
                await asyncio.sleep(self.aggregation_interval)
                # Perform any necessary aggregations
                await self._perform_aggregations()
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {e}")
                
    async def _cleanup_loop(self) -> None:
        """Background loop for old data cleanup"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                
    async def _perform_aggregations(self) -> None:
        """Perform metric aggregations"""
        # Calculate derived metrics
        events_processed_metric = self.metrics.get("events_processed")
        events_processed = events_processed_metric.get_current_value() if events_processed_metric else 0
        
        events_failed_metric = self.metrics.get("events_failed")
        events_failed = events_failed_metric.get_current_value() if events_failed_metric else 0
        
        total_events = (events_processed or 0) + (events_failed or 0)
        
        if total_events > 0:
            error_rate = ((events_failed or 0) / total_events) * 100
            self.set_gauge("error_rate", error_rate)
            
    async def _cleanup_old_data(self) -> None:
        """Clean up old metric data"""
        cutoff_time = datetime.utcnow().timestamp() - (self.retention_hours * 3600)
        
        for metric in self.metrics.values():
            # Remove old values
            while metric.values and metric.values[0].timestamp < cutoff_time:
                metric.values.popleft()
                
    async def shutdown(self) -> None:
        """Shutdown metrics collector"""
        self.logger.info("Shutting down metrics collector")
        
        if self._aggregation_task:
            self._aggregation_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
            
        self.logger.info("Metrics collector shutdown complete")


class TimerContext:
    """Context manager for timing operations"""

    def __init__(self, collector: MetricsCollector, metric_name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.utcnow().timestamp()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = datetime.utcnow().timestamp() - self.start_time
            self.collector.observe_histogram(self.metric_name, duration, self.labels)