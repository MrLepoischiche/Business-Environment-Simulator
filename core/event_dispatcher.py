"""
Event Dispatcher for handling business events in simulation
Implements publish-subscribe pattern with event filtering and routing
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from uuid import uuid4
import json
import simpy
from collections import defaultdict, deque


class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BusinessEvent:
    """Represents a business event in the simulation"""
    event_type: str
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    timestamp: Optional[float] = None
    event_id: str = field(default_factory=lambda: str(uuid4()))
    source: Optional[str] = None
    requires_ai_processing: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().timestamp()
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "requires_ai_processing": self.requires_ai_processing,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessEvent':
        """Create event from dictionary"""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            data=data["data"],
            priority=EventPriority(data["priority"]),
            timestamp=data["timestamp"],
            source=data.get("source"),
            requires_ai_processing=data.get("requires_ai_processing", False),
            metadata=data.get("metadata", {})
        )


class EventFilter:
    """Filters events based on various criteria"""
    
    def __init__(self, event_types: Optional[Set[str]] = None, priorities: Optional[Set[EventPriority]] = None,
                 source_filter: Optional[Set[str]] = None, custom_filter: Optional[Callable[[BusinessEvent], bool]] = None):
        self.event_types = event_types or {"*"}  # "*" means all types
        self.priorities = priorities
        self.source_filter = source_filter
        self.custom_filter = custom_filter
        
    def matches(self, event: BusinessEvent) -> bool:
        """Check if event matches this filter"""
        # Check event type
        if "*" not in self.event_types and event.event_type not in self.event_types:
            return False
            
        # Check priority
        if self.priorities and event.priority not in self.priorities:
            return False
            
        # Check source
        if self.source_filter and event.source not in self.source_filter:
            return False
            
        # Check custom filter
        if self.custom_filter and not self.custom_filter(event):
            return False
            
        return True


@dataclass
class EventSubscription:
    """Represents a subscription to events"""
    subscriber_id: str
    callback: Callable[[BusinessEvent], None]
    event_filter: EventFilter
    active: bool = True


class EventQueue:
    """Priority queue for events"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues = {
            EventPriority.CRITICAL: deque(),
            EventPriority.HIGH: deque(),
            EventPriority.NORMAL: deque(),
            EventPriority.LOW: deque()
        }
        self.total_size = 0
        
    def put(self, event: BusinessEvent) -> bool:
        """Add event to queue, returns False if queue is full"""
        if self.total_size >= self.max_size:
            return False
            
        self.queues[event.priority].append(event)
        self.total_size += 1
        return True
        
    def get(self) -> Optional[BusinessEvent]:
        """Get highest priority event from queue"""
        # Process in priority order
        for priority in [EventPriority.CRITICAL, EventPriority.HIGH, 
                        EventPriority.NORMAL, EventPriority.LOW]:
            if self.queues[priority]:
                event = self.queues[priority].popleft()
                self.total_size -= 1
                return event
        return None
        
    def size(self) -> int:
        """Get total queue size"""
        return self.total_size
        
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.total_size == 0


class EventDispatcher:
    """
    Central event dispatcher for the simulation.
    Handles event routing, filtering, and delivery using pub-sub pattern.
    """
    
    def __init__(self, simpy_env: simpy.Environment):
        self.env = simpy_env
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.event_queue = EventQueue()
        self.logger = logging.getLogger("event_dispatcher")
        
        # Statistics
        self.events_processed = 0
        self.events_failed = 0
        self.events_by_type = defaultdict(int)
        self.processing_times = deque(maxlen=1000)  # Keep last 1000 processing times
        
        # Configuration
        self.batch_size = 10
        self.processing_interval = 0.01  # seconds
        self.max_retry_attempts = 3
        
        # Start background processing
        self.env.process(self._process_events())
        
    def _process_events(self):
        """Background process to handle queued events"""
        while True:
            try:
                # Process batch of events
                processed_count = 0
                start_time = self.env.now
                
                while not self.event_queue.empty() and processed_count < self.batch_size:
                    event = self.event_queue.get()
                    if event:
                        yield self.env.process(self._deliver_event(event))
                        processed_count += 1
                        
                # Update processing time statistics
                if processed_count > 0:
                    processing_time = self.env.now - start_time
                    self.processing_times.append(processing_time)
                    
                # Wait before next batch
                yield self.env.timeout(self.processing_interval)
                
            except Exception as e:
                self.logger.error(f"Error in event processing loop: {e}")
                yield self.env.timeout(1.0)  # Wait before retrying
                
    def _deliver_event(self, event: BusinessEvent):
        """Deliver event to matching subscribers"""
        matching_subscriptions = []
        
        # Find matching subscriptions
        for subscription in self.subscriptions.values():
            if subscription.active and subscription.event_filter.matches(event):
                matching_subscriptions.append(subscription)
                
        # Deliver to subscribers
        for subscription in matching_subscriptions:
            try:
                subscription.callback(event)
                self.events_processed += 1
            except Exception as e:
                self.logger.error(f"Error delivering event {event.event_id} to {subscription.subscriber_id}: {e}")
                self.events_failed += 1
                
        # Update statistics
        self.events_by_type[event.event_type] += 1
        
        # Small delay to prevent blocking
        yield self.env.timeout(0.001)
        
    async def dispatch(self, event: BusinessEvent) -> bool:
        """
        Dispatch an event to the system.
        Returns True if event was queued successfully.
        """
        try:
            # Add source if not specified
            if not event.source:
                event.source = "simulation"
                
            # Queue the event
            if self.event_queue.put(event):
                self.logger.debug(f"Event {event.event_id} queued: {event.event_type}")
                return True
            else:
                self.logger.warning(f"Event queue full, dropping event: {event.event_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to dispatch event {event.event_id}: {e}")
            return False
            
    def subscribe(self, event_types: str | List[str], callback: Callable[[BusinessEvent], None],
                 priorities: Optional[Set[EventPriority]] = None, source_filter: Optional[Set[str]] = None,
                 custom_filter: Optional[Callable[[BusinessEvent], bool]] = None,
                 subscriber_id: Optional[str] = None) -> str:
        """
        Subscribe to events with filtering.
        Returns subscription ID for later unsubscription.
        """
        if subscriber_id is None:
            subscriber_id = str(uuid4())
            
        # Convert single event type to set
        if isinstance(event_types, str):
            event_type_set = {event_types}
        else:
            event_type_set = set(event_types)
            
        event_filter = EventFilter(
            event_types=event_type_set,
            priorities=priorities,
            source_filter=source_filter,
            custom_filter=custom_filter
        )
        
        subscription = EventSubscription(
            subscriber_id=subscriber_id,
            callback=callback,
            event_filter=event_filter
        )
        
        self.subscriptions[subscriber_id] = subscription
        self.logger.info(f"New subscription: {subscriber_id} for events: {event_type_set}")
        
        return subscriber_id
        
    def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe from events"""
        if subscriber_id in self.subscriptions:
            del self.subscriptions[subscriber_id]
            self.logger.info(f"Unsubscribed: {subscriber_id}")
            return True
        return False
        
    def pause_subscription(self, subscriber_id: str) -> bool:
        """Pause a subscription temporarily"""
        if subscriber_id in self.subscriptions:
            self.subscriptions[subscriber_id].active = False
            return True
        return False
        
    def resume_subscription(self, subscriber_id: str) -> bool:
        """Resume a paused subscription"""
        if subscriber_id in self.subscriptions:
            self.subscriptions[subscriber_id].active = True
            return True
        return False
        
    def add_callback(self, callback: Callable[[BusinessEvent], None]) -> str:
        """Add a callback for all events (convenience method)"""
        return self.subscribe("*", callback)
        
    def create_event(self, event_type: str, data: Dict[str, Any], 
                    priority: EventPriority = EventPriority.NORMAL,
                    requires_ai: bool = False, source: Optional[str] = None) -> BusinessEvent:
        """Factory method to create events"""
        return BusinessEvent(
            event_type=event_type,
            data=data,
            priority=priority,
            requires_ai_processing=requires_ai,
            source=source or "dispatcher"
        )
        
    def get_event_count(self) -> int:
        """Get total number of processed events"""
        return self.events_processed
        
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.event_queue.size()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed event processing statistics"""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times) 
            if self.processing_times else 0
        )
        
        return {
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "success_rate": (self.events_processed / max(self.events_processed + self.events_failed, 1)),
            "queue_size": self.event_queue.size(),
            "active_subscriptions": len([s for s in self.subscriptions.values() if s.active]),
            "total_subscriptions": len(self.subscriptions),
            "events_by_type": dict(self.events_by_type),
            "average_processing_time": avg_processing_time,
            "queue_priority_breakdown": {
                priority.name: len(self.event_queue.queues[priority])
                for priority in EventPriority
            }
        }
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for reporting"""
        return {
            "total_events": self.events_processed + self.events_failed,
            "successful_events": self.events_processed,
            "failed_events": self.events_failed,
            "most_common_events": sorted(
                self.events_by_type.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]  # Top 5 most common event types
        }
        
    def clear_statistics(self) -> None:
        """Clear all statistics (useful for testing)"""
        self.events_processed = 0
        self.events_failed = 0
        self.events_by_type.clear()
        self.processing_times.clear()
        
    def shutdown(self) -> None:
        """Shutdown the event dispatcher"""
        self.logger.info("Shutting down event dispatcher")
        self.subscriptions.clear()
        # Clear queues
        for queue in self.event_queue.queues.values():
            queue.clear()
        self.event_queue.total_size = 0


# Utility functions for creating common event types
def create_transaction_event(transaction_id: str, amount: float, customer_id: str, 
                           merchant: str, requires_fraud_check: bool = True) -> BusinessEvent:
    """Create a banking transaction event"""
    return BusinessEvent(
        event_type="transaction",
        data={
            "transaction_id": transaction_id,
            "amount": amount,
            "customer_id": customer_id,
            "merchant": merchant,
            "agent_type": "fraud_detection" if requires_fraud_check else None
        },
        requires_ai_processing=requires_fraud_check,
        priority=EventPriority.HIGH if amount > 1000 else EventPriority.NORMAL
    )


def create_customer_interaction_event(customer_id: str, interaction_type: str, 
                                    data: Dict[str, Any]) -> BusinessEvent:
    """Create a customer interaction event"""
    return BusinessEvent(
        event_type="customer_interaction",
        data={
            "customer_id": customer_id,
            "interaction_type": interaction_type,
            **data
        },
        requires_ai_processing=interaction_type in ["complaint", "inquiry", "support"],
        priority=EventPriority.NORMAL
    )


def create_system_event(event_type: str, severity: str, data: Dict[str, Any]) -> BusinessEvent:
    """Create a system monitoring event"""
    priority_map = {
        "info": EventPriority.LOW,
        "warning": EventPriority.NORMAL,
        "error": EventPriority.HIGH,
        "critical": EventPriority.CRITICAL
    }
    
    return BusinessEvent(
        event_type=f"system_{event_type}",
        data={
            "severity": severity,
            **data
        },
        priority=priority_map.get(severity, EventPriority.NORMAL),
        source="system"
    )