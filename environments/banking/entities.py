"""
Banking Environment Entities
Defines all business entities for banking simulation
"""
import random
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class CustomerType(Enum):
    INDIVIDUAL = "individual"
    BUSINESS = "business"
    PREMIUM = "premium"


class AccountType(Enum):
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT = "credit"
    BUSINESS = "business"


class TransactionType(Enum):
    DEBIT = "debit"
    CREDIT = "credit"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    PAYMENT = "payment"


class TransactionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class Channel(Enum):
    ATM = "atm"
    ONLINE = "online"
    MOBILE = "mobile"
    BRANCH = "branch"
    PHONE = "phone"
    POS = "pos"  # Point of Sale


@dataclass
class Address:
    """Customer address information"""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "US"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "street": self.street,
            "city": self.city,
            "state": self.state,
            "zip_code": self.zip_code,
            "country": self.country
        }


@dataclass
class CustomerProfile:
    """Customer behavioral profile for simulation"""
    risk_score: float  # 0-1, higher = riskier
    avg_monthly_transactions: int
    avg_transaction_amount: float
    preferred_channels: List[Channel]
    peak_hours: List[int]  # Hours of day (0-23)
    fraud_probability: float = 0.01  # Base fraud probability
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "avg_monthly_transactions": self.avg_monthly_transactions,
            "avg_transaction_amount": self.avg_transaction_amount,
            "preferred_channels": [c.value for c in self.preferred_channels],
            "peak_hours": self.peak_hours,
            "fraud_probability": self.fraud_probability
        }


@dataclass 
class Customer:
    """Banking customer entity"""
    customer_id: str
    first_name: str
    last_name: str
    email: str
    phone: str
    address: Address
    customer_type: CustomerType
    date_of_birth: datetime
    registration_date: datetime
    profile: CustomerProfile
    accounts: List['Account'] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.customer_id:
            self.customer_id = str(uuid.uuid4())
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    @property
    def age(self) -> int:
        return (datetime.now() - self.date_of_birth).days // 365
    
    def add_account(self, account: 'Account') -> None:
        """Add account to customer"""
        self.accounts.append(account)
        account.customer_id = self.customer_id
    
    def get_account(self, account_id: str) -> Optional['Account']:
        """Get account by ID"""
        return next((acc for acc in self.accounts if acc.account_id == account_id), None)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address.to_dict(),
            "customer_type": self.customer_type.value,
            "date_of_birth": self.date_of_birth.isoformat(),
            "registration_date": self.registration_date.isoformat(),
            "profile": self.profile.to_dict(),
            "accounts": [acc.to_dict() for acc in self.accounts]
        }


@dataclass
class Account:
    """Bank account entity"""
    account_id: str
    customer_id: str
    account_type: AccountType
    balance: float
    credit_limit: float = 0.0
    interest_rate: float = 0.0
    created_date: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    daily_transaction_limit: float = 5000.0
    monthly_transaction_limit: float = 50000.0
    
    def __post_init__(self):
        if not self.account_id:
            self.account_id = f"ACC_{uuid.uuid4().hex[:8].upper()}"
    
    @property
    def available_balance(self) -> float:
        """Available balance including credit limit"""
        return self.balance + self.credit_limit
    
    def can_debit(self, amount: float) -> bool:
        """Check if account can be debited"""
        return self.is_active and self.available_balance >= amount
    
    def debit(self, amount: float) -> bool:
        """Debit account if possible"""
        if self.can_debit(amount):
            self.balance -= amount
            return True
        return False
    
    def credit(self, amount: float) -> None:
        """Credit account"""
        if self.is_active:
            self.balance += amount
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_id": self.account_id,
            "customer_id": self.customer_id,
            "account_type": self.account_type.value,
            "balance": self.balance,
            "credit_limit": self.credit_limit,
            "interest_rate": self.interest_rate,
            "created_date": self.created_date.isoformat(),
            "is_active": self.is_active,
            "daily_transaction_limit": self.daily_transaction_limit,
            "monthly_transaction_limit": self.monthly_transaction_limit,
            "available_balance": self.available_balance
        }


@dataclass
class Merchant:
    """Merchant entity for transactions"""
    merchant_id: str
    name: str
    category: str
    mcc_code: str  # Merchant Category Code
    risk_level: str = "low"  # low, medium, high
    
    def __post_init__(self):
        if not self.merchant_id:
            self.merchant_id = f"MER_{uuid.uuid4().hex[:8].upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "merchant_id": self.merchant_id,
            "name": self.name,
            "category": self.category,
            "mcc_code": self.mcc_code,
            "risk_level": self.risk_level
        }


@dataclass
class Transaction:
    """Banking transaction entity"""
    transaction_id: str
    account_id: str
    customer_id: str
    transaction_type: TransactionType
    amount: float
    description: str
    merchant: Optional[Merchant] = None
    channel: Channel = Channel.ONLINE
    timestamp: datetime = field(default_factory=datetime.now)
    status: TransactionStatus = TransactionStatus.PENDING
    reference_id: Optional[str] = None
    location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Fraud detection fields
    fraud_score: Optional[float] = None
    fraud_reasons: List[str] = field(default_factory=list)
    is_suspicious: bool = False
    
    def __post_init__(self):
        if not self.transaction_id:
            self.transaction_id = f"TXN_{uuid.uuid4().hex[:12].upper()}"
    
    def mark_as_fraud(self, score: float, reasons: List[str]) -> None:
        """Mark transaction as potentially fraudulent"""
        self.fraud_score = score
        self.fraud_reasons = reasons
        self.is_suspicious = score > 0.7
        if score > 0.9:
            self.status = TransactionStatus.REJECTED
    
    def approve(self) -> None:
        """Approve the transaction"""
        self.status = TransactionStatus.APPROVED

    def reject(self, reason: Optional[str] = None) -> None:
        """Reject the transaction"""
        self.status = TransactionStatus.REJECTED
        if reason:
            self.fraud_reasons.append(reason)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "account_id": self.account_id,
            "customer_id": self.customer_id,
            "transaction_type": self.transaction_type.value,
            "amount": self.amount,
            "description": self.description,
            "merchant": self.merchant.to_dict() if self.merchant else None,
            "channel": self.channel.value,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "reference_id": self.reference_id,
            "location": self.location,
            "metadata": self.metadata,
            "fraud_score": self.fraud_score,
            "fraud_reasons": self.fraud_reasons,
            "is_suspicious": self.is_suspicious
        }


class CustomerFactory:
    """Factory for creating realistic customers"""
    
    FIRST_NAMES = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily", 
                   "William", "Jessica", "James", "Ashley", "Christopher", "Amanda", "Daniel"]
    
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", 
                  "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez"]
    
    CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
              "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville"]
    
    STATES = ["NY", "CA", "IL", "TX", "AZ", "PA", "FL", "WA", "OH", "GA", "NC", "MI"]
    
    MERCHANTS = [
        ("Walmart", "retail", "5411"), ("Amazon", "online", "5969"), 
        ("Starbucks", "restaurant", "5814"), ("Shell", "gas", "5541"),
        ("McDonald's", "restaurant", "5814"), ("Target", "retail", "5411"),
        ("Home Depot", "retail", "5200"), ("CVS Pharmacy", "pharmacy", "5912"),
        ("Costco", "retail", "5411"), ("Apple Store", "electronics", "5732")
    ]
    
    @classmethod
    def create_customer(cls, customer_type: CustomerType = CustomerType.INDIVIDUAL) -> Customer:
        """Create a realistic customer"""
        first_name = random.choice(cls.FIRST_NAMES)
        last_name = random.choice(cls.LAST_NAMES)
        
        # Generate realistic profile
        if customer_type == CustomerType.PREMIUM:
            risk_score = random.uniform(0.1, 0.3)  # Lower risk
            avg_monthly = random.randint(50, 200)
            avg_amount = random.uniform(200, 2000)
        elif customer_type == CustomerType.BUSINESS:
            risk_score = random.uniform(0.2, 0.5)
            avg_monthly = random.randint(100, 500)
            avg_amount = random.uniform(500, 5000)
        else:
            risk_score = random.uniform(0.1, 0.7)
            avg_monthly = random.randint(10, 100)
            avg_amount = random.uniform(50, 500)
        
        profile = CustomerProfile(
            risk_score=risk_score,
            avg_monthly_transactions=avg_monthly,
            avg_transaction_amount=avg_amount,
            preferred_channels=random.sample(list(Channel), k=random.randint(2, 4)),
            peak_hours=random.sample(range(8, 20), k=random.randint(2, 6)),
            fraud_probability=risk_score * 0.02
        )
        
        # Generate address
        city = random.choice(cls.CITIES)
        state = random.choice(cls.STATES)
        address = Address(
            street=f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'First', 'Second'])} St",
            city=city,
            state=state,
            zip_code=f"{random.randint(10000, 99999)}",
            country="US"
        )
        
        # Create customer
        customer = Customer(
            customer_id="",  # Will be auto-generated
            first_name=first_name,
            last_name=last_name,
            email=f"{first_name.lower()}.{last_name.lower()}@email.com",
            phone=f"{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
            address=address,
            customer_type=customer_type,
            date_of_birth=datetime.now() - timedelta(days=random.randint(18*365, 80*365)),
            registration_date=datetime.now() - timedelta(days=random.randint(30, 3650)),
            profile=profile
        )
        
        # Create accounts
        num_accounts = 1 if customer_type == CustomerType.INDIVIDUAL else random.randint(1, 3)
        
        for i in range(num_accounts):
            if i == 0:
                account_type = AccountType.CHECKING
                balance = random.uniform(500, 10000)
            elif i == 1:
                account_type = AccountType.SAVINGS
                balance = random.uniform(1000, 50000)
            else:
                account_type = AccountType.CREDIT
                balance = random.uniform(-1000, 0)  # Credit accounts can have negative balance
                
            account = Account(
                account_id="",  # Will be auto-generated
                customer_id=customer.customer_id,
                account_type=account_type,
                balance=balance,
                credit_limit=5000.0 if account_type == AccountType.CREDIT else 0.0,
                interest_rate=random.uniform(0.01, 0.05) if account_type == AccountType.SAVINGS else 0.0
            )
            
            customer.add_account(account)
        
        return customer
    
    @classmethod
    def create_merchant(cls) -> Merchant:
        """Create a realistic merchant"""
        name, category, mcc = random.choice(cls.MERCHANTS)
        
        # Add some variation to merchant names
        if random.random() < 0.3:
            name += f" #{random.randint(1000, 9999)}"
        
        risk_level = random.choices(
            ["low", "medium", "high"],
            weights=[70, 25, 5]  # Most merchants are low risk
        )[0]
        
        return Merchant(
            merchant_id="",  # Will be auto-generated
            name=name,
            category=category,
            mcc_code=mcc,
            risk_level=risk_level
        )


class TransactionGenerator:
    """Generates realistic banking transactions"""
    
    def __init__(self, customers: List[Customer], merchants: List[Merchant]):
        self.customers = customers
        self.merchants = merchants

    def generate_transaction(self, customer: Optional[Customer] = None,
                           force_suspicious: bool = False) -> Transaction:
        """Generate a realistic transaction"""
        if customer is None:
            customer = random.choice(self.customers)
        
        # Select account (prefer checking for transactions)
        checking_accounts = [acc for acc in customer.accounts if acc.account_type == AccountType.CHECKING]
        if checking_accounts:
            account = random.choice(checking_accounts)
        else:
            account = random.choice(customer.accounts)
        
        # Determine transaction type and amount
        tx_type = random.choices(
            [TransactionType.DEBIT, TransactionType.CREDIT, TransactionType.WITHDRAWAL, TransactionType.DEPOSIT],
            weights=[60, 20, 15, 5]
        )[0]
        
        # Generate amount based on customer profile
        if force_suspicious:
            # Make suspicious transaction
            amount = random.uniform(customer.profile.avg_transaction_amount * 5, 
                                  min(account.daily_transaction_limit, 10000))
        else:
            # Normal transaction
            base_amount = customer.profile.avg_transaction_amount
            amount = random.uniform(base_amount * 0.1, base_amount * 3)
        
        amount = round(amount, 2)
        
        # Select merchant and channel
        merchant = random.choice(self.merchants)
        channel = random.choice(customer.profile.preferred_channels)
        
        # Generate description
        if tx_type in [TransactionType.DEBIT, TransactionType.PAYMENT]:
            description = f"Purchase at {merchant.name}"
        elif tx_type == TransactionType.CREDIT:
            description = "Direct deposit" if amount > 1000 else "Refund"
        elif tx_type == TransactionType.WITHDRAWAL:
            description = "ATM Withdrawal"
            channel = Channel.ATM
        else:
            description = "Deposit"
            channel = Channel.BRANCH
        
        # Create transaction
        transaction = Transaction(
            transaction_id="",  # Will be auto-generated
            account_id=account.account_id,
            customer_id=customer.customer_id,
            transaction_type=tx_type,
            amount=amount,
            description=description,
            merchant=merchant if tx_type in [TransactionType.DEBIT, TransactionType.PAYMENT] else None,
            channel=channel,
            timestamp=datetime.now(),
            location=f"{customer.address.city}, {customer.address.state}" if random.random() < 0.8 else "Online",
            metadata={
                "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "device_id": f"device_{random.randint(1000, 9999)}",
                "session_id": str(uuid.uuid4())
            }
        )
        
        # Add suspicious indicators if forced
        if force_suspicious:
            transaction.metadata.update({
                "unusual_time": datetime.now().hour < 6 or datetime.now().hour > 22,
                "unusual_location": random.choice(["Foreign Country", "High Risk Area"]),
                "velocity_flag": True
            })
        
        return transaction