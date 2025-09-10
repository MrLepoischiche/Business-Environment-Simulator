# Banking Environment Simulator

*[English version below](#english-version)*

Simulateur d'environnements métier pour la validation d'agents IA dans le secteur bancaire.

## 🚀 Installation et Démarrage Rapide

### Prérequis
- Python 3.11+
- Node.js 16+ (pour l'interface web)
- Docker et Docker Compose (optionnel)

### Installation

```bash
# Cloner le projet
git clone <repository-url>
cd banking-simulator

# Installer les dépendances Python
pip install -r requirements.txt

# Installer les dépendances frontend (optionnel)
cd frontend && npm install
```

### Démarrage Rapide - Demo Python

```bash
# Demo complète en ligne de commande
python banking_demo.py

# Ou directement
python -m environments.banking.simulation
```

### Démarrage Complet avec Interface Web

```bash
# 1. Démarrer l'infrastructure (optionnel)
docker-compose up -d postgres redis influxdb

# 2. Démarrer l'API
python api/main.py

# 3. Démarrer l'interface web
cd frontend && npm start
```

Accéder à l'interface : http://localhost:3000

## 🏗️ Architecture

### Structure du Projet

```
banking-simulator/
├── core/                     # Framework de simulation
│   ├── simulation_engine.py  # Moteur principal SimPy
│   ├── agent_manager.py      # Gestion agents IA
│   ├── event_dispatcher.py   # Système événements
│   └── metrics_collector.py  # Collecte métriques
├── environments/banking/     # Environnement bancaire
│   ├── entities.py          # Entités métier
│   ├── scenarios.py         # Scénarios simulation
│   ├── metrics.py           # Métriques bancaires
│   └── simulation.py        # Intégration
├── api/                     # API REST FastAPI
├── frontend/               # Interface React
├── tests/                  # Tests complets
└── validation/             # Suite validation
```

### Composants Principaux

- **Core Framework** : Moteur de simulation générique extensible
- **Banking Environment** : Spécialisé secteur bancaire avec détection fraude  
- **REST API** : Interface de contrôle et monitoring
- **Web Interface** : Dashboard temps réel avec métriques
- **Test Suite** : Tests unitaires et d'intégration
- **Validation Suite** : Validation performance et précision

## 🏦 Environnement Banking

### Entités Métier

- **Customer** : Clients avec profils comportementaux
- **Account** : Comptes avec soldes et limites
- **Transaction** : Transactions avec métadonnées fraude
- **Merchant** : Marchands avec codes MCC

### Scénarios Disponibles

1. **Retail Banking** : Banque de détail standard (2 tx/h/client)
2. **High Volume** : Test de charge (10 tx/h/client) 
3. **Fraud Detection** : Focus détection fraude (5% taux fraude)

### Métriques Bancaires

- **Fraud Detection** : Précision, Rappel, F1-Score, Exactitude
- **Business** : Volume transactions, satisfaction client
- **Operational** : Temps réponse, disponibilité système

## 🧪 Tests et Validation

### Exécuter les Tests

```bash
# Tests unitaires
pytest tests/test_core.py -v

# Tests banking
pytest tests/test_banking_environment.py -v

# Tests complets avec couverture
pytest tests/ -v --cov=core --cov=environments --cov-report=html

# Suite de validation complète
python validation/validation_suite.py
```

### Validation Suite

La suite de validation teste :
- ✅ Fonctionnalité (tous composants opérationnels)
- ✅ Précision (taux transactions, détection fraude)  
- ✅ Performance (throughput, scalabilité)
- ✅ Détection Fraude (précision >85%, rappel >70%)

## 📊 Utilisation API

### Endpoints Principaux

```bash
# Santé API
GET /health

# Créer simulation
POST /api/simulation/create
{
  "name": "Test Simulation",
  "duration_days": 7,
  "customer_count": 1000,
  "scenario_type": "fraud_detection",
  "fraud_rate": 0.02
}

# Démarrer simulation
POST /api/simulation/{id}/start

# Status en temps réel
GET /api/simulation/{id}/status

# Métriques détaillées
GET /api/simulation/{id}/metrics

# Rapport complet
GET /api/simulation/{id}/report
```

### Exemple d'Utilisation

```python
import asyncio
from core.simulation_engine import SimulationConfig
from environments.banking.simulation import create_banking_simulation

async def example():
    # Configuration
    config = SimulationConfig(
        name="Example Simulation",
        duration_days=1,
        time_acceleration=100.0,
        environment_type="banking"
    )
    
    # Créer moteur
    engine = create_banking_simulation(config)
    
    # Configuration environnement
    env_config = {
        "scenario_type": "retail",
        "scenario_params": {"customer_count": 500},
        "agents": [{
            "agent_id": "fraud_detector",
            "agent_type": "mock",
            "name": "Fraud Detector",
            "max_concurrent_requests": 5
        }]
    }
    
    # Exécuter
    await engine.initialize(env_config)
    results = await engine.run()
    
    # Analyser résultats
    fraud_metrics = results["fraud_summary"]
    print(f"Precision: {fraud_metrics['precision']}")
    print(f"Recall: {fraud_metrics['recall']}")

asyncio.run(example())
```

## 🐳 Déploiement Docker

### Démarrage Complet

```bash
# Infrastructure complète
docker-compose up -d

# Services disponibles :
# - API : http://localhost:8000
# - Grafana : http://localhost:3000  
# - Prometheus : http://localhost:9090
# - Interface : http://localhost:80
```

### Configuration Environnements

```yaml
# Variables d'environnement
DATABASE_URL=postgresql://user:pass@localhost:5432/simulator
REDIS_URL=redis://localhost:6379
INFLUXDB_URL=http://localhost:8086
LOG_LEVEL=INFO
```

## 📈 Métriques et Monitoring

### Métriques Principales

- **Throughput** : Transactions/seconde
- **Latency** : Temps de réponse agents
- **Fraud Detection** : Précision, Rappel, F1
- **System** : CPU, Mémoire, Disponibilité

### Dashboards Grafana

- Vue d'ensemble simulation
- Performance agents IA
- Métriques bancaires temps réel
- Alertes système

## 🔧 Personnalisation

### Ajouter un Nouvel Agent

```python
from core.agent_manager import AIAgentInterface

class CustomFraudAgent(AIAgentInterface):
    async def process_event(self, event):
        # Logique personnalisée
        return {
            "decision": "approve|reject",
            "confidence": 0.85,
            "reasoning": "Custom logic applied"
        }
```

### Créer un Nouveau Scénario

```python
from environments.banking.scenarios import BankingScenario

class CustomScenario(BankingScenario):
    def __init__(self, env, **kwargs):
        config = ScenarioConfig(
            name="Custom Scenario",
            # Configuration personnalisée
        )
        super().__init__(config, env)
```

## 🎯 Cas d'Usage

### Validation Pré-Production
- Tester agents IA avant déploiement
- Valider métriques business
- Simuler conditions extrêmes

### Formation et Démonstration  
- Environnement safe pour apprentissage
- Démonstration valeur ajoutée IA
- Workshops clients

### Recherche et Développement
- Benchmark algorithmes fraude
- Test nouvelles approches IA
- Analyse comportementale

## 📚 Documentation Avancée

### Extensions Possibles

1. **Nouveaux Secteurs** : Retail, Assurance, Santé
2. **Agents Avancés** : Intégration modèles réels
3. **Événements Complexes** : Crises, réglementations
4. **Analytics** : ML sur données simulation

### Contribution

1. Fork du projet
2. Créer une branche feature
3. Tests et validation
4. Pull request avec documentation

## 🔗 Liens Utiles

- **Documentation API** : http://localhost:8000/docs (après démarrage)
- **Métriques Prometheus** : http://localhost:9090
- **Dashboards Grafana** : http://localhost:3000
- **Interface Web** : http://localhost:80

## ⚠️ Limitations Connues

- Agents mock pour démonstration (pas de ML réel)
- Simulation déterministe (seed fixe)
- Données synthétiques uniquement
- Performance limitée par Python GIL

## 🆘 Dépannage

### Problèmes Courants

**Erreur Import** : Vérifier PYTHONPATH et structure dossiers
**Port Occupé** : Modifier ports dans docker-compose.yml
**Mémoire** : Réduire customer_count pour tests

### Logs et Debug

```bash
# Logs détaillés
export LOG_LEVEL=DEBUG
python banking_demo.py

# Logs Docker
docker-compose logs -f api
```

---

**Projet développé pour démonstration d'expertise en simulation d'environnements métier pour validation d'agents IA.**

---

## English Version

# Banking Environment Simulator

A business environment simulator for validating AI agents in the banking sector.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 16+ (for web interface)
- Docker and Docker Compose (optional)

### Installation

```bash
# Clone the project
git clone <repository-url>
cd py_environment_simulator

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies (optional)
cd frontend && npm install
```

### Quick Demo - Python CLI

```bash
# Complete CLI demo
python banking_demo.py

# Or directly
python -m environments.banking.simulation
```

### Full Setup with Web Interface

```bash
# 1. Start infrastructure (optional)
docker-compose up -d postgres redis influxdb

# 2. Start API
python api/main.py

# 3. Start web interface
cd frontend && npm start
```

Access the interface: http://localhost:3000

## 🏗️ Architecture

### Project Structure

```
py_environment_simulator/
├── core/                     # Simulation framework
│   ├── simulation_engine.py  # Main SimPy engine
│   ├── agent_manager.py      # AI agent management
│   ├── event_dispatcher.py   # Event system
│   └── metrics_collector.py  # Metrics collection
├── environments/banking/     # Banking environment
│   ├── entities.py          # Business entities
│   ├── scenarios.py         # Simulation scenarios
│   ├── metrics.py           # Banking metrics
│   └── simulation.py        # Integration layer
├── api/                     # FastAPI REST API
├── frontend/               # React interface
├── tests/                  # Comprehensive tests
└── validation/             # Validation suite
```

### Key Components

- **Core Framework**: Extensible generic simulation engine
- **Banking Environment**: Banking sector specialization with fraud detection
- **REST API**: Control and monitoring interface
- **Web Interface**: Real-time dashboard with metrics
- **Test Suite**: Unit and integration tests
- **Validation Suite**: Performance and accuracy validation

## 🏦 Banking Environment Features

### Business Entities

- **Customer**: Customers with behavioral profiles
- **Account**: Accounts with balances and limits
- **Transaction**: Transactions with fraud metadata
- **Merchant**: Merchants with MCC codes

### Available Scenarios

1. **Retail Banking**: Standard retail banking (2 tx/h/customer)
2. **High Volume**: Load testing (10 tx/h/customer)
3. **Fraud Detection**: Fraud detection focus (5% fraud rate)

### Banking Metrics

- **Fraud Detection**: Precision, Recall, F1-Score, Accuracy
- **Business**: Transaction volume, customer satisfaction
- **Operational**: Response time, system availability

## 🧪 Testing & Validation

### Running Tests

```bash
# Unit tests
pytest tests/test_core.py -v

# Banking tests
pytest tests/test_banking_environment.py -v

# Full test suite with coverage
pytest tests/ -v --cov=core --cov=environments --cov-report=html

# Complete validation suite
python validation/validation_suite.py
```

### Validation Results

The validation suite tests:
- ✅ Functionality (all components operational)
- ✅ Accuracy (transaction rates, fraud detection)
- ✅ Performance (throughput, scalability)
- ✅ Fraud Detection (precision >85%, recall >70%)

## 📊 API Usage

### Key Endpoints

```bash
# API health
GET /health

# Create simulation
POST /api/simulation/create
{
  "name": "Test Simulation",
  "duration_days": 7,
  "customer_count": 1000,
  "scenario_type": "fraud_detection",
  "fraud_rate": 0.02
}

# Start simulation
POST /api/simulation/{id}/start

# Real-time status
GET /api/simulation/{id}/status

# Detailed metrics
GET /api/simulation/{id}/metrics

# Complete report
GET /api/simulation/{id}/report
```

## 🐳 Docker Deployment

### Complete Setup

```bash
# Full infrastructure
docker-compose up -d

# Available services:
# - API: http://localhost:8000
# - Grafana: http://localhost:3000
# - Prometheus: http://localhost:9090
# - Interface: http://localhost:80
```

## 🔧 Technical Features

- **Async/Await**: Modern Python asynchronous programming
- **SimPy**: Discrete event simulation framework
- **FastAPI**: High-performance REST API
- **React**: Modern web interface
- **Docker**: Containerized deployment
- **Prometheus/Grafana**: Monitoring and visualization
- **Comprehensive Testing**: Unit, integration, and validation tests

## 🎯 Use Cases

### Pre-Production Validation
- Test AI agents before deployment
- Validate business metrics
- Simulate extreme conditions

### Training & Demonstration
- Safe learning environment
- AI value proposition demonstration
- Client workshops

### Research & Development
- Benchmark fraud algorithms
- Test new AI approaches
- Behavioral analysis

## ⚠️ Known Limitations

- Mock agents for demonstration (no real ML)
- Deterministic simulation (fixed seed)
- Synthetic data only
- Performance limited by Python GIL

---

**Project developed to demonstrate expertise in business environment simulation for AI agent validation.**