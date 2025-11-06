# Carbon-Aware Route Optimization in Supply Chain Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)

This repository contains the complete implementation code and datasets for the paper:

**"Carbon-Aware Route Optimization in Supply Chain Networks Using Machine Learning and Genetic Algorithms"**

## Overview

This framework integrates **Machine Learning** (Random Forest + XGBoost ensemble) with **NSGA-II Genetic Algorithm** to optimize transportation routes, balancing carbon emissions reduction with operational costs.

### Key Results
- **ML Performance**: 9.48% MAPE, 0.928 R² for emission prediction
- **Synthetic Experiments**: 19.5% emission reduction, 4.7% cost increase (n=3,500 routes)
- **Quasi-Real Case Study**: 41.4% emission reduction, 8.6% cost increase (Salamanca network, n=12 routes)

---

## Quick Start

### 1. Prerequisites

- Python 3.13+
- 16GB RAM recommended
- macOS, Linux, or Windows

### 2. Installation

```bash
# Clone repository
git clone https://github.com/lspusal/carbon-aware-route-optimization.git
cd carbon-aware-route-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Reproduce Paper Results

#### Train ML Models (Table 2)
```bash
python train_paper_models.py
```
**Output**: ML performance metrics (MAPE, RMSE, R², MAE) for RF, XGBoost, and Ensemble

#### Generate Figures (Figures 2-5)
```bash
python generate_paper_figures.py
```
**Output**: Publication-quality figures saved to `figures/` directory

#### Run Quasi-Real Case Study (Table 4, Section 4.7)
```bash
python quasi_real_case_study.py
```
**Output**: Salamanca network optimization results with modal shift recommendations

---

## Repository Structure

```
.
├── src/                          # Source code modules
│   ├── ml_models/                # ML ensemble (RF + XGBoost)
│   │   ├── emission_predictors.py
│   │   └── __init__.py
│   ├── optimization/             # NSGA-II genetic algorithm
│   │   ├── hybrid_ml_ga.py
│   │   └── __init__.py
│   ├── data_prep/                # Data generation & connectors
│   │   ├── synthetic_generator.py
│   │   ├── realistic_generator.py
│   │   ├── epa_scraper.py
│   │   ├── climatiq_connector.py
│   │   └── __init__.py
│   └── utils/                    # Metrics & utilities
│       ├── metrics.py
│       └── __init__.py
├── data/                         # Datasets
│   ├── raw/                      # EPA, Climatiq emission factors (20+72 records)
│   ├── processed/                # Feature-engineered datasets (3,500 routes)
│   └── synthetic/                # Generated networks (50-2000 nodes)
├── train_paper_models.py        # Main training script (reproduces Table 2)
├── generate_paper_figures.py    # Figure generation (reproduces Figures 2-5)
├── quasi_real_case_study.py     # Salamanca case study (reproduces Table 4)
├── config.py                     # Configuration parameters
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Datasets

### Real-World Emission Factors
- **EPA Supply Chain GHG Emission Factors** v1.3 (2024): 20 transportation records
- **Climatiq API**: 72 emission factor records with regional variations
- **Sources**: Publicly accessible, authoritative data

### Synthetic Networks
- **3 scales**: 500, 1000, 2000 routes per network
- **Features**: Distance, cargo weight, mode, origin-destination pairs
- **Total**: 3,500 routes across controlled scenarios

### Quasi-Real Case Study
- **Network**: Salamanca (Spain) → 12 major Spanish cities
- **Distances**: 169-643 km (OpenStreetMap via OSRM API)
- **Modes**: Road, rail (ADIF network), maritime
- **Costs**: Spanish INE statistics (€0.082/ton-km road, €0.045/ton-km rail)

---

## Methodology

### Machine Learning Ensemble
- **Random Forest** (25% weight): Robustness, handles categorical features
- **XGBoost** (75% weight): High accuracy, gradient boosting
- **Training**: 5-fold cross-validation, blocked by OD pairs (prevents leakage)
- **Hyperparameters**: Grid search optimization

### NSGA-II Genetic Algorithm
- **Population**: 100 individuals
- **Generations**: 50
- **Operators**: Two-point crossover (0.8), adaptive mutation (0.1)
- **Objectives**: Minimize cost, minimize emissions (Pareto optimization)
- **Complexity**: O(M·N²) theoretical, O(R^1.2) empirical

### Physics-Based Features
- Distance normalization (log transformation)
- Cargo weight interactions
- Mode-specific emission factors
- Origin-destination embeddings (target encoding)

---

## Reproducing Paper Results

### Table 2: ML Performance Comparison

```bash
python train_paper_models.py
```

**Expected Output** (30 runs, seeds 42-71):
| Model      | MAPE (%) | RMSE (kg CO₂e) | R²    | MAE (kg CO₂e) |
|------------|----------|----------------|-------|---------------|
| RF         | 10.23    | 6,543          | 0.919 | 3,124         |
| XGBoost    | 9.88     | 6,087          | 0.933 | 2,987         |
| **Ensemble** | **9.48** | **6,143**  | **0.928** | **2,856** |

### Table 3: Optimization Results (Synthetic)

Generated during training, showing:
- Emission reduction: 19.5% average
- Cost increase: 4.7% average
- Pareto front coverage: 30-50 solutions
- Statistical significance: Cliff's delta δ=0.89 (large effect)

### Table 4: Quasi-Real Case Study (Salamanca)

```bash
python quasi_real_case_study.py
```

**Key Results**:
- 5 routes shifted from road → rail (distances ≥465 km)
- 776.6 tons CO₂e saved annually (41.4% reduction)
- €131,500 additional cost (8.6% increase)
- Strategic routes: Salamanca → Barcelona, Valencia, Málaga, Sevilla, Bilbao

### Figures 2-5: Performance Visualizations

```bash
python generate_paper_figures.py
```

**Generated Files**:
- `figures/figure2_ml_performance.pdf` - MAPE comparison (RF, XGBoost, Ensemble)
- `figures/figure3_prediction_validation.pdf` - Predicted vs. actual with error bands
- `figures/figure4_pareto_front.pdf` - Cost-emission trade-off curves
- `figures/figure5_scalability.pdf` - Runtime analysis (O(R^1.2) validation)

---

## Configuration

Edit `config.py` to customize:

```python
# ML Configuration
ML_CONFIG = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10
    },
    'xgboost': {
        'n_estimators': 150,
        'max_depth': 8,
        'learning_rate': 0.05
    },
    'ensemble_weights': {
        'rf': 0.25,
        'xgb': 0.75
    }
}

# GA Configuration
GA_CONFIG = {
    'population_size': 100,
    'generations': 50,
    'crossover_rate': 0.8,
    'mutation_rate': 0.1,
    'elite_size': 10
}

# Seeds for reproducibility
SEEDS = list(range(42, 72))  # 30 runs
```

---

## Dependencies

Core libraries (see `requirements.txt` for exact versions):

```
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.4.2
xgboost==2.0.3
matplotlib==3.8.3
seaborn==0.13.2
deap==1.4.1
```

**Installation**:
```bash
pip install -r requirements.txt
```

---

## Testing

### Quick Validation
```bash
# Test ML pipeline (single seed)
python train_paper_models.py --seed 42 --quick

# Test GA optimization (100 routes)
python -c "from src.optimization.hybrid_ml_ga import run_optimization; run_optimization(n_routes=100)"
```

### Full Validation (reproduces all paper results)
```bash
# Warning: Takes ~2-3 hours on M1 Pro
python train_paper_models.py  # 30 runs × 3 models
python generate_paper_figures.py
python quasi_real_case_study.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This research has been supported by the project "COordinated intelligent Services for Adaptive Smart areaS (COSASS)", Reference: PID2021-123673OB-C33, financed by MCIN/AEI/10.13039/501100011033/FEDER, UE.

---

## Data Sources

All data sources are publicly accessible:

1. **EPA Supply Chain GHG Emission Factors**: https://www.epa.gov/climateleadership/ghg-emission-factors-hub
2. **Climatiq API**: https://www.climatiq.io/
3. **OpenStreetMap (OSRM)**: https://project-osrm.org/
4. **ADIF (Spanish Rail Network)**: https://www.adif.es/
5. **INE (Spanish Statistics Institute)**: https://www.ine.es/
