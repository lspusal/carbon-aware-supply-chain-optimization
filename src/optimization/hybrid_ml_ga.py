"""
Hybrid ML-GA Optimization Framework

Combines machine learning predictions with genetic algorithm optimization
for multi-objective route selection in carbon-aware supply chain planning.

Features:
- Multi-objective optimization (minimize cost and emissions)
- Genetic algorithm implementation using DEAP
- Integration with ML emission prediction models
- Support for various optimization strategies
- Pareto frontier analysis and visualization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Any, Optional
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import logging
import time
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class Route:
    """Data class representing a transportation route with optimization parameters."""
    origin: str
    destination: str
    distance_km: float
    weight_tons: float
    commodity_type: str
    transport_mode: str
    temperature_c: float = 20.0
    congestion_factor: float = 1.0
    slope_degrees: float = 0.0
    is_peak_hour: bool = False
    is_weekend: bool = False
    weather_condition: str = 'clear'


class HybridMLGA:
    """
    Hybrid Machine Learning and Genetic Algorithm optimizer.
    
    Uses ML models to predict emissions and GA to optimize routes
    for multi-objective cost and emission minimization.
    """
    
    def __init__(self, ml_predictor, cost_calculator, population_size: int = 100):
        """
        Initialize hybrid ML-GA optimizer.
        
        Args:
            ml_predictor: Trained ML model for emission prediction
            cost_calculator: Function to calculate route costs
            population_size: Size of GA population
        """
        self.ml_predictor = ml_predictor
        self.cost_calculator = cost_calculator
        self.population_size = population_size
        
        # Setup DEAP framework
        self._setup_deap()
        
        # Optimization parameters
        self.crossover_prob = 0.7
        self.mutation_prob = 0.2
        self.n_generations = 100
        
        # Results storage
        self.optimization_history = []
        self.pareto_front = []
        
    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework."""
        
        # Create multi-objective minimization problem
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Minimize cost and emissions
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selNSGA2)
        
    def optimize_routes(self, routes: List[Route], n_generations: int = None) -> Dict:
        """
        Optimize routes using hybrid ML-GA approach.
        
        Args:
            routes: List of routes to optimize
            n_generations: Number of GA generations
            
        Returns:
            Dict: Optimization results including Pareto front
        """
        if n_generations:
            self.n_generations = n_generations
            
        logger.info(f"Starting hybrid ML-GA optimization for {len(routes)} routes")
        start_time = time.time()
        
        # Store original routes for comparison
        self.original_routes = routes.copy()
        
        # Generate initial population
        population = self._generate_initial_population(routes)
        
        # Evaluate initial population
        for individual in population:
            individual.fitness.values = self.toolbox.evaluate(individual)
            
        # Evolution loop
        for generation in range(self.n_generations):
            # Select parents
            offspring = algorithms.varAnd(population, self.toolbox, 
                                        self.crossover_prob, self.mutation_prob)
            
            # Evaluate offspring
            for individual in offspring:
                if not individual.fitness.valid:
                    individual.fitness.values = self.toolbox.evaluate(individual)
                    
            # Select next generation
            population = self.toolbox.select(offspring + population, self.population_size)
            
            # Record statistics
            self._record_generation_stats(generation, population)
            
            if generation % 20 == 0:
                logger.info(f"Generation {generation}/{self.n_generations}")
                
        # Extract Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        self.pareto_front = pareto_front
        
        end_time = time.time()
        
        results = {
            'pareto_front': self._decode_pareto_front(pareto_front),
            'optimization_time': end_time - start_time,
            'n_generations': self.n_generations,
            'population_size': self.population_size,
            'optimization_history': self.optimization_history,
            'improvement_metrics': self._calculate_improvements()
        }
        
        logger.info(f"Optimization completed in {results['optimization_time']:.2f} seconds")
        return results
        
    def _generate_initial_population(self, routes: List[Route]) -> List:
        """Generate initial population for GA."""
        population = []
        
        for _ in range(self.population_size):
            # Create individual by randomly modifying routes
            individual = []
            
            for route in routes:
                # Encode route parameters that can be optimized
                # Transport mode (0: truck, 1: rail, 2: ship, 3: air)
                transport_modes = ['truck', 'rail', 'ship', 'air']
                mode_idx = random.randint(0, len(transport_modes) - 1)
                
                # Route variations (could represent alternative paths)
                route_variation = random.uniform(0.9, 1.1)  # ±10% distance variation
                
                # Timing optimization (avoid peak hours)
                avoid_peak = random.choice([True, False])
                
                individual.extend([mode_idx, route_variation, int(avoid_peak)])
                
            population.append(creator.Individual(individual))
            
        return population
        
    def _evaluate_individual(self, individual: List) -> Tuple[float, float]:
        """
        Evaluate individual's fitness (cost and emissions).
        
        Args:
            individual: GA individual representing route configuration
            
        Returns:
            Tuple[float, float]: (total_cost, total_emissions)
        """
        total_cost = 0.0
        total_emissions = 0.0
        
        # Decode individual into route modifications
        routes_per_individual = len(individual) // 3
        
        for i in range(routes_per_individual):
            if i >= len(self.original_routes):
                break
                
            base_route = self.original_routes[i]
            
            # Extract parameters for this route
            mode_idx = int(individual[i * 3])
            route_variation = individual[i * 3 + 1]
            avoid_peak = bool(individual[i * 3 + 2])
            
            # Create modified route
            transport_modes = ['truck', 'rail', 'ship', 'air']
            mode_idx = min(mode_idx, len(transport_modes) - 1)
            
            modified_route = Route(
                origin=base_route.origin,
                destination=base_route.destination,
                distance_km=base_route.distance_km * route_variation,
                weight_tons=base_route.weight_tons,
                commodity_type=base_route.commodity_type,
                transport_mode=transport_modes[mode_idx],
                temperature_c=base_route.temperature_c,
                congestion_factor=base_route.congestion_factor,
                slope_degrees=base_route.slope_degrees,
                is_peak_hour=base_route.is_peak_hour and not avoid_peak,
                is_weekend=base_route.is_weekend,
                weather_condition=base_route.weather_condition
            )
            
            # Calculate cost and emissions
            route_cost = self._calculate_route_cost(modified_route)
            route_emissions = self._predict_route_emissions(modified_route)
            
            total_cost += route_cost
            total_emissions += route_emissions
            
        return total_cost, total_emissions
        
    def _calculate_route_cost(self, route: Route) -> float:
        """Calculate cost for a single route."""
        # Base cost per km by transport mode
        base_costs = {
            'truck': 1.5,
            'rail': 0.8,
            'ship': 0.3,
            'air': 12.0
        }
        
        base_cost = route.distance_km * base_costs.get(route.transport_mode, 1.5)
        
        # Apply adjustments
        if route.is_peak_hour:
            base_cost *= 1.15
            
        base_cost *= route.congestion_factor
        
        # Slope adjustment
        slope_factor = 1 + (route.slope_degrees * 0.02)
        base_cost *= slope_factor
        
        return base_cost
        
    def _predict_route_emissions(self, route: Route) -> float:
        """Predict emissions for a route using ML model."""
        # Convert route to format expected by ML model
        route_data = {
            'distance_km': route.distance_km,
            'weight_tons': route.weight_tons,
            'transport_mode': route.transport_mode,
            'commodity_type': route.commodity_type,
            'temperature_c': route.temperature_c,
            'congestion_factor': route.congestion_factor,
            'slope_degrees': route.slope_degrees,
            'is_peak_hour': route.is_peak_hour,
            'is_weekend': route.is_weekend,
            'weather_condition': route.weather_condition
        }
        
        # Use ML predictor
        if hasattr(self.ml_predictor, 'predict_emissions'):
            return self.ml_predictor.predict_emissions(route_data)
        else:
            # Fallback calculation if ML predictor not available
            emission_factors = {
                'truck': 0.161, 'rail': 0.041, 'ship': 0.015, 'air': 0.602
            }
            factor = emission_factors.get(route.transport_mode, 0.161)
            return route.distance_km * route.weight_tons * factor
            
    def _mutate_individual(self, individual: List, mutation_rate: float = 0.1) -> Tuple[List]:
        """Custom mutation operator for route optimization."""
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                if i % 3 == 0:  # Transport mode
                    individual[i] = random.randint(0, 3)
                elif i % 3 == 1:  # Route variation
                    individual[i] = random.uniform(0.9, 1.1)
                else:  # Avoid peak
                    individual[i] = random.choice([0, 1])
                    
        return individual,
        
    def _record_generation_stats(self, generation: int, population: List):
        """Record statistics for each generation."""
        costs = [ind.fitness.values[0] for ind in population]
        emissions = [ind.fitness.values[1] for ind in population]
        
        stats = {
            'generation': generation,
            'mean_cost': np.mean(costs),
            'mean_emissions': np.mean(emissions),
            'min_cost': np.min(costs),
            'min_emissions': np.min(emissions),
            'std_cost': np.std(costs),
            'std_emissions': np.std(emissions)
        }
        
        self.optimization_history.append(stats)
        
    def _decode_pareto_front(self, pareto_front: List) -> List[Dict]:
        """Decode Pareto front solutions into readable format."""
        solutions = []
        
        for individual in pareto_front:
            cost, emissions = individual.fitness.values
            
            solution = {
                'total_cost': cost,
                'total_emissions': emissions,
                'cost_per_route': cost / max(1, len(self.original_routes)),
                'emissions_per_route': emissions / max(1, len(self.original_routes)),
                'encoding': list(individual)
            }
            
            solutions.append(solution)
            
        return solutions
        
    def _calculate_improvements(self) -> Dict:
        """Calculate improvement metrics vs original routes."""
        if not self.original_routes:
            return {}
            
        # Calculate baseline metrics
        baseline_cost = sum(self._calculate_route_cost(route) for route in self.original_routes)
        baseline_emissions = sum(self._predict_route_emissions(route) for route in self.original_routes)
        
        # Get best solutions from Pareto front
        if not self.pareto_front:
            return {}
            
        best_cost_solution = min(self.pareto_front, key=lambda x: x.fitness.values[0])
        best_emission_solution = min(self.pareto_front, key=lambda x: x.fitness.values[1])
        
        cost_improvement = (baseline_cost - best_cost_solution.fitness.values[0]) / baseline_cost * 100
        emission_improvement = (baseline_emissions - best_emission_solution.fitness.values[1]) / baseline_emissions * 100
        
        return {
            'baseline_cost': baseline_cost,
            'baseline_emissions': baseline_emissions,
            'best_cost': best_cost_solution.fitness.values[0],
            'best_emissions': best_emission_solution.fitness.values[1],
            'cost_improvement_percent': cost_improvement,
            'emission_improvement_percent': emission_improvement,
            'pareto_front_size': len(self.pareto_front)
        }
        
    def plot_pareto_front(self, save_path: str = None):
        """Plot Pareto front visualization."""
        if not self.pareto_front:
            logger.warning("No Pareto front available for plotting")
            return
            
        costs = [ind.fitness.values[0] for ind in self.pareto_front]
        emissions = [ind.fitness.values[1] for ind in self.pareto_front]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(costs, emissions, alpha=0.7, s=50, c='red', label='Pareto Front')
        
        # Add baseline point if available
        if self.original_routes:
            baseline_cost = sum(self._calculate_route_cost(route) for route in self.original_routes)
            baseline_emissions = sum(self._predict_route_emissions(route) for route in self.original_routes)
            plt.scatter([baseline_cost], [baseline_emissions], s=100, c='blue', 
                       marker='*', label='Baseline')
        
        plt.xlabel('Total Cost')
        plt.ylabel('Total Emissions (kg CO2e)')
        plt.title('Pareto Front: Cost vs Emissions Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pareto front plot saved to {save_path}")
            
        plt.show()
        
    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization convergence history."""
        if not self.optimization_history:
            logger.warning("No optimization history available")
            return
            
        generations = [stats['generation'] for stats in self.optimization_history]
        mean_costs = [stats['mean_cost'] for stats in self.optimization_history]
        mean_emissions = [stats['mean_emissions'] for stats in self.optimization_history]
        min_costs = [stats['min_cost'] for stats in self.optimization_history]
        min_emissions = [stats['min_emissions'] for stats in self.optimization_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Cost evolution
        ax1.plot(generations, mean_costs, label='Mean Cost', alpha=0.7)
        ax1.plot(generations, min_costs, label='Min Cost', alpha=0.7)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Cost')
        ax1.set_title('Cost Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Emissions evolution
        ax2.plot(generations, mean_emissions, label='Mean Emissions', alpha=0.7)
        ax2.plot(generations, min_emissions, label='Min Emissions', alpha=0.7)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Emissions (kg CO2e)')
        ax2.set_title('Emissions Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization history plot saved to {save_path}")
            
        plt.show()


def main():
    """
    Main function to demonstrate hybrid ML-GA optimization.
    """
    logger.info("Hybrid ML-GA optimization module initialized")
    
    # Example usage would go here when integrated with ML models and data


if __name__ == "__main__":
    main()
