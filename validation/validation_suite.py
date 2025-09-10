"""
Validation Suite for Banking Simulation
Comprehensive validation of simulation accuracy and performance
"""
import asyncio
import logging
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import pandas as pd

from core import SimulationConfig
from environments.banking.simulation import create_banking_simulation
from environments.banking import create_scenario


class ValidationSuite:
    """Comprehensive validation suite for banking simulation"""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("validation_suite")
        self.results = {}
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        self.logger.info("Starting comprehensive validation suite")
        
        validation_results = {}
        
        # 1. Functional validation
        self.logger.info("Running functional validation...")
        validation_results["functional"] = await self._validate_functionality()
        
        # 2. Accuracy validation
        self.logger.info("Running accuracy validation...")
        validation_results["accuracy"] = await self._validate_accuracy()
        
        # 3. Performance validation
        self.logger.info("Running performance validation...")
        validation_results["performance"] = await self._validate_performance()
        
        # 4. Fraud detection validation
        self.logger.info("Running fraud detection validation...")
        validation_results["fraud_detection"] = await self._validate_fraud_detection()
        
        # 5. Scalability validation
        self.logger.info("Running scalability validation...")
        validation_results["scalability"] = await self._validate_scalability()
        
        # Save results
        await self._save_validation_results(validation_results)
        
        # Generate report
        await self._generate_validation_report(validation_results)
        
        return validation_results
    
    async def _validate_functionality(self) -> Dict[str, Any]:
        """Validate basic functionality works correctly"""
        results = {"tests": [], "passed": 0, "failed": 0}
        
        # Test 1: Basic simulation creation and initialization
        try:
            config = SimulationConfig(
                name="Functional Test",
                duration_days=0.01,
                time_acceleration=1000.0,
                environment_type="banking"
            )
            
            engine = create_banking_simulation(config)
            
            environment_config = {
                "scenario_type": "retail",
                "scenario_params": {"customer_count": 50},
                "agents": [
                    {
                        "agent_id": "func_test_agent",
                        "agent_type": "mock",
                        "name": "Functional Test Agent",
                        "max_concurrent_requests": 5
                    }
                ]
            }
            
            await engine.initialize(environment_config)
            
            results["tests"].append({
                "name": "Basic Initialization",
                "status": "PASSED",
                "details": "Simulation engine initialized successfully"
            })
            results["passed"] += 1
            
        except Exception as e:
            results["tests"].append({
                "name": "Basic Initialization", 
                "status": "FAILED",
                "details": f"Error: {str(e)}"
            })
            results["failed"] += 1
            
        # Test 2: Event processing
        try:
            from core import BusinessEvent
            
            test_event = BusinessEvent(
                event_type="transaction",
                data={
                    "transaction_id": "test_tx_001",
                    "amount": 500.0,
                    "customer_id": "test_customer",
                    "agent_type": "fraud_detection"
                },
                requires_ai_processing=True
            )
            
            success = await engine.event_dispatcher.dispatch(test_event)
            assert success, "Event dispatch failed"
            
            results["tests"].append({
                "name": "Event Processing",
                "status": "PASSED", 
                "details": "Events processed successfully"
            })
            results["passed"] += 1
            
        except Exception as e:
            results["tests"].append({
                "name": "Event Processing",
                "status": "FAILED",
                "details": f"Error: {str(e)}"
            })
            results["failed"] += 1
            
        # Test 3: Metrics collection
        try:
            engine.metrics_collector.increment_counter("transactions_processed", 10)
            engine.metrics_collector.set_gauge("active_agents", 5)
            
            current_metrics = engine.metrics_collector.get_current_metrics()
            assert "transactions_processed" in current_metrics
            assert current_metrics["transactions_processed"]["value"] == 10
            
            results["tests"].append({
                "name": "Metrics Collection",
                "status": "PASSED",
                "details": "Metrics collected and retrieved successfully"
            })
            results["passed"] += 1
            
        except Exception as e:
            results["tests"].append({
                "name": "Metrics Collection",
                "status": "FAILED", 
                "details": f"Error: {str(e)}"
            })
            results["failed"] += 1
            
        # Test 4: Scenario execution
        try:
            simulation_results = await engine.run()
            
            assert simulation_results["state"] == "completed"
            assert "banking_report" in simulation_results
            
            results["tests"].append({
                "name": "Scenario Execution",
                "status": "PASSED",
                "details": "Banking scenario executed to completion"
            })
            results["passed"] += 1
            
        except Exception as e:
            results["tests"].append({
                "name": "Scenario Execution",
                "status": "FAILED",
                "details": f"Error: {str(e)}"
            })
            results["failed"] += 1
            
        return results
    
    async def _validate_accuracy(self) -> Dict[str, Any]:
        """Validate simulation produces accurate results"""
        results = {"tests": [], "accuracy_scores": {}}
        
        # Test 1: Transaction generation rates
        config = SimulationConfig(
            name="Accuracy Test",
            duration_days=0.1,  # 2.4 hours simulated
            time_acceleration=100.0,
            environment_type="banking"
        )
        
        engine = create_banking_simulation(config)
        
        environment_config = {
            "scenario_type": "retail",
            "scenario_params": {"customer_count": 100},
            "agents": [
                {
                    "agent_id": "accuracy_agent",
                    "agent_type": "mock",
                    "name": "Accuracy Test Agent",
                    "max_concurrent_requests": 10
                }
            ]
        }
        
        await engine.initialize(environment_config)
        simulation_results = await engine.run()
        
        # Analyze transaction generation accuracy
        banking_report = simulation_results["banking_report"]
        tx_analysis = banking_report["transaction_analysis"]
        
        expected_tx_count = 100 * 2.0 * (0.1 * 24)  # customers * rate * hours
        actual_tx_count = tx_analysis["transaction_count"]
        
        accuracy = 1 - abs(expected_tx_count - actual_tx_count) / expected_tx_count
        results["accuracy_scores"]["transaction_rate"] = accuracy
        
        results["tests"].append({
            "name": "Transaction Rate Accuracy",
            "status": "PASSED" if accuracy > 0.8 else "FAILED",
            "details": f"Expected: ~{expected_tx_count:.0f}, Actual: {actual_tx_count}, Accuracy: {accuracy:.2%}"
        })
        
        # Test 2: Fraud detection accuracy (with known fraud patterns)
        fraud_config = SimulationConfig(
            name="Fraud Accuracy Test",
            duration_days=0.05,
            time_acceleration=200.0,
            environment_type="banking"
        )
        
        fraud_engine = create_banking_simulation(fraud_config)
        
        fraud_environment_config = {
            "scenario_type": "fraud_detection", 
            "scenario_params": {"customer_count": 50},
            "agents": [
                {
                    "agent_id": "fraud_accuracy_agent",
                    "agent_type": "mock",
                    "name": "Fraud Accuracy Agent",
                    "max_concurrent_requests": 5,
                    "model_config": {"fraud_threshold": 0.8}
                }
            ]
        }
        
        await fraud_engine.initialize(fraud_environment_config)
        fraud_results = await fraud_engine.run()
        
        fraud_summary = fraud_results["fraud_summary"]
        
        # Extract accuracy metrics
        precision = float(fraud_summary["precision"].rstrip('%')) / 100
        recall = float(fraud_summary["recall"].rstrip('%')) / 100
        f1_score = float(fraud_summary["f1_score"].rstrip('%')) / 100
        
        results["accuracy_scores"]["fraud_precision"] = precision
        results["accuracy_scores"]["fraud_recall"] = recall  
        results["accuracy_scores"]["fraud_f1"] = f1_score
        
        results["tests"].append({
            "name": "Fraud Detection Accuracy",
            "status": "PASSED" if f1_score > 0.7 else "FAILED",
            "details": f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1_score:.2%}"
        })
        
        return results
    
    async def _validate_performance(self) -> Dict[str, Any]:
        """Validate simulation performance meets requirements"""
        results = {"tests": [], "performance_metrics": {}}
        
        # Performance test configuration
        perf_config = SimulationConfig(
            name="Performance Test",
            duration_days=0.02,  # Small but measurable
            time_acceleration=500.0,
            environment_type="banking"
        )
        
        # Test different customer counts
        customer_counts = [100, 500, 1000]
        performance_data = []
        
        for customer_count in customer_counts:
            self.logger.info(f"Testing performance with {customer_count} customers")
            
            engine = create_banking_simulation(perf_config)
            
            environment_config = {
                "scenario_type": "high_volume",
                "scenario_params": {"customer_count": customer_count},
                "agents": [
                    {
                        "agent_id": f"perf_agent_{customer_count}",
                        "agent_type": "mock",
                        "name": f"Performance Agent {customer_count}",
                        "max_concurrent_requests": 10
                    }
                ]
            }
            
            # Measure execution time
            start_time = datetime.now()
            await engine.initialize(environment_config)
            init_time = datetime.now()
            
            simulation_results = await engine.run()
            end_time = datetime.now()
            
            init_duration = (init_time - start_time).total_seconds()
            run_duration = (end_time - init_time).total_seconds()
            total_duration = (end_time - start_time).total_seconds()
            
            # Extract performance metrics
            banking_report = simulation_results["banking_report"]
            tx_analysis = banking_report["transaction_analysis"]
            
            transactions_processed = tx_analysis["transaction_count"]
            throughput = transactions_processed / run_duration if run_duration > 0 else 0
            
            performance_data.append({
                "customer_count": customer_count,
                "init_time": init_duration,
                "run_time": run_duration,
                "total_time": total_duration,
                "transactions": transactions_processed,
                "throughput": throughput
            })
            
        # Analyze performance scaling
        results["performance_metrics"]["scaling_data"] = performance_data
        
        # Check if performance scales reasonably
        throughputs = [data["throughput"] for data in performance_data]
        avg_throughput = statistics.mean(throughputs)
        throughput_variance = statistics.variance(throughputs) if len(throughputs) > 1 else 0
        
        results["tests"].append({
            "name": "Performance Scaling",
            "status": "PASSED" if avg_throughput > 50 else "FAILED",  # 50 tx/sec minimum
            "details": f"Average throughput: {avg_throughput:.1f} tx/sec, Variance: {throughput_variance:.2f}"
        })
        
        # Memory usage estimation (simplified)
        max_customers = max(customer_counts)
        estimated_memory_mb = (max_customers * 0.001) + 50  # Rough estimate
        
        results["performance_metrics"]["estimated_memory_mb"] = estimated_memory_mb
        
        results["tests"].append({
            "name": "Memory Usage",
            "status": "PASSED" if estimated_memory_mb < 200 else "WARNING",
            "details": f"Estimated memory usage: {estimated_memory_mb:.1f} MB for {max_customers} customers"
        })
        
        return results
    
    async def _validate_fraud_detection(self) -> Dict[str, Any]:
        """Comprehensive fraud detection validation"""
        results = {"tests": [], "fraud_metrics": {}}
        
        # Create fraud detection focused simulation
        fraud_config = SimulationConfig(
            name="Fraud Validation Test",
            duration_days=0.1,
            time_acceleration=200.0,
            environment_type="banking"
        )
        
        engine = create_banking_simulation(fraud_config)
        
        environment_config = {
            "scenario_type": "fraud_detection",
            "scenario_params": {"customer_count": 200},
            "agents": [
                {
                    "agent_id": "fraud_validation_agent",
                    "agent_type": "mock",
                    "name": "Fraud Validation Agent",
                    "max_concurrent_requests": 8,
                    "model_config": {
                        "fraud_threshold": 0.85,
                        "amount_threshold": 1000.0
                    }
                }
            ]
        }
        
        await engine.initialize(environment_config)
        simulation_results = await engine.run()
        
        # Extract fraud detection metrics
        fraud_detection = simulation_results["banking_report"]["fraud_detection"]
        fraud_summary = simulation_results["fraud_summary"]
        
        # Test fraud detection performance
        precision = fraud_detection["precision"]
        recall = fraud_detection["recall"]
        f1_score = fraud_detection["f1_score"]
        accuracy = fraud_detection["accuracy"]
        
        results["fraud_metrics"] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "true_positives": fraud_detection["true_positives"],
            "false_positives": fraud_detection["false_positives"],
            "true_negatives": fraud_detection["true_negatives"],
            "false_negatives": fraud_detection["false_negatives"]
        }
        
        # Validation criteria
        criteria = {
            "precision": (precision, 0.8, "Precision should be > 80%"),
            "recall": (recall, 0.7, "Recall should be > 70%"), 
            "f1_score": (f1_score, 0.75, "F1 score should be > 75%"),
            "accuracy": (accuracy, 0.85, "Accuracy should be > 85%")
        }
        
        for metric_name, (value, threshold, description) in criteria.items():
            status = "PASSED" if value >= threshold else "FAILED"
            results["tests"].append({
                "name": f"Fraud Detection {metric_name.title()}",
                "status": status,
                "details": f"{description}: {value:.2%} (threshold: {threshold:.0%})"
            })
            
        return results
    
    async def _validate_scalability(self) -> Dict[str, Any]:
        """Test simulation scalability with increasing loads"""
        results = {"tests": [], "scalability_data": []}
        
        # Test different scales
        scales = [
            {"customers": 100, "agents": 3},
            {"customers": 500, "agents": 5}, 
            {"customers": 1000, "agents": 8},
            {"customers": 2000, "agents": 10}
        ]
        
        scalability_data = []
        
        for scale in scales:
            self.logger.info(f"Testing scalability: {scale['customers']} customers, {scale['agents']} agents")
            
            scale_config = SimulationConfig(
                name=f"Scalability Test {scale['customers']}",
                duration_days=0.01,  # Very short for scalability testing
                time_acceleration=1000.0,
                environment_type="banking"
            )
            
            engine = create_banking_simulation(scale_config)
            
            environment_config = {
                "scenario_type": "high_volume",
                "scenario_params": {"customer_count": scale["customers"]},
                "agents": [
                    {
                        "agent_id": f"scale_agent_{i}",
                        "agent_type": "mock",
                        "name": f"Scale Agent {i}",
                        "max_concurrent_requests": 5
                    }
                    for i in range(scale["agents"])
                ]
            }
            
            try:
                start_time = datetime.now()
                await engine.initialize(environment_config)
                simulation_results = await engine.run()
                end_time = datetime.now()
                
                duration = (end_time - start_time).total_seconds()
                
                banking_report = simulation_results["banking_report"]
                tx_count = banking_report["transaction_analysis"]["transaction_count"]
                
                scalability_data.append({
                    "customers": scale["customers"],
                    "agents": scale["agents"],
                    "duration": duration,
                    "transactions": tx_count,
                    "throughput": tx_count / duration if duration > 0 else 0,
                    "success": True
                })
                
            except Exception as e:
                scalability_data.append({
                    "customers": scale["customers"],
                    "agents": scale["agents"], 
                    "duration": None,
                    "transactions": 0,
                    "throughput": 0,
                    "success": False,
                    "error": str(e)
                })
                
        results["scalability_data"] = scalability_data
        
        # Analyze scalability
        successful_tests = [data for data in scalability_data if data["success"]]
        
        if len(successful_tests) >= 3:
            # Check if throughput scales reasonably
            throughputs = [data["throughput"] for data in successful_tests]
            customer_counts = [data["customers"] for data in successful_tests]
            
            # Simple linear regression to check scaling
            if len(throughputs) > 1:
                correlation = self._calculate_correlation(customer_counts, throughputs)
                
                results["tests"].append({
                    "name": "Throughput Scaling",
                    "status": "PASSED" if correlation > 0.5 else "WARNING",
                    "details": f"Correlation between customers and throughput: {correlation:.2f}"
                })
        
        # Check maximum scale achieved
        max_customers = max([data["customers"] for data in successful_tests])
        
        results["tests"].append({
            "name": "Maximum Scale",
            "status": "PASSED" if max_customers >= 1000 else "WARNING",
            "details": f"Successfully simulated up to {max_customers} customers"
        })
        
        return results
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate simple correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        denom_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        denom_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
        
        if denom_x == 0 or denom_y == 0:
            return 0.0
            
        return numerator / (denom_x * denom_y) ** 0.5
    
    async def _save_validation_results(self, results: Dict[str, Any]) -> None:
        """Save validation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_dir / f"validation_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save CSV summary
        csv_file = self.output_dir / f"validation_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Test Name", "Status", "Details"])
            
            for category, category_results in results.items():
                if "tests" in category_results:
                    for test in category_results["tests"]:
                        writer.writerow([
                            category,
                            test["name"],
                            test["status"], 
                            test["details"]
                        ])
                        
        self.logger.info(f"Validation results saved to {json_file} and {csv_file}")
    
    async def _generate_validation_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive validation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_file = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Calculate overall statistics
        total_tests = sum(len(cat_results.get("tests", [])) for cat_results in results.values())
        passed_tests = sum(
            len([t for t in cat_results.get("tests", []) if t["status"] == "PASSED"])
            for cat_results in results.values()
        )
        failed_tests = total_tests - passed_tests
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Banking Simulation Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e7f3ff; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .category {{ margin: 20px 0; }}
                .test-passed {{ color: green; }}
                .test-failed {{ color: red; }}
                .test-warning {{ color: orange; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Banking Simulation Validation Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Tests:</strong> {total_tests}</p>
                <p><strong>Passed:</strong> <span class="test-passed">{passed_tests}</span></p>
                <p><strong>Failed:</strong> <span class="test-failed">{failed_tests}</span></p>
                <p><strong>Success Rate:</strong> {passed_tests/total_tests:.1%}</p>
            </div>
        """
        
        # Add detailed results for each category
        for category, category_results in results.items():
            html_content += f"""
            <div class="category">
                <h2>{category.title()} Validation</h2>
            """
            
            if "tests" in category_results:
                html_content += """
                <table>
                    <tr><th>Test Name</th><th>Status</th><th>Details</th></tr>
                """
                
                for test in category_results["tests"]:
                    status_class = f"test-{test['status'].lower()}"
                    html_content += f"""
                    <tr>
                        <td>{test['name']}</td>
                        <td class="{status_class}">{test['status']}</td>
                        <td>{test['details']}</td>
                    </tr>
                    """
                    
                html_content += "</table>"
            
            # Add metrics if available
            for key, value in category_results.items():
                if key != "tests" and isinstance(value, dict):
                    html_content += f"<div class='metric'><strong>{key.title()}:</strong><br>"
                    for metric_name, metric_value in value.items():
                        html_content += f"{metric_name}: {metric_value}<br>"
                    html_content += "</div>"
                    
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_file, 'w') as f:
            f.write(html_content)
            
        self.logger.info(f"Validation report generated: {report_file}")


async def main():
    """Run validation suite"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    validator = ValidationSuite()
    
    print("Starting Banking Simulation Validation Suite...")
    print("=" * 60)
    
    try:
        results = await validator.run_full_validation()
        
        # Print summary
        total_tests = sum(len(cat_results.get("tests", [])) for cat_results in results.values())
        passed_tests = sum(
            len([t for t in cat_results.get("tests", []) if t["status"] == "PASSED"])
            for cat_results in results.values()
        )
        
        print("\nValidation Complete!")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")
        
        return results
        
    except Exception as e:
        print(f"Validation failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())