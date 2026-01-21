#!/usr/bin/env python3
"""
Test runner for plot testing module.
This script runs the tests for plotting functions.
"""

import sys
import os
import subprocess
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_pytest_tests():
    """Run tests using pytest."""
    print("Running tests with pytest...")
    
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pytest on the test directory
    cmd = [sys.executable, "-m", "pytest", test_dir, "-v"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"\nReturn code: {result.returncode}")
        return result.returncode == 0
        
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest: pip install pytest")
        return False
    except Exception as e:
        print(f"Error running pytest: {e}")
        return False

def run_direct_tests():
    """Run tests directly without pytest."""
    print("Running tests directly...")
    
    try:
        # Import and run the tests
        from test_plot_prediction_time_series import (
            TestPlotPredictionTimeSeries,
            TestPlotUtilities,
            test_plot_save_functionality
        )
        
        # Create test instances
        test_plot = TestPlotPredictionTimeSeries()
        test_plot.setup_method()
        
        test_utils = TestPlotUtilities()
        
        # Run tests
        tests_passed = 0
        tests_failed = 0
        
        # Test basic plotting
        try:
            test_plot.test_plot_time_series_with_physics_basic()
            print("✓ test_plot_time_series_with_physics_basic passed")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_plot_time_series_with_physics_basic failed: {e}")
            tests_failed += 1
        
        # Test MAE calculation
        try:
            test_plot.test_plot_time_series_with_physics_mae_calculation()
            print("✓ test_plot_time_series_with_physics_mae_calculation passed")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_plot_time_series_with_physics_mae_calculation failed: {e}")
            tests_failed += 1
        
        # Test regime coloring
        try:
            test_plot.test_plot_time_series_with_physics_regime_coloring()
            print("✓ test_plot_time_series_with_physics_regime_coloring passed")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_plot_time_series_with_physics_regime_coloring failed: {e}")
            tests_failed += 1
        
        # Test legend
        try:
            test_plot.test_plot_time_series_with_physics_legend()
            print("✓ test_plot_time_series_with_physics_legend passed")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_plot_time_series_with_physics_legend failed: {e}")
            tests_failed += 1
        
        # Test allowed input types
        try:
            test_plot.test_allowed_input_types()
            print("✓ test_allowed_input_types passed")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_allowed_input_types failed: {e}")
            tests_failed += 1
        
        # Test utility functions
        try:
            test_utils.test_mae_calculation()
            print("✓ test_mae_calculation passed")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_mae_calculation failed: {e}")
            tests_failed += 1
        
        try:
            test_utils.test_bdi_regime_classification()
            print("✓ test_bdi_regime_classification passed")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_bdi_regime_classification failed: {e}")
            tests_failed += 1
        
        try:
            test_utils.test_regime_transition_detection()
            print("✓ test_regime_transition_detection passed")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_regime_transition_detection failed: {e}")
            tests_failed += 1
        
        # Test plot save functionality
        try:
            test_plot_save_functionality()
            print("✓ test_plot_save_functionality passed")
            tests_passed += 1
        except Exception as e:
            print(f"✗ test_plot_save_functionality failed: {e}")
            tests_failed += 1
        
        print(f"\nTest results: {tests_passed} passed, {tests_failed} failed")
        
        return tests_failed == 0
        
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        print("Make sure you have the required dependencies installed.")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        ('numpy', 'np'),
        ('matplotlib', 'plt'),
        ('pytest', 'pytest')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(package_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"✗ {package_name} is NOT installed")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    print("\nAll dependencies are installed.")
    return True

def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description='Run plot function tests')
    parser.add_argument('--method', choices=['pytest', 'direct', 'both'], 
                       default='both', help='Test method to use')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies before running tests')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Plot Function Test Runner")
    print("=" * 60)
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            print("\nDependencies check failed. Exiting.")
            return 1
    
    success = True
    
    # Run tests based on selected method
    if args.method in ['pytest', 'both']:
        print("\n" + "=" * 60)
        print("Running tests with pytest...")
        print("=" * 60)
        if not run_pytest_tests():
            success = False
    
    if args.method in ['direct', 'both']:
        print("\n" + "=" * 60)
        print("Running tests directly...")
        print("=" * 60)
        if not run_direct_tests():
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("All tests passed successfully! ✓")
        return 0
    else:
        print("Some tests failed. ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())
