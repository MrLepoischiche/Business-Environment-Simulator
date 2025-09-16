#!/bin/bash

# NORMA Integration Test Runner
# Comprehensive test script for validating NORMA compatibility

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="$SCRIPT_DIR/results_$TIMESTAMP"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log "Python version: $PYTHON_VERSION"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check virtual environment (recommended)
    if [[ -z "$VIRTUAL_ENV" ]]; then
        warning "Not running in a virtual environment. This is recommended."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        success "Running in virtual environment: $VIRTUAL_ENV"
    fi
}

# Install dependencies
install_dependencies() {
    log "Installing integration test dependencies..."
    
    # Install main project dependencies
    pip3 install -r "$PROJECT_ROOT/requirements.txt"
    
    # Install integration test specific dependencies
    if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
        pip3 install -r "$SCRIPT_DIR/requirements.txt"
    fi
    
    success "Dependencies installed successfully"
}

# Setup test environment
setup_environment() {
    log "Setting up test environment..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Set environment variables for testing
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    
    # Set demo API URLs if not already set
    export DEMO_API_URL="${DEMO_API_URL:-http://localhost:8000}"
    export AZURE_API_URL="${AZURE_API_URL:-https://banking-sim-azure.norma.dev}"
    export GCP_API_URL="${GCP_API_URL:-https://banking-sim-gcp.norma.dev}"
    export IONOS_API_URL="${IONOS_API_URL:-https://banking-sim-ionos.norma.dev}"
    
    # Create test configuration file
    cat > "$RESULTS_DIR/test_config.json" << EOF
{
    "test_run_id": "norma_integration_$TIMESTAMP",
    "timestamp": "$(date -Iseconds)",
    "environment": {
        "python_version": "$PYTHON_VERSION",
        "virtual_env": "${VIRTUAL_ENV:-none}",
        "working_directory": "$PROJECT_ROOT"
    },
    "api_endpoints": {
        "demo": "$DEMO_API_URL",
        "azure": "$AZURE_API_URL",
        "gcp": "$GCP_API_URL",
        "ionos": "$IONOS_API_URL"
    }
}
EOF
    
    success "Test environment configured"
}

# Start local API server if needed
start_local_api() {
    log "Checking if local API server is needed..."
    
    if curl -s "$DEMO_API_URL/health" > /dev/null 2>&1; then
        success "Local API server is already running at $DEMO_API_URL"
        return 0
    fi
    
    log "Starting local API server..."
    cd "$PROJECT_ROOT"
    
    # Start API server in background
    python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload > "$RESULTS_DIR/api_server.log" 2>&1 &
    API_PID=$!
    echo $API_PID > "$RESULTS_DIR/api_server.pid"
    
    # Wait for server to start
    for i in {1..30}; do
        if curl -s "$DEMO_API_URL/health" > /dev/null 2>&1; then
            success "Local API server started successfully (PID: $API_PID)"
            return 0
        fi
        sleep 1
    done
    
    error "Failed to start local API server within 30 seconds"
    return 1
}

# Run integration tests
run_tests() {
    log "Starting NORMA integration tests..."
    
    cd "$SCRIPT_DIR"
    
    local exit_code=0
    
    # Run the integration test suite
    if [[ "${TEST_FORMAT:-cli}" == "pytest" ]]; then
        log "Running tests with pytest..."
        python3 -m pytest test_norma_integration.py::TestNormaIntegrationPytest \
            -v \
            --tb=short \
            --html="$RESULTS_DIR/pytest_report.html" \
            --self-contained-html \
            --junitxml="$RESULTS_DIR/junit_results.xml" \
            --cov=. \
            --cov-report=html:"$RESULTS_DIR/coverage_html" \
            --cov-report=xml:"$RESULTS_DIR/coverage.xml" \
            > "$RESULTS_DIR/pytest_output.log" 2>&1
        exit_code=$?
    else
        log "Running tests with CLI runner..."
        python3 test_norma_integration.py --verbose > "$RESULTS_DIR/integration_output.log" 2>&1
        exit_code=$?
    fi
    
    # Copy any generated reports
    if [[ -f "norma_integration_report_*.json" ]]; then
        cp norma_integration_report_*.json "$RESULTS_DIR/"
    fi
    
    return $exit_code
}

# Generate summary report
generate_summary() {
    log "Generating test summary..."
    
    local exit_code=$1
    local status
    
    if [[ $exit_code -eq 0 ]]; then
        status="PASSED"
        status_color=$GREEN
    elif [[ $exit_code -eq 1 ]]; then
        status="PARTIAL_PASS"
        status_color=$YELLOW
    else
        status="FAILED"
        status_color=$RED
    fi
    
    # Create summary report
    cat > "$RESULTS_DIR/test_summary.md" << EOF
# NORMA Integration Test Summary

## Test Run Information
- **Run ID**: norma_integration_$TIMESTAMP
- **Date**: $(date -Iseconds)
- **Duration**: $((SECONDS / 60)) minutes $((SECONDS % 60)) seconds
- **Status**: $status
- **Exit Code**: $exit_code

## Test Environment
- **Python Version**: $PYTHON_VERSION
- **Virtual Environment**: ${VIRTUAL_ENV:-none}
- **Project Root**: $PROJECT_ROOT

## API Endpoints Tested
- **Demo**: $DEMO_API_URL
- **Azure**: $AZURE_API_URL
- **GCP**: $GCP_API_URL
- **Ionos**: $IONOS_API_URL

## Results Location
- **Results Directory**: $RESULTS_DIR
- **Detailed Logs**: Available in results directory
- **Integration Report**: JSON format with full details

## Status Interpretation
- **PASSED**: All integration tests passed successfully
- **PARTIAL_PASS**: Most tests passed, some issues to review
- **FAILED**: Critical integration issues found

## Next Steps
EOF
    
    if [[ $exit_code -eq 0 ]]; then
        cat >> "$RESULTS_DIR/test_summary.md" << EOF
✅ **Ready for NORMA Partnership**
- Schedule technical demo with NORMA team
- Proceed with pilot client integration
- Begin production deployment planning
EOF
    elif [[ $exit_code -eq 1 ]]; then
        cat >> "$RESULTS_DIR/test_summary.md" << EOF
⚠️ **Review Issues Before Demo**
- Address partial test failures
- Validate critical functionality
- Re-run tests after fixes
EOF
    else
        cat >> "$RESULTS_DIR/test_summary.md" << EOF
❌ **Address Critical Issues**
- Fix integration problems
- Validate core functionality
- Complete testing before demo
EOF
    fi
    
    # Display summary
    echo
    echo "================================================================"
    echo -e "                ${status_color}NORMA INTEGRATION TEST SUMMARY${NC}"
    echo "================================================================"
    echo -e "Status: ${status_color}$status${NC}"
    echo "Results: $RESULTS_DIR"
    echo "Duration: $((SECONDS / 60))m $((SECONDS % 60))s"
    echo "================================================================"
    echo
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Stop local API server if we started it
    if [[ -f "$RESULTS_DIR/api_server.pid" ]]; then
        local api_pid=$(cat "$RESULTS_DIR/api_server.pid")
        if kill -0 "$api_pid" 2>/dev/null; then
            log "Stopping local API server (PID: $api_pid)"
            kill "$api_pid"
            rm -f "$RESULTS_DIR/api_server.pid"
        fi
    fi
    
    # Archive results
    if [[ -d "$RESULTS_DIR" ]]; then
        log "Archiving results to $RESULTS_DIR.tar.gz"
        tar -czf "$RESULTS_DIR.tar.gz" -C "$(dirname "$RESULTS_DIR")" "$(basename "$RESULTS_DIR")"
    fi
}

# Main execution
main() {
    local start_time=$SECONDS
    
    log "Starting NORMA Integration Test Suite"
    log "Results will be saved to: $RESULTS_DIR"
    
    # Setup trap for cleanup
    trap cleanup EXIT
    
    # Run all steps
    check_prerequisites
    install_dependencies
    setup_environment
    
    # Start local API if needed (non-fatal if it fails)
    start_local_api || warning "Local API server not available, using remote endpoints only"
    
    # Run the actual tests
    local test_exit_code=0
    if run_tests; then
        success "Integration tests completed successfully"
    else
        test_exit_code=$?
        warning "Integration tests completed with issues (exit code: $test_exit_code)"
    fi
    
    # Generate summary
    generate_summary $test_exit_code
    
    return $test_exit_code
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --format)
            export TEST_FORMAT="$2"
            shift 2
            ;;
        --provider)
            export TEST_PROVIDER="$2"
            shift 2
            ;;
        --verbose|-v)
            export LOG_LEVEL="DEBUG"
            shift
            ;;
        --help|-h)
            echo "NORMA Integration Test Runner"
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --format cli|pytest    Test runner format (default: cli)"
            echo "  --provider PROVIDER     Test specific provider only"
            echo "  --verbose, -v          Enable verbose output"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  DEMO_API_URL           Demo API endpoint"
            echo "  AZURE_API_URL          Azure API endpoint"
            echo "  GCP_API_URL            GCP API endpoint"
            echo "  IONOS_API_URL          Ionos API endpoint"
            echo ""
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
    exit $?
fi