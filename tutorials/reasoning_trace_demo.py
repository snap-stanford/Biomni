#!/usr/bin/env python3
"""
Reasoning Trace Demonstration
============================

This script demonstrates the reasoning trace functionality with example queries
that showcase biomni's ability to perform complex reasoning and generate
insightful visualizations.

The queries are designed to:
- Demonstrate multi-step reasoning processes
- Generate visualizations and plots
- Show comprehensive reasoning capabilities
- Provide examples for different use cases
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Set matplotlib backend to avoid GUI issues
os.environ['MPLBACKEND'] = 'Agg'

# Add the biomni package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from biomni.agent.a1_with_trace import A1WithTrace

# Load environment variables
load_dotenv()

# LLM configuration - using biomni tutorial style
# Users should configure their own LLM settings as needed

# Example queries designed to demonstrate reasoning capabilities
example_queries = [
    # Query 1: PK/PD modeling example
    {
        "query": "Create a PK/PD model for bepirovirsen (GSK3228836) in chronic HBV infection using the following steps:\n\n1. First, create and save a plot showing plasma concentration over 12 weeks for 150mg vs 300mg weekly subcutaneous doses. Use literature values: half-life=28 days, volume of distribution=12L.\n\n2. Next, create and save a plot of HBV DNA suppression over time, using an Imax model with IC50=2.5 mg/L and baseline viral load of 5.5 log10 IU/mL.\n\n3. Finally, create and save a plot showing predicted HBsAg levels over time, assuming baseline=3.5 log10 IU/mL and an exponential relationship between drug concentration and antigen reduction.\n\nUse plt.savefig() to save each plot - it will automatically save to the query folder.",
        "expected_tools": ["scipy", "matplotlib", "numpy"],
        "expected_datasets": [],
        "complexity": "High - requires PK/PD modeling, viral dynamics, and biomarker prediction"
    },
    
    # Query 2: Gene regulatory network simulation
    {
        "query": "Simulate a simple 3-gene regulatory network with the following interactions:\n\n1. Gene A activates Gene B (activation constant = 0.8)\n2. Gene B inhibits Gene C (inhibition constant = 0.6)\n3. Gene C activates Gene A (activation constant = 0.7)\n\nCreate and save a plot showing the dynamics of all three genes over 50 time units, starting with initial concentrations [A=0.1, B=0.2, C=0.3]. Use a simple ODE model with Hill functions for the regulatory interactions.\n\nThen create and save a phase portrait showing the relationship between Gene A and Gene B concentrations over time.\n\nUse plt.savefig() to save each plot - it will automatically save to the query folder.",
        "expected_tools": ["scipy", "matplotlib", "numpy"],
        "expected_datasets": [],
        "complexity": "Moderate - requires ODE modeling and phase space analysis"
    },
    
    # Query 3: Cell population dynamics simulation
    {
        "query": "Simulate the growth dynamics of a cancer cell population under drug treatment:\n\n1. Model exponential growth of untreated cells (growth rate = 0.1 per day)\n2. Add drug treatment starting at day 10 with a cytotoxic effect (kill rate = 0.05 per day)\n3. Include drug resistance development (resistance rate = 0.01 per day)\n\nCreate and save a plot showing total cell count over 30 days, with separate curves for sensitive and resistant populations.\n\nThen create and save a plot showing the drug concentration over time, assuming first-order elimination (half-life = 8 hours) and daily dosing.\n\nUse plt.savefig() to save each plot - it will automatically save to the query folder.",
        "expected_tools": ["scipy", "matplotlib", "numpy"],
        "expected_datasets": [],
        "complexity": "Moderate - requires population dynamics and pharmacokinetics"
    }
]

def main():
    """Run reasoning trace demonstration queries."""
    
    print("🧬 Reasoning Trace Demonstration")
    print("=" * 70)
    print("This script demonstrates biomni's reasoning trace capabilities")
    print("with example queries that showcase complex reasoning processes.")
    print()
    
    # Initialize the agent with trace reporting
    agent = A1WithTrace(
        path="./biomni_data",  # Use biomni data directory
        llm="claude-sonnet-4-20250514",  # Use Claude Sonnet 4
        enable_trace_reporting=True,
        trace_output_dir="evaluation_results/reasoning_trace_demo",
        timeout_seconds=600
    )
    
    print(f"\n📋 Running {len(example_queries)} example queries...")
    print("These queries demonstrate reasoning trace functionality.")
    
    for i, query_info in enumerate(example_queries, 1):
        query = query_info["query"]
        expected_tools = query_info["expected_tools"]
        expected_datasets = query_info["expected_datasets"]
        complexity = query_info["complexity"]
        
        print(f"\n🔍 Query {i}: {complexity}")
        print("-" * 70)
        print(f"Query: {query}")
        print(f"Expected tools: {', '.join(expected_tools)}")
        print(f"Expected datasets: {', '.join(expected_datasets)}")
        print("=" * 70)
        
        try:
            # Execute the query with trace reporting
            log, final_result = agent.go(query)
            
            # Set the final result for the final user report
            agent.set_final_result(final_result)
            
            # Generate the detailed reasoning trace report
            report_path = agent.generate_trace_report()
            print(f"✅ Generated detailed reasoning trace: {report_path}")
            
            # Generate the final user report
            final_report_path = agent.generate_final_user_report()
            print(f"📋 Generated final user report: {final_report_path}")
            
            # Save complete terminal output
            terminal_output_path = agent.save_complete_terminal_output()
            print(f"📝 Saved complete terminal output: {terminal_output_path}")
            
            # Get trace analysis
            analysis = agent.analyze_tool_usage_patterns()
            print(f"📊 Analysis:")
            print(f"   - Total steps: {len(analysis.get('reasoning_steps_breakdown', {}))}")
            print(f"   - Code executions: {analysis.get('code_execution_frequency', {}).get('generated', 0)}")
            
            # Export trace data for further analysis
            json_path = agent.export_trace_data("json")
            print(f"📁 Exported trace data: {json_path}")
            
            print(f"\n🎯 Final result preview: {final_result[:200]}...")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            continue
        
        print("\n" + "="*70)
    
    print(f"\n🎯 All reasoning trace demo reports saved to: {agent.trace_reporter.output_dir}")
    print("📖 Open the HTML files in your browser to view detailed reasoning traces!")
    print("\n💡 These reports will show:")
    print("   - Complete reasoning process for each query")
    print("   - Code generation and execution for complex analyses")
    print("   - Results interpretation and conclusions")
    print("   - Generated visualizations and plots")

def interactive_demo_mode():
    """Interactive mode for testing custom reasoning trace queries."""
    
    print("\n🎮 Interactive Reasoning Trace Demo Mode")
    print("Enter your queries (type 'quit' to exit):")
    print("💡 Try queries that involve:")
    print("   - Multi-step reasoning processes")
    print("   - Data analysis and visualization")
    print("   - Complex problem solving")
    print("   - Code generation and execution")
    
    agent = A1WithTrace(
        path="./biomni_data",  # Use biomni data directory
        llm="claude-sonnet-4-20250514",  # Use Claude Sonnet 4
        enable_trace_reporting=True,
        trace_output_dir="evaluation_results/reasoning_trace_demo",
        timeout_seconds=600
    )
    
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\n🚀 Processing query: {query[:100]}...")
            
            # Execute the query
            log, final_result = agent.go(query)
            
            # Generate reports
            agent.set_final_result(final_result)
            report_path = agent.generate_trace_report()
            final_report_path = agent.generate_final_user_report()
            terminal_output_path = agent.save_complete_terminal_output()
            json_path = agent.export_trace_data("json")
            
            print(f"✅ Reports generated:")
            print(f"   - Detailed trace: {report_path}")
            print(f"   - Final report: {final_report_path}")
            print(f"   - Terminal output: {terminal_output_path}")
            print(f"   - Trace data: {json_path}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo_mode()
    else:
        main()
