#!/usr/bin/env python3
"""
Command-line interface for the Investor Relations Agent System.

Usage:
    python cli.py
    
This will start an interactive CLI where you can enter company names
and get investment-related information.
"""

import os
import sys
import json
import logging
from dotenv import load_dotenv

# Import the main system class
# Assuming the main file is named investor_relations_agent_system.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from investor_relations_agent_system import InvestorRelationsAgentSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ir_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def display_qa(qa_list):
    """Display questions and answers in a formatted way."""
    for i, qa in enumerate(qa_list):
        print(f"\n{'='*80}")
        print(f"Question {i+1}: {qa['question']}")
        print(f"{'='*80}")
        print(f"Answer: {qa['answer']}")
        print(f"{'='*80}")

def handle_disambiguation(ir_system, result):
    """Handle disambiguation when multiple potential companies are found."""
    print("\nMultiple companies match your query. Please select one:")
    for i, option in enumerate(result["options"]):
        print(f"{i+1}. {option}")
    
    while True:
        try:
            choice = int(input("\nEnter the number of your selection: "))
            if 1 <= choice <= len(result["options"]):
                selected_option = result["options"][choice-1]
                print(f"Selected: {selected_option}")
                return ir_system.resolve_disambiguation(selected_option)
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")

def save_results(result, company_name):
    """Save the results to a JSON file."""
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Create a sanitized filename from the company name
    filename = company_name.lower().replace(" ", "_").replace(".", "").replace(",", "")
    output_path = f"output/{filename}_analysis.json"
    
    # Save the results
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    return output_path

def main():
    """Main CLI function."""
    # Load environment variables
    load_dotenv()
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please create a .env file with your OpenAI API key")
        return
    
    # Initialize the agent system
    print("\n" + "="*80)
    print("Investor Relations Agent System".center(80))
    print("="*80)
    print("Initializing system...")
    
    try:
        ir_system = InvestorRelationsAgentSystem()
        print("System ready!\n")
        
        while True:
            query = input("\nEnter a company name or query (or 'exit' to quit): ")
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            if not query.strip():
                continue
                
            print("\nProcessing your query. This may take a few moments...\n")
            
            try:
                result = ir_system.process_query(query)
                
                # Handle disambiguation
                if result["status"] == "disambiguation_needed":
                    result = handle_disambiguation(ir_system, result)
                
                # Handle errors
                if result["status"] == "error":
                    print(f"Error: {result.get('message', 'Unknown error')}")
                    continue
                
                # Display successful results
                if result["status"] == "success":
                    company_name = result["company_name"]
                    print(f"\nAnalysis for {company_name}:")
                    
                    # Show verification info
                    verify_info = result.get("company_info", {})
                    if verify_info:
                        print(f"\nCompany: {company_name}")
                        print(f"Listed: {verify_info.get('is_listed', 'Unknown')}")
                        if verify_info.get("is_listed"):
                            print(f"Primary Exchange: {verify_info.get('primary_exchange', 'Unknown')}")
                    
                    # Display Q&A
                    qa_list = result.get("questions_and_answers", [])
                    if qa_list:
                        display_qa(qa_list)
                    else:
                        print("No questions and answers generated.")
                    
                    # Save results
                    save_results(result, company_name)
                    
            except Exception as e:
                logger.exception("Error processing query")
                print(f"An error occurred: {str(e)}")
                
    except Exception as e:
        logger.exception("Error initializing system")
        print(f"Error initializing the system: {str(e)}")

if __name__ == "__main__":
    main()