#!/usr/bin/env python
import os
import sys

# Import your specific Crew class
# Note: If your crew.py is inside a specific folder (e.g., src/contract_crew/), 
# adjust this import to match your project structure (e.g., from contract_crew.crew import ContractAnalysisCrew)
from legal_team.crew import LegalTeam

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


# Create output directory if it doesn't exist to store the final report
os.makedirs('output', exist_ok=True)

def run():
    """
    Run the Contract Analysis crew.
    """
    # Define the inputs required by your tasks.yaml
    # We explicitly defined {file_path} in the Extractor Agent's task description
    inputs = {
        'file_path': 'D:\\Projects\\Automated-Contract-Risk-Analysis\\SampleContract.pdf' # Replace this with the actual path to the PDF you want to test
    }

    print(f"Starting Multi-Agent Contract Risk Analysis on: {inputs['file_path']}...\n")

    # Create and run the crew
    try:
        # Kickoff the sequential process
        result = LegalTeam().crew().kickoff(inputs=inputs)

        # Print the final result to the console
        print("\n\n=== FINAL CONTRACT RISK REPORT ===\n\n")
        print(result.raw)

        # Confirm where the markdown file was saved
        print("\n\nReport has been successfully saved to output/final_contract_risk_report.md")
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        print("Please ensure your API keys (GROQ_API_KEY, PINECONE_API_KEY, LLAMAPARSE_API_KEY) are set in your environment variables.")

if __name__ == "__main__":
    run()