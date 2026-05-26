#!/usr/bin/env python
import os
import glob
import time
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

# Import your Crew
from legal_team.crew import LegalTeam

load_dotenv()

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Define directories
CONTRACTS_DIR = "/home/sherry/Projects/Automated-Contract-Risk-Analysis/SampleContracts"
OUTPUT_DIR = "/home/sherry/Projects/Automated-Contract-Risk-Analysis/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. RATE-LIMIT PROTECTED EXECUTION
# ==========================================
# @retry will automatically pause and retry if the API throws a rate limit error
@retry(wait=wait_exponential(multiplier=1, min=5, max=30), stop=stop_after_attempt(3))
def run_crew_with_retry(file_path):
    """Executes the CrewAI pipeline with exponential backoff for rate limits."""
    crew_instance = LegalTeam().crew()
    result = crew_instance.kickoff(inputs={'file_path': file_path})
    return result

# ==========================================
# 3. MAIN BATCH EVALUATION LOOP
# ==========================================
def run_evaluation_batch():
    # Find all PDFs in the target folder
    pdf_files = glob.glob(os.path.join(CONTRACTS_DIR, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {CONTRACTS_DIR}/ directory. Please add some test files.")
        return

    print(f"Found {len(pdf_files)} contracts. Starting Quantitative Batch Evaluation...\n")
    
    evaluation_results = []

    for index, file_path in enumerate(pdf_files):
        filename = os.path.basename(file_path)
        print(f"[{index + 1}/{len(pdf_files)}] Analyzing: {filename}...")
        
        start_time = time.time()
        
        try:
            # 1. Run the Multi-Agent Crew
            result = run_crew_with_retry(file_path)
            
            # 2. Calculate Execution Time
            end_time = time.time()
            time_taken = round(end_time - start_time, 2)
            
            # 3. Extract Token Usage
            try:
                total_tokens = result.token_usage.total_tokens
                prompt_tokens = result.token_usage.prompt_tokens
                completion_tokens = result.token_usage.completion_tokens
            except AttributeError:
                total_tokens, prompt_tokens, completion_tokens = "N/A", "N/A", "N/A"
            
            # 4. Append Metrics to our tracking list
            evaluation_results.append({
                "Filename": filename,
                "Status": "Success",
                "Time_Taken_Seconds": time_taken,
                "Total_Tokens": total_tokens,
                "Prompt_Tokens": prompt_tokens,
                "Completion_Tokens": completion_tokens
            })
            
            print(f"    -> Success! Crew finished in {time_taken} seconds (Total Tokens: {total_tokens}).")
            
        except Exception as e:
            print(f"    [!] Error processing {filename}: {e}")
            evaluation_results.append({
                "Filename": filename,
                "Status": f"Failed: {str(e)}",
                "Time_Taken_Seconds": None,
                "Total_Tokens": None,
                "Prompt_Tokens": None,
                "Completion_Tokens": None
            })

        # 5. COOLDOWN (Adjust based on your specific Cerebras tier limits)
        if index < len(pdf_files) - 1:
            cooldown_seconds = 15 
            print(f"    -> Cooling down for {cooldown_seconds} seconds to respect API limits...\n")
            time.sleep(cooldown_seconds)

    # ==========================================
    # 4. EXPORT TO CSV
    # ==========================================
    print("\n" + "="*50)
    print("Batch Evaluation Complete! Exporting to CSV...")
    
    df = pd.DataFrame(evaluation_results)
    csv_path = os.path.join(OUTPUT_DIR, "system_evaluation_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Evaluation metrics successfully saved to: {csv_path}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation_batch()