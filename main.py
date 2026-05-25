import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="AI Swarm Detection on Twitter (X) - CLI Orchestrator")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-pipeline', action='store_true', help="Run the modular training and evaluation pipeline (re-builds features, clusters, and fits classifier)")
    group.add_argument('--run-app', action='store_true', help="Start the interactive Streamlit dashboard web application")
    group.add_argument('--run-simple', action='store_true', help="Run the monolithic master pipeline execution script (useful for rapid proof of concept)")
    
    args = parser.parse_args()
    
    if args.run_pipeline:
        print("\n" + "="*60)
        print("Starting Modular AI Swarm Detection Pipeline")
        print("="*60 + "\n")
        # Run scripts/run_pipeline.py
        pipeline_script = os.path.join("scripts", "run_pipeline.py")
        if not os.path.exists(pipeline_script):
            print(f"[ERROR] Pipeline script not found at '{pipeline_script}'. Please ensure files are copied properly.")
            sys.exit(1)
        
        result = subprocess.run([sys.executable, pipeline_script])
        sys.exit(result.returncode)
        
    elif args.run_app:
        print("\n" + "="*60)
        print("Starting Streamlit Dashboard Web Application")
        print("="*60 + "\n")
        # Run the presentation-ready streamlit app
        app_script = os.path.join("app", "streamlit_app.py")
        if not os.path.exists(app_script):
            print(f"[ERROR] Streamlit app script not found at '{app_script}'. Please ensure files are copied properly.")
            sys.exit(1)
            
        result = subprocess.run(["streamlit", "run", app_script])
        sys.exit(result.returncode)
        
    elif args.run_simple:
        print("\n" + "="*60)
        print("Running Monolithic Master Pipeline")
        print("="*60 + "\n")
        # Run scripts/master_pipeline.py
        master_script = os.path.join("scripts", "master_pipeline.py")
        if not os.path.exists(master_script):
            print(f"[ERROR] Master pipeline script not found at '{master_script}'. Please ensure files are copied properly.")
            sys.exit(1)
            
        result = subprocess.run([sys.executable, master_script])
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
