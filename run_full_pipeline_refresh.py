import sys
import os
import subprocess

print("--- STARTING FULL PIPELINE REFRESH (data changed) ---")

scripts = [
    ("Phase 1: Internal Metric Audit", "final_audit_check.py"),
    ("Phase 3: Cluster Evaluation", "eval_mape_cluster.py"),
    ("Phase 4: Recursive Analysis", "analyze_recursive_mape.py"),
    ("Phase 5: Real-World Validation", "validate_full_clean_list.py"),
    ("Fairness: Macro-MAPE Audit", "eval_mape_internal.py")
]

results = {}

for name, script in scripts:
    print(f"\n>>> Running {name} ({script})...")
    try:
        # We assume the patch_expanded_whitelist.py ALREADY RAN in previous turn.
        # But Phase 3 Cluster needs to reload the new data.
        # Check if script exists
        if not os.path.exists(script):
            print(f"ERROR: {script} not found.")
            continue
            
        cmd = [sys.executable, script]
        # Capture output
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        if res.returncode == 0:
            print("SUCCESS.")
            # We want to parse the key metrics from output.
            # For now, just print the tail or specific lines.
            print("Output Summary:")
            lines = res.stdout.split('\n')
            relevant = [l for l in lines if any(k in l for k in ['R2', 'MAPE', 'RMSE', 'Score', 'Median'])]
            for r in relevant:
                print(f"  {r.strip()}")
            results[name] = relevant
        else:
            print(f"FAILED.")
            print(res.stderr)
    except Exception as e:
        print(f"Error: {e}")

print("\n--- REFRESH COMPLETE ---")
