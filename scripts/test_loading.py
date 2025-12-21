import sys
import os

# Add scripts directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    print("Testing import of vision.py...")
    import vision
    print("SUCCESS: vision.py imported without error.")
except Exception as e:
    print(f"FAILURE: An error occurred: {e}")
