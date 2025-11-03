# Ensure repo root is on sys.path so `import src...` works in CI
import os, sys
sys.path.insert(0, os.path.abspath(os.getcwd()))
