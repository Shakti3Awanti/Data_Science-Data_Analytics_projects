import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import pandas
    print(f"Pandas version: {pandas.__version__}")
except ImportError as e:
    print(f"Pandas import failed: {e}")

try:
    import numpy
    print(f"Numpy version: {numpy.__version__}")
except ImportError as e:
    print(f"Numpy import failed: {e}")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
    import matplotlib.pyplot as plt
    print("matplotlib.pyplot imported successfully")
except ImportError as e:
    print(f"Matplotlib import failed: {e}")
except Exception as e:
    print(f"Matplotlib error: {e}")

try:
    import seaborn
    print(f"Seaborn version: {seaborn.__version__}")
except ImportError as e:
    print(f"Seaborn import failed: {e}")
