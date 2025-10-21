import sys
print('Python:', sys.version)
try:
    import numpy as np
    print('NumPy:', np.__version__)
except Exception as e:
    print('NumPy import failed:', e)
try:
    import matplotlib
    print('matplotlib:', matplotlib.__version__)
except Exception as e:
    print('matplotlib import failed:', e)
