import sys
sys.path.append("./build")
import rpca_module
import numpy as np
D = np.random.rand(10, 3).astype(np.float64)
results = rpca_module.process_batch([D])
print("Success!")
