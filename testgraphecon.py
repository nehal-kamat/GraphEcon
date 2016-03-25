from graphecon import LinearUtility
import numpy as np

U = LinearUtility(np.array([1,2,3]))

print U.bestPurchase(np.array([0.5,0.3,0.1]), 6)
