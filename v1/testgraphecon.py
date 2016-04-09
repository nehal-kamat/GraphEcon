from graphecon import Utility, LinearUtility, NonLinearUtility
import numpy as np

'''
U = LinearUtility(np.array([1,2,3]))

print U.bestPurchase(np.array([0.5,0.3,0.1]), 6)

'''
U = Utility(np.array(['math.pow(x, 2)+x', 'math.pow(x, 3)', '4*x']))

print U.bestPurchase(np.array([0.5,0.3,0.1]), 6)

U = Utility(np.array(['x', '2*x', '3*x']))

print U.bestPurchase(np.array([0.5,0.3,0.1]), 6)