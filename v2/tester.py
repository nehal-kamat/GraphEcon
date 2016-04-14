from itertools import product
"""
variants = {
  "1" : ["A", "B"],
  "2" : ["1", "2", "3"],
  "3" : ["x","y"]
}
"""

variants = {
  0: [(2.0, 0.0)],
  1: [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)],
  2: [(0.0, 2.0)]
 }
"""

 """

combinations = list(product(*(variants[n] for n in variants)))

print len(combinations)
