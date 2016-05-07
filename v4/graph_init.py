import numpy as np
import networkx as nx

from market_status import Market
from utility import LinearUtility

def main():
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2)]) # modify tuples according to your preferred edgelist configuration

    ngoods = 2  # modify according to your preferred number of goods

    # nx.draw(G, with_labels = True)
    # plt.show()
    # plt.savefig("InitialGraph.png")

    # modify values according to your preferred {node-label : endowment-per-good vector} configuration
    endowment = {0:np.array([1,2]), 1:np.array([1,1]), 2:np.array([2,1])}

    # modify values according to your preferred {node-label : utility-per-good vector} configuration
    utility = {0:LinearUtility([1,0]), 1:LinearUtility([1,1]), 2:LinearUtility([0,1])}

    # modify values according to your preferred {node-label : price-per-good vector} configuration
    prices = {0:np.array([2,1]), 1:np.array([2,2]), 2:np.array([1,2])}

    mkt = Market(G, endowment, utility, ngoods)

    mkt.checkClear(prices)


if __name__ == "__main__":
    main()
