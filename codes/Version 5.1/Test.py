import networkx as nx
from pymnet import *
import random
import matplotlib
import numpy as np
import math
import cascade as cas
import pandas as pd

matplotlib.use('TkAgg')

coords = {}

def draw_network(net):
    fig = draw(net, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5, show=True)
    fig.savefig("test.pdf")


if __name__ == "__main__":
    df = pd.DataFrame({'Name': ['Kim', 'Park','Choi'],
                       'ID': [1, 2, 3],
                       'Age': [21, 23, 19]})
    df.to_csv('test.csv')