import networkx as nx
from pymnet import *
import random
import matplotlib
import numpy as np
import math
import cascade as cas
import pandas as pd
import glob
from PIL import Image

matplotlib.use('TkAgg')

coords = {}

def draw_network(net):
    fig = draw(net, nodeCoords=coords, nodeLabelRule={}, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.05}, defaultEdgeWidth=0.9, show=True, layerLabelRule={})
    fig.savefig("test.pdf")

def stitch():
    rep = 1

    fp_in = "-{:02d}-*.png".format(rep)
    fp_out = "-{:02d}.gif".format(rep)
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=500, loop=0)

if __name__ == "__main__":
    stitch()