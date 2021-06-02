import networkx as nx
from pymnet import *
import random
import matplotlib
import numpy as np
import math
import cascade as cas
import pandas as pd
import csv

matplotlib.use('TkAgg')

coords = {}

def draw_network(net):
    fig = draw(net, nodeSizeRule={'rule':'scaled', 'scalecoeff': 0.01}, defaultEdgeWidth=0.5, show=True)
    fig.savefig("test.pdf")


if __name__ == "__main__":
    """
    data_set = [['Kim', 1, 21], ['Park', 2, 23, 70], ['Lee', 3, 19], ['Choi', 4]]
    max_len = 4
    for data in data_set:
        print(data)
        if len(data) < max_len:
            while len(data) < max_len:
                data.append(0)
    
    print(data_set)

    df1 = pd.DataFrame({'Name': ['Kim', 'Park', 'Lee', 'Choi'],
                        'ID': [1, 2, 3, 4],
                        'Age': [21, 23, 19, 0],
                        'Score': [0, 70, 0, 0]})

    index = 1
    df2 = pd.DataFrame(data_set, columns=['Name%d' % index, 'ID', 'Age%d' % index, 'Score'])
    # df2.to_csv('test.csv')

    f = open('test.csv', 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(data_set[0])
    wr.writerow(data_set[1])
    wr.writerow(data_set[2])

    f.close()

    len = 4
    f = open('test.csv', 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(data_set[3])
    f.close()

    f = open('test.csv', 'w', newline='')
    wr = csv.writer(f)
    header = ["name"]
    for i in range(len - 1):
        header.append("here%d" % i)
    wr.writerow(header)
    f.close()
    

    dic = {1: (1, 1), 2: (2, 2), 3: (3, 3), 4: (4, 4)}
    dic[5] = (5, 5)
    dic[1] = (6, 6)

    print(dic)
    

    target = list(range(1680))
    answer_0 = []
    answer_1 = []
    answer_2 = []
    answer_3 = []
    answer_4 = []
    answer_5 = []
    answer_6 = []
    answer_7 = []
    answer_8 = []
    answer_9 = []

    for i in target:
        if (i*i) % 1680 == 0:
            answer_0.append(i)
        elif (i*i) % 1680 == 1:
            answer_1.append(i)
        elif (i*i) % 1680 == 2:
            answer_2.append(i)
        elif (i*i) % 1680 == 3:
            answer_3.append(i)
        elif (i*i) % 1680 == 4:
            answer_4.append(i)
        elif (i*i) % 1680 == 5:
            answer_5.append(i)
        elif (i*i) % 1680 == 6:
            answer_6.append(i)
        elif (i*i) % 1680 == 7:
            answer_7.append(i)
        elif (i*i) % 1680 == 8:
            answer_8.append(i)
        elif (i*i) % 1680 == 9:
            answer_9.append(i)

    print(answer_0)
    print(len(answer_0))
    print(answer_1)
    print(len(answer_1))
    print(answer_2)
    print(answer_3)
    print(answer_4)
    print(len(answer_4))
    print(answer_5)
    print(answer_6)
    print(answer_7)
    print(answer_8)
    print(answer_9)
    print(len(answer_9))
    """
    test_dict = {}
    test_dict[1] = 3

    print(test_dict)