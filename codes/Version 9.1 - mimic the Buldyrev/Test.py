import networkx as nx
import statistics
import random
import matplotlib
import numpy as np
import math
import cascade as cas
import pandas as pd
import csv
matplotlib.use('TkAgg')

coords = {}

if __name__ == "__main__":
    a, b = [0 for i in range(1000)]

    c = True
    while c:
        del a

        a = nx.utils.powerlaw_sequence(1000, 3)
        a = [round(i) for i in a]

        if max(a) < 10:
            c = False

    c = True
    while c:
        del b

        b = np.random.normal(loc=4, size=100)
        b = [round(i) for i in b]

        if max(b) < 10:
            c = False

    a_prob_1 = a.count(1) / len(a)
    a_prob_2 = a.count(2) / len(a)
    a_prob_3 = a.count(3) / len(a)
    a_prob_4 = a.count(4) / len(a)
    a_prob_5 = a.count(5) / len(a)
    a_prob_6 = a.count(6) / len(a)
    a_prob_7 = a.count(7) / len(a)
    a_prob_8 = a.count(8) / len(a)
    a_prob_9 = a.count(9) / len(a)
    a_prob_10 = a.count(10) / len(a)

    b_prob_1 = b.count(1) / len(b)
    b_prob_2 = b.count(2) / len(b)
    b_prob_3 = b.count(3) / len(b)
    b_prob_4 = b.count(4) / len(b)
    b_prob_5 = b.count(5) / len(b)
    b_prob_6 = b.count(6) / len(b)
    b_prob_7 = b.count(7) / len(b)
    b_prob_8 = b.count(8) / len(b)
    b_prob_9 = b.count(9) / len(b)
    b_prob_10 = b.count(10) / len(b)

    new_prob_1 = a_prob_1 * b_prob_1
    new_prob_2 = a_prob_2 * b_prob_2
    new_prob_3 = a_prob_3 * b_prob_3
    new_prob_4 = a_prob_4 * b_prob_4
    new_prob_5 = a_prob_5 * b_prob_5
    new_prob_6 = a_prob_6 * b_prob_6
    new_prob_7 = a_prob_7 * b_prob_7
    new_prob_8 = a_prob_8 * b_prob_8
    new_prob_9 = a_prob_9 * b_prob_9
    new_prob_10 = a_prob_10 * b_prob_10


    print(a)
    print(statistics.mean(a))
    print(b)
    print(statistics.mean(b))

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
    
    test_dict = {}
    test_dict[1] = 3

    print(test_dict)
    """
