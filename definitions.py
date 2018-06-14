import numpy as np
import random as rd


def levenshtein_distance(str1: str, str2: str):
    n, m = len(str1), len(str2)
    arr = np.zeros((n+1, m+1), dtype='uint8')

    for i in range(n+1):
        arr[i][0] = i

    for j in range(m+1):
        arr[0][j] = j

    for i in range(n):
        for j in range(m):
            cost = 0 if str1[i] == str2[j] else 1
            arr[i+1, j+1] = min(arr[i, j+1] + 1,
                                arr[i+1, j] + 1,
                                arr[i, j] + cost)
    return arr[n][m]


def apply_ldist_to_datas(dlist: np.array):
    n = len(dlist)
    arr = np.zeros((n, n))
    arr[:, :] = np.nan

    for i in range(n):
        for j in range(i+1, n):
            arr[i, j] = levenshtein_distance(dlist[i], dlist[j])
    return arr


def kmeans(arr: np.array, cnum: int):
    tags = np.zeros(arr.shape)
    n = tags.shape[0]
    tags[:] = np.nan
    mlist = np.array(list(range(cnum)), dtype='float')

    for i in range(n):
        for j in range(i+1, n):
            tags[i, j] = rd.randrange(cnum)

    def itr():
        # 中心の更新
        for i in range(cnum):
            index = np.where(tags == i)
            s = 0
            for j, k in zip(index[0], index[1]):
                s += arr[j][k]
            mlist[i] = float(s) / len(index[0])
        # 新しいタグ付け
        for i in range(n):
            for j in range(i+1, n):
                idx = np.abs(mlist - arr[i, j]).argmin()
                tags[i, j] = idx
    while True:
        print(mlist)
        lasttags = tags
        itr()
        if np.all((lasttags == tags) | np.isnan(lasttags) & np.isnan(tags)):
            break
    print(mlist)
    return tags, mlist


def outdotfile(fname:str, arr: np.array, cutoff: int, datas: np.array):
    f = open(fname + '.dot', 'w')
    nodes = set([])
    edges = set([])
    nums = 0
    mindist = np.nanmin(arr)
    maxdist = np.nanmax(arr)
    for i in range(int(mindist), int(maxdist) + 1):
        idx = np.where(arr == i)
        nums += len(idx[0])
        for j, k in zip(idx[0], idx[1]):
            nodes.add(j)
            nodes.add(k)
            edges.add((j, k))
        if len(nodes) >= cutoff:
            break
    codes = "digraph graph1{\n" \
            "graph[layout=neato];\n" \
            "edge[arrowhead=none,len=2.6, penwidth=2];\n" \
            'node[shape=circle,fixedsize="MS Gothic",penwidth=2];\n'
    for i in nodes:
        codes += 'a' + str(i) + '[label="' + str(i+1) + '"];\n'

    for t in edges:
        codes += 'a' + str(t[0]) + '->' + 'a' + str(t[1]) + ';\n'

    codes += '}'

    f.write(codes)
    f.close()


def makegraph(arr: np.array, threshold: int):
    nodes = set([])
    edges = set([])
    shape = arr.shape()
    for i in range(0, shape[0]+1):
        for j in range(i+1, shape[1]+1):
            if arr[i, j] <= threshold:
                nodes.add(i)
                nodes.add(j)
                edges.add((i, j))
    return nodes, edges

def clusteringcoefficient(graph: tuple):
    clist=[]
    nodes = graph[0]
    edges = graph[1]