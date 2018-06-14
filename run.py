import definitions as df
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

csv_read = pd.read_csv("./Ogura_Hyakunin_Isshu.csv-master/ogurahyakuninisshu.csv", index_col=0, usecols=[0, 4, 5])

dataset = np.sum(csv_read.values, axis=1)
#dataset = ['うらみわびほさぬそでだにあるものをこひにくちなむなこそをしけれ',
#           'おもひわびさてもいのちはあるものをうきにたへぬはなみだなりけり',
#           'はるのよのゆめばかりなるたまくらにかひなくたたむなこそをしけれ',
#           'きみがためはるののにいでてわかなつむわがころもでにゆきはふりつつ',
#           'きみがためをしからざりしいのちさへながくもがなとおもひけるかな']
for i in [31, 32, 33]:
    summation = 0
    for j in dataset:
        if len(j) == i:
            summation += 1
    print(str(i) + '文字: ' + str(summation) + '首')

arr = df.apply_ldist_to_datas(dataset)
mindist = np.nanmin(arr)
maxdist = np.nanmax(arr)
df.outdotfile('graph', arr, 60, dataset)
plt.hist(arr[~np.isnan(arr)], rwidth=0.8, ec='black', bins=range(int(mindist), int(maxdist)+2))
plt.title('histgram')
plt.xlabel('Levenshtein Distance')
plt.ylabel('numbers of Waka pair')
plt.savefig('hist.png')

for i in range(int(mindist), int(maxdist)+1):
    idx = np.where(arr == i)
    print('編集距離' + str(i) + ':' + str(len(idx[0])) + '個')

print('総数が11個以下の編集距離数を取り上げて表示します.')
for i in range(int(mindist), int(maxdist)+1):
    idx = np.where(arr == i)
    if len(idx[0]) <= 11:
        print('編集距離:' + str(i))
        for j, k in zip(idx[0], idx[1]):
            print('"' + dataset[j] + '"---"' + dataset[k] + '"')

np.nan_to_num(arr)
mdist = np.sum(np.nan_to_num(arr) + np.nan_to_num(arr.T), axis=0)/99
mminidx = np.argmin(mdist)
mmaxidx = np.argmax(mdist)

print('最小平均距離(' + str(mdist[mminidx]) + '):"' + dataset[mminidx] + '"')
print('最大平均距離(' + str(mdist[mmaxidx]) + '):"' + dataset[mmaxidx] + '"')

print('平均距離ランキング')
rank = np.argsort(mdist)
for i in range(100):
    print(str(i+1) + '位(' + str(mdist[rank[i]]) + '):"' + dataset[rank[i]] + '"')
# res = df.kmeans(arr, 5)
# print(res[0])

"""
minidx = np.where(res[0] == np.argmin(res[1]))
for i, j in zip(minidx[0], minidx[1]):
    print((i, j, arr[i, j]))
print('*************')
maxidx = np.where(res[0] == np.argmax(res[1]))
for i, j in zip(maxidx[0], maxidx[1]):
    print((i, j, arr[i, j]))
"""
# Z = linkage(arr, 'ward')
# dn = dendrogram(Z)
# plt.show()
# print(np.where(arr == np.nanmin(arr)))
#linkage(ld.levenshtein_distance(csv.read))