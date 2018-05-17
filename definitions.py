import numpy as np


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

    return arr


