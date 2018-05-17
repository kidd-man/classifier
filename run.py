import definitions as ld
import csv
import pandas as pd

csv_read = pd.read_csv("./Ogura_Hyakunin_Isshu.csv-master/ogurahyakuninisshu.csv", index_col=0, usecols=[0, 4, 5])

print(csv_read)
