import pandas as pd
import numpy as np

data=pd.read_csv('ws.csv')

concepts=np.array(data)[:,:-1]
target=np.array(data)[:,-1]

def train(con,tar):
    for i, val in enumerate(tar):
        if val.lower()=='yes':
            specific_h=con[i].copy()
            break
    for i,val in enumerate(con):
        if tar[i].lower()=='yes':
            for x in range(len(specific_h)):
                if val[x]!=specific_h[x]:
                    specific_h[x]='?'

                else:
                    pass
    return specific_h
print(train(concepts,target))