import pandas as pd
import numpy as np

data=pd.read_csv('ws.csv')

concepts=np.array(data)[:,:-1]
target=np.array(data)[:,-1]

def train(con,tar):
    specific_h=con[0].copy()
    general_h=[['?' for x in range(len(specific_h))] for x in range(len(specific_h))]

    for i, val in enumerate(con):
        if tar[i]=='yes':
            for x in range(len(specific_h)):
                if val[x]!=specific_h[x]:
                    specific_h[x]='?'
                    general_h[x][x]='?'

        else:
            for x in range(len(specific_h)):
                if val[x]!=specific_h[x]:
                    general_h[x][x]=specific_h[x]

                else:
                    general_h[x][x]='?'
        print("iteratioms"+str(i+1)+"}")
        print("Specific:"+str(specific_h))
        print("general Hyp:"+str(general_h)+"\n")
    general_h=[general_h[i] for i, val in enumerate(general_h) if val!=['?' for x in range(len(specific_h))]]
    return specific_h,general_h
specific,general=train(concepts,target)
print("The final hypothesis is:")
print("SH"+str(specific))
print("GH"+str(general))

