import os
import sys, time, pickle, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,"src")
from pathsnap.io import load_trip, list_trips
from pathsnap.pipeline import prepare_trip
from pathsnap.features import FEATURES
from pathsnap.hmm import *
from pathsnap.metrics import *
S="experiments/_cache/"
ST=os.environ.get("PATHSNAP_DATA", "Student folder") + "/student_test"
TR=pd.read_parquet(S+"tr2.parquet")
m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=42).fit(TR[FEATURES],TR.label)
test={}
for tid in list_trips(ST):
    p=prepare_trip(load_trip(ST,tid)); st=precompute_steps(p.df,p.pxy,p.times,p.net); test[tid]=(p,st,m.predict_proba(p.df[FEATURES])[:,1])
pickle.dump({k:(v[2]) for k,v in test.items()},open(S+"probs.pkl","wb"))
res=[]
for tid,(p,st,pr) in test.items():
    row=dict(trip=tid,n=len(p.pxy))
    row["vit_0.5_5_8"]=pointwise_accuracy(per_point(st,viterbi(st,pr,HMMParams(alpha=0.5,beta=5,uturn=8)),len(p.pxy)),p.ref)
    for a in [0.5,1]:
        for b in [3,5,10]:
            for u in [3,8]:
                row[f"post_{a}_{b}_{u}"]=pointwise_accuracy(per_point(st,posterior(st,pr,HMMParams(alpha=a,beta=b,uturn=u))[0],len(p.pxy)),p.ref)
    res.append(row)
r=pd.DataFrame(res); w=r.n/r.n.sum(); s=r.drop(columns=["trip","n"]).mul(w,axis=0).sum().sort_values(ascending=False); print(s.round(4).to_string())
print(r[["trip","vit_0.5_5_8",s.index[0]]].round(3).to_string())
