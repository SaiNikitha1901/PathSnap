import os
import sys, time, pickle, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,"src")
from pathsnap.io import load_trip, list_trips
from pathsnap.pipeline import prepare_trip
from pathsnap.features import FEATURES
from pathsnap.hmm import *
from pathsnap.metrics import *
S="experiments/_cache/"
TRR=os.environ.get("PATHSNAP_DATA", "Student folder") + "/train"; ST=os.environ.get("PATHSNAP_DATA", "Student folder") + "/student_test"
t0=time.time()
TR=pd.concat([prepare_trip(load_trip(TRR,t)).df for t in list_trips(TRR)],ignore_index=True); print("train feats %.0fs"%(time.time()-t0), len(TR))
TR.to_parquet(S+"tr2.parquet")
test={}
for tid in list_trips(ST):
    p=prepare_trip(load_trip(ST,tid)); st=precompute_steps(p.df,p.pxy,p.times,p.net); test[tid]=(p,st)
print("test prep %.0fs"%(time.time()-t0))
m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=42)
m.fit(TR[FEATURES],TR.label)
res=[]
for tid,(p,st) in test.items():
    pr=m.predict_proba(p.df[FEATURES])[:,1]
    row=dict(trip=tid,n=len(p.pxy),clf=pointwise_accuracy(per_point(st,classifier_top1(st,pr),len(p.pxy)),p.ref))
    for a in [0.5,1,2]:
        for b in [1,2,5]:
            for u in [0,3,8]:
                row[f"a{a}_b{b}_u{u}"]=pointwise_accuracy(per_point(st,viterbi(st,pr,HMMParams(alpha=a,beta=b,uturn=u)),len(p.pxy)),p.ref)
    res.append(row)
r=pd.DataFrame(res); w=r.n/r.n.sum(); s=r.drop(columns=["trip","n"]).mul(w,axis=0).sum().sort_values(ascending=False)
print(s.head(8).round(4).to_string()); best=s.index[0] if s.index[0]!="clf" else s.index[1]
print(r[["trip","n","clf",best]].round(3).to_string())
