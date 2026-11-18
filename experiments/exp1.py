import os
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,"src")
from pathsnap.io import load_trip, list_trips
from pathsnap.pipeline import prepare_trip
from pathsnap.features import FEATURES
from pathsnap.hmm import *
from pathsnap.metrics import *
TR=pd.read_parquet("results/features_train.parquet")
ST=os.environ.get("PATHSNAP_DATA", "Student folder") + "/student_test"
pw=(1-TR.label.mean())/TR.label.mean()
t0=time.time()
m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=42)
m.fit(TR[FEATURES],TR.label); print("fit full train %.1fs"%(time.time()-t0))
res=[]
for tid in list_trips(ST):
    p=prepare_trip(load_trip(ST,tid)); pr=m.predict_proba(p.df[FEATURES])[:,1]
    st=precompute_steps(p.df,p.pxy,p.times,p.net)
    c=per_point(st,classifier_top1(st,pr),len(p.pxy))
    row=dict(trip=tid,n=len(p.pxy),clf=pointwise_accuracy(c,p.ref))
    for a in [0.5,1,2]:
        for b in [2,5,10,20]:
            h=per_point(st,viterbi(st,pr,HMMParams(alpha=a,beta=b)),len(p.pxy))
            row[f"h{a}_{b}"]=pointwise_accuracy(h,p.ref)
    # error analysis for one config
    h=per_point(st,viterbi(st,pr,HMMParams(alpha=1,beta=5)),len(p.pxy))
    wrong=(h!=p.ref)
    adj=[p.net.connected(int(x),int(y),1) or p.net.connected(int(y),int(x),1) for x,y in zip(h[wrong],p.ref[wrong]) if x>=0]
    rev=[(p.net.arcs[x,0]==p.net.arcs[y,1]) and (p.net.arcs[x,1]==p.net.arcs[y,0]) for x,y in zip(h[wrong],p.ref[wrong]) if x>=0]
    row["err_adjacent"]=np.mean(adj) if adj else 0; row["err_reverse"]=np.mean(rev) if rev else 0
    res.append(row)
r=pd.DataFrame(res); print(r.round(3).to_string())
w=r.n/r.n.sum(); print((r.drop(columns=["trip","n"]).mul(w,axis=0).sum()).round(4).to_string())
