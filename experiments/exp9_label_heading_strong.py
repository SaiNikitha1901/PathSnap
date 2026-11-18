import os
import sys, time, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,"src")
from pathsnap.io import load_trip, list_trips
from pathsnap.pipeline import prepare_trip, DEFAULT_CFG
from pathsnap.features import FEATURES
from pathsnap.hmm import *
from pathsnap.metrics import *
TRR=os.environ.get("PATHSNAP_DATA", "Student folder") + "/train"; ST=os.environ.get("PATHSNAP_DATA", "Student folder") + "/student_test"
for hw in [10.0, 20.0]:
    cfg=dict(DEFAULT_CFG, label_heading_weight=hw)
    TR=pd.concat([prepare_trip(load_trip(TRR,t),cfg).df for t in list_trips(TRR)],ignore_index=True)
    m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=42).fit(TR[FEATURES],TR.label)
    res=[]
    for tid in list_trips(ST):
        tr=load_trip(ST,tid); p=prepare_trip(tr,cfg); st=precompute_steps(p.df,p.pxy,p.times,p.net); pr=m.predict_proba(p.df[FEATURES])[:,1]
        row=dict(trip=tid,n=len(p.pxy),clf=pointwise_accuracy(per_point(st,classifier_top1(st,pr),len(p.pxy)),p.ref))
        for a,b,u in [(0.5,5,8),(1,3,15),(0.5,5,15),(1,5,8),(0.5,3,8)]:
            h=per_point(st,viterbi(st,pr,HMMParams(alpha=a,beta=b,uturn=u)),len(p.pxy))
            row[f"{a}_{b}_{u}"]=pointwise_accuracy(h,p.ref)
            if (a,b,u)==(0.5,5,8):
                rp=route_prf(reconstruct_route(h,p.net),tr.route,p.net); row["routeF1"]=rp["route_f1"]; row["routeP"]=rp["route_precision"]; row["routeR"]=rp["route_recall"]; row["cont"]=route_continuity(h,p.net)
        res.append(row)
    r=pd.DataFrame(res); w=r.n/r.n.sum()
    print("label_heading_weight",hw); print(r.drop(columns=["trip","n"]).mul(w,axis=0).sum().round(4).to_string()); print(r[["trip","clf","0.5_5_8","routeF1"]].round(3).to_string())
