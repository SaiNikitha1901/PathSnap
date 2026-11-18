import os
import sys, time, pickle, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,"src")
from pathsnap.io import load_trip, list_trips
from pathsnap.pipeline import prepare_trip
from pathsnap.features import FEATURES
from pathsnap.hmm import *
from pathsnap.metrics import *
from sklearn.model_selection import GroupKFold
S="experiments/_cache/"
ST=os.environ.get("PATHSNAP_DATA", "Student folder") + "/student_test"
TR=pd.read_parquet(S+"tr2.parquet")
P=dict(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=42)
oof=np.zeros(len(TR))
for tri,vai in GroupKFold(5).split(TR,groups=TR.trip):
    oof[vai]=lgb.LGBMClassifier(**P).fit(TR.iloc[tri][FEATURES],TR.iloc[tri].label).predict_proba(TR.iloc[vai][FEATURES])[:,1]
m1=lgb.LGBMClassifier(**P).fit(TR[FEATURES],TR.label)
def ctx(df,p1):
    d=df[["trip","pt","arc"]].copy(); d["p1"]=p1
    d["p1n"]=d.p1/d.groupby(["trip","pt"]).p1.transform("sum").clip(lower=1e-9)
    d["p1rank"]=d.groupby(["trip","pt"]).p1.rank(ascending=False)
    base=d[["trip","pt","arc","p1n"]]
    for o in [1,2,3,5,8]:
        for sgn in [1,-1]:
            sh=base.copy(); sh["pt"]=sh.pt-sgn*o
            d=d.merge(sh.rename(columns={"p1n":f"same_{sgn*o}"}),on=["trip","pt","arc"],how="left")
    cols=[c for c in d.columns if c.startswith("same_")]
    d[cols]=d[cols].fillna(0)
    d["same_mean"]=d[cols].mean(axis=1)
    return d[["p1","p1n","p1rank"]+cols+["same_mean"]].to_numpy(), ["p1","p1n","p1rank"]+cols+["same_mean"]
Xc,names=ctx(TR,oof); F2=FEATURES+names
X2=np.column_stack([TR[FEATURES].to_numpy(),Xc])
m2=lgb.LGBMClassifier(**dict(P,n_estimators=400)).fit(X2,TR.label)
res=[]
for tid in list_trips(ST):
    p=prepare_trip(load_trip(ST,tid)); st=precompute_steps(p.df,p.pxy,p.times,p.net)
    p1=m1.predict_proba(p.df[FEATURES])[:,1]; xc,_=ctx(p.df,p1)
    p2=m2.predict_proba(np.column_stack([p.df[FEATURES].to_numpy(),xc]))[:,1]
    row=dict(trip=tid,n=len(p.pxy),clf1=pointwise_accuracy(per_point(st,classifier_top1(st,p1),len(p.pxy)),p.ref),clf2=pointwise_accuracy(per_point(st,classifier_top1(st,p2),len(p.pxy)),p.ref))
    for a,b,u in [(0.5,5,8),(1,3,15),(1,5,8),(0.3,5,8),(0.5,10,8),(1,10,8),(2,5,8)]:
        row[f"h2_{a}_{b}_{u}"]=pointwise_accuracy(per_point(st,viterbi(st,p2,HMMParams(alpha=a,beta=b,uturn=u)),len(p.pxy)),p.ref)
    res.append(row)
r=pd.DataFrame(res); w=r.n/r.n.sum(); s=r.drop(columns=["trip","n"]).mul(w,axis=0).sum().sort_values(ascending=False); print(s.round(4).to_string())
print(r[["trip","clf1","clf2",s.index[0]]].round(3).to_string())
