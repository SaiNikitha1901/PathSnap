import os
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,"src")
from pathsnap.io import load_trip, list_trips
from pathsnap.pipeline import prepare_trip
from pathsnap.features import FEATURES
from pathsnap.hmm import *
from pathsnap.metrics import *
from pathsnap.geo import point_segment
S="experiments/_cache/"
ST=os.environ.get("PATHSNAP_DATA", "Student folder") + "/student_test"
TR=pd.read_parquet(S+"tr2.parquet")
m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=42).fit(TR[FEATURES],TR.label)
m.booster_.save_model(S+"m2.txt")
tot=[]
for tid in list_trips(ST):
    tr=load_trip(ST,tid); p=prepare_trip(tr); st=precompute_steps(p.df,p.pxy,p.times,p.net); pr=m.predict_proba(p.df[FEATURES])[:,1]
    h=per_point(st,viterbi(st,pr,HMMParams(alpha=0.5,beta=5,uturn=8)),len(p.pxy))
    net=p.net; wr=np.where(h!=p.ref)[0]
    inroute=np.isin(h[wr],tr.route)
    rev=(net.arcs[h[wr],0]==net.arcs[p.ref[wr],1])&(net.arcs[h[wr],1]==net.arcs[p.ref[wr],0])
    adj=np.array([net.connected(int(a),int(b),1) or net.connected(int(b),int(a),1) for a,b in zip(h[wr],p.ref[wr])])
    dpred,_=point_segment(p.pxy[wr,0],p.pxy[wr,1],net.ax[h[wr]],net.ay[h[wr]],net.bx[h[wr]],net.by[h[wr]])
    print(tid,"err %d/%d  in-route %.2f reverse %.2f adjacent %.2f | dist to pred %.1f vs to label %.1f | label dist>30m %.2f"%(len(wr),len(h),inroute.mean(),rev.mean(),adj.mean(),np.median(dpred),np.median(p.ref_dist[wr]),(p.ref_dist[wr]>30).mean()))
