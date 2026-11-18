import os
import sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from pathsnap.io import load_trip
from pathsnap.pipeline import prepare_trip
from pathsnap.features import point_motion
from pathsnap.geo import angle_diff
ST=os.environ.get("PATHSNAP_DATA", "Student folder") + "/student_test"
for tid in ["00000077","00000007"]:
    tr=load_trip(ST,tid); p=prepare_trip(tr); net=p.net
    hd,valid,sp,_=point_motion(p.pxy,p.times)
    d=angle_diff(hd,net.bearing[p.ref])
    print(tid,"heading diff GPS vs LABEL arc: pct within 45deg %.3f, >135deg %.3f, invalid %.3f, speed median %.1f"%((d[valid]<45).mean(),(d[valid]>135).mean(),(~valid).mean(),np.median(sp)))
    # route consistency: consecutive route arcs connected?
    r=tr.route; ok=np.mean(net.arcs[r[:-1],1]==net.arcs[r[1:],0]); print(" route consecutive connected frac %.3f, len %d, unique %d"%(ok,len(r),len(set(r))))
    # direction along route vs time: does reference position k increase along GPS?
    bad=np.where(valid&(d>135))[0]; print(" first bad pts",bad[:20], "speed there",np.round(sp[bad[:10]],1))
