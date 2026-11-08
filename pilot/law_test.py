import numpy as np, json
from pilot.geometry import load, stack, PAIRS
from pilot.metrics import auroc
from scipy.stats import norm

MODELS={"3b":"results/runs/v5_261.jsonl","7b":"results/runs/v5_261_7b.jsonl",
        "llava":"results/runs/v5_261_llava.jsonl","gemma":"results/runs/v5_261_gemma.jsonl"}
rows=[]
for tag,path in MODELS.items():
    recs,H=load(path)
    Xn,_=stack(recs,H,PAIRS["none"][0]); An,_=stack(recs,H,PAIRS["none"][1])
    d0=Xn.mean(0)-An.mean(0); w=d0/np.linalg.norm(d0)
    s0=np.concatenate([Xn@w,An@w]); y0=np.r_[np.ones(len(Xn)),np.zeros(len(An))]
    def pooled_sd(s,y):
        return np.sqrt(((s[y==1]-s[y==1].mean())**2).sum()+((s[y==0]-s[y==0].mean())**2).sum())/(len(s)-2)
    sd0=pooled_sd(s0,y0); gap0=(Xn@w).mean()-(An@w).mean(); dp0=gap0/sd0
    au0=auroc(s0,y0)
    print(f"== {tag} gap0={gap0:.3f} sd0={sd0:.3f} d'={dp0:.2f} AUC={au0:.3f}")
    for cond,(pk,ak) in PAIRS.items():
        X=stack(recs,H,pk)[0]; A=stack(recs,H,ak)[0]
        s=np.r_[X@w,A@w]; y=np.r_[np.ones(len(X)),np.zeros(len(A))]
        sd=pooled_sd(s,y); gap=(X@w).mean()-(A@w).mean(); dp=gap/sd; au=auroc(s,y)
        rows.append(dict(model=tag,cond=cond,gap_ratio=gap/gap0,sd_ratio=sd/sd0,dp_ratio=dp/dp0,auc=au,dp=dp))
        print(f"  {cond:11s} gap_ratio={gap/gap0:.3f} sd_ratio={sd/sd0:.3f} d'_ratio={dp/dp0:.3f} AUC={au:.3f} d'={dp:.2f}")
rows=[r for r in rows if r['cond']!='none']
lg=np.log([r['gap_ratio'] for r in rows]); ls=np.log([r['sd_ratio'] for r in rows]); ld=np.log([r['dp_ratio'] for r in rows])
A=np.c_[lg,np.ones(len(lg))]; coef=np.linalg.lstsq(A,ls,rcond=None)[0]
pred=A@coef; r2=1-((ls-pred)**2).sum()/((ls-ls.mean())**2).sum()
print(f"\nlog sd_ratio = {coef[0]:.3f} log gap_ratio + {coef[1]:.3f}; R2={r2:.3f} (n={len(rows)})")
print(f"d' ratio mean={np.exp(ld).mean():.3f} range {np.exp(ld).min():.3f}-{np.exp(ld).max():.3f}")
# predicted AUROC from d' under equal-variance Gaussian
for r in rows:
    pd_au=norm.cdf(r['dp']/np.sqrt(2))
    r['pred_auc']=pd_au
mae=np.mean([abs(r['pred_auc']-r['auc']) for r in rows])
print("mean |pred AUC from d' - observed AUC| =",round(mae,4))

# d' ratio as a function of the measured gap ratio: if the within-class spread
# contraction were condition-independent, d'_c/d'_0 would be proportional to the
# gap ratio with gain 1/(SD ratio).
g=np.array([r['gap_ratio'] for r in rows]); d=np.array([r['dp_ratio'] for r in rows])
Ag=np.c_[g,np.ones(len(g))]; coefd=np.linalg.lstsq(Ag,d,rcond=None)[0]
predd=Ag@coefd; r2d=1-((d-predd)**2).sum()/((d-d.mean())**2).sum()
slope0=(g*d).sum()/(g*g).sum()
rmse0=np.sqrt((((d-slope0*g)**2).mean())); max0=float(np.abs(d-slope0*g).max())
sdrs=np.array([r['sd_ratio'] for r in rows])
print(f"\nd' ratio = {coefd[0]:.3f} gap ratio + {coefd[1]:.3f}; R2={r2d:.3f}")
print(f"through-origin slope={slope0:.3f}; RMSE={rmse0:.4f}; max abs={max0:.4f}")
print(f"SD ratio mean={sdrs.mean():.3f} sd={sdrs.std():.3f} range {sdrs.min():.3f}-{sdrs.max():.3f}")
per_model={}
for tag in MODELS:
    sub=[r for r in rows if r['model']==tag]
    ratios=np.array([r['dp_ratio']/r['gap_ratio'] for r in sub])
    per_model[tag]=dict(gain_mean=float(ratios.mean()),gain_min=float(ratios.min()),
                        gain_max=float(ratios.max()),
                        sd_ratio_mean=float(np.mean([r['sd_ratio'] for r in sub])))
    print(f"  {tag:6s} d'/gap={ratios.mean():.3f} ({ratios.min():.3f}-{ratios.max():.3f})")
fit=dict(slope=float(coefd[0]),intercept=float(coefd[1]),r2=float(r2d),
         through_origin_slope=float(slope0),through_origin_rmse=float(rmse0),
         through_origin_max_abs=float(max0),sd_ratio_mean=float(sdrs.mean()),
         sd_ratio_sd=float(sdrs.std()),sd_ratio_min=float(sdrs.min()),
         sd_ratio_max=float(sdrs.max()),per_model=per_model)
json.dump(dict(rows=rows,fit=fit),open("results/law_isotropic.json","w"),indent=1,default=float)
