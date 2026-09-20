"""Face pairings of the five-cell quotient mesh of S^3/Gamma used by pds.py, printed as a LaTeX table."""
import numpy as np,itertools
from pds import V,reps,LM,one
idx={i:n for n,i in enumerate(sorted({v for t in reps for v in t}))}
print('% vertices (unit quaternions) used by the five representative cells')
for i,n in idx.items():print(f'% v{n} = {np.round(V[i],6).tolist()}')
faces={}
for c,t in enumerate(reps):
    for f in range(4):faces[(c,f)]=tuple(t[j] for j in range(4) if j!=f)
done=set();rows=[]
for (c,f),A in faces.items():
    if (c,f) in done:continue
    for g in range(120):
        img=tuple(int(LM[g,i]) for i in A)
        for (c2,f2),B in faces.items():
            if (c2,f2)!=(c,f) and (c2,f2) not in done and set(img)==set(B):
                perm=tuple(B.index(x) for x in img);rows.append((c,f,c2,f2,perm,g));done|={(c,f),(c2,f2)};break
        if (c,f) in done:break
print(len(rows),'pairings');
for c,f,c2,f2,perm,g in rows:
    print(f'{c} & {f} & {c2} & {f2} & $({perm[0]}\\,{perm[1]}\\,{perm[2]})$ & $({", ".join(f"{x:.4f}" for x in V[g])})$\\\\')
print('cells as vertex labels:',[[idx[i] for i in t] for t in reps])
