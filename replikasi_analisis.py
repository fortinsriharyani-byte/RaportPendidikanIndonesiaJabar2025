"""Skrip replikasi: Tipologi Mutu Layanan Pendidikan Lintas Jalur dan Kewenangan (Rapor Pendidikan Jawa Barat 2025).
Menghasilkan seluruh statistik pada naskah revisi. Python 3 + NumPy, SciPy, scikit-learn. Seed 20251119.
"""
import numpy as np
names = ["SD Umum","SD Kesetaraan","SMP Umum","SMP Kesetaraan","SMA Umum","SMA Kesetaraan","SMK Umum",
         "SDLB","SMPLB","SMALB","MI","MTs","MA","SD Pendidikan Keagamaan","SMP Pendidikan Keagamaan",
         "SMA Pendidikan Keagamaan","PK SD Setara","PK SMP Setara","PK SMA Setara"]
short = ["SD Umum","SD Kes.","SMP Umum","SMP Kes.","SMA Umum","SMA Kes.","SMK Umum","SDLB","SMPLB","SMALB",
         "MI","MTs","MA","SD PK","SMP PK","SMA PK","PK SD Setara","PK SMP Setara","PK SMA Setara"]
X = np.array([
[2,2,3,3,3,3,3],[2,2,3,2,3,2,3],[3,2,3,2,3,3,3],[2,2,2,2,2,2,3],[3,2,3,3,3,3,3],[1,2,2,2,2,2,3],
[2,2,3,2,3,3,3],[3,3,3,3,3,3,3],[3,3,3,3,3,3,3],[2,2,3,3,2,2,3],[2,2,2,2,2,2,3],[2,2,2,2,2,2,3],
[2,2,3,2,3,3,3],[3,3,3,2,2,3,3],[3,3,3,2,2,3,3],[3,3,3,3,1,3,3],[2,2,2,2,1,2,2],[2,2,2,2,1,2,3],[2,2,3,2,1,2,3]])
ind = ["LIT","NUM","KAR","KPB","AMAN","BINEKA","INKLUSI"]
cl = np.array([2,1,2,1,2,1,2,3,3,1,1,1,2,3,3,3,1,1,1])
jalur = np.array([0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1])  # 1 nonformal
kew = np.array([0]*10+[1]*9)  # 1 Kemenag
jenjang = np.array([0,0,1,1,2,2,2,0,1,2,0,1,2,0,1,2,0,1,2])
subs = np.array([0,1,0,1,0,1,0,2,2,2,3,3,3,4,4,4,5,5,5])  # umum,kes,SLB,madrasah,PK,PKsetara

def gower(X):
    R = X.max(0)-X.min(0); R[R==0]=1
    D = np.abs(X[:,None,:]-X[None,:,:])/R
    return D.mean(2)


# ===== analisis =====
import sys, itertools, json
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

rng = np.random.default_rng(20251119)
N = len(X); out = {}
D = gower(X)
# ---------- MCA
cats = []
Z = []
for k in range(7):
    for v in sorted(set(X[:, k])):
        cats.append(f"{ind[k]}-{'KSB'[v-1]}"); Z.append((X[:, k] == v).astype(float))
Z = np.array(Z).T
Q = 7; J = Z.shape[1]
P = Z / Z.sum(); r = P.sum(1); c = P.sum(0)
S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
U, sv, Vt = np.linalg.svd(S, full_matrices=False)
lam = sv**2
benz = np.array([((Q/(Q-1))*(l-1/Q))**2 if l > 1/Q else 0 for l in lam])
out['mca_lam'] = lam[:4].round(4).tolist()
out['benz_pct'] = (benz/benz.sum()*100)[:3].round(1).tolist()
gtot = Q/(Q-1)*((lam**2).sum() - (J-Q)/Q**2)
out['greenacre_pct'] = (benz/gtot*100)[:3].round(1).tolist()
F_row = U[:, :2]*sv[:2]/np.sqrt(r)[:, None]          # row principal coords
G_col = Vt[:2].T*sv[:2]/np.sqrt(c)[:, None]          # column principal coords
# eta2 per variable per dimension
std_row = U[:, :2]/np.sqrt(r)[:, None]
eta = np.zeros((7, 2))
for k in range(7):
    for d in range(2):
        x = F_row[:, d]; g = X[:, k]
        ssb = sum(((g == v).sum())*(x[g == v].mean()-x.mean())**2 for v in set(g))
        eta[k, d] = ssb/((x-x.mean())**2).sum()*lam[d]/lam[d] if True else 0
# eta2 as discrimination measure = correlation ratio of row std coords
for k in range(7):
    for d in range(2):
        x = std_row[:, d]; g = X[:, k]
        eta[k, d] = sum(((g == v).sum())*(x[g == v].mean()-x.mean())**2 for v in set(g))/((x-x.mean())**2).sum()
out['eta'] = dict(zip(ind, eta.round(3).tolist())); out['eta_mean'] = eta.mean(0).round(3).tolist()

# ---------- Kendall
tau = {}
for a, b in itertools.combinations(range(7), 2):
    t, p = stats.kendalltau(X[:, a], X[:, b]); tau[f"{ind[a]}-{ind[b]}"] = (round(t, 2), round(p, 3))
out['tau_sel'] = {k: tau[k] for k in ["LIT-NUM", "KAR-BINEKA", "NUM-AMAN", "LIT-AMAN", "KAR-AMAN", "AMAN-BINEKA"]}

# ---------- clustering
sq = np.sqrt(D)
Lw = linkage(squareform(sq, checks=False), 'ward')
La = linkage(squareform(D, checks=False), 'average')
coph = cophenet(Lw, squareform(sq, checks=False))[0]
out['coph'] = round(coph, 2)
part = {k: fcluster(Lw, k, 'maxclust') for k in range(2, 7)}
out['ari_ward_avg_k3'] = adjusted_rand_score(part[3], fcluster(La, 3, 'maxclust'))
out['sil_gower'] = {k: round(silhouette_score(D, part[k], metric='precomputed'), 2) for k in range(2, 7)}
out['sil_sqrt'] = {k: round(silhouette_score(sq, part[k], metric='precomputed'), 2) for k in range(2, 7)}
out['merge_heights'] = Lw[-6:, 2].round(3).tolist()
out['k3_equals_given'] = adjusted_rand_score(part[3], cl)
# unique profiles
uniq, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
out['n_unique'] = len(uniq)
out['twins'] = [[names[i] for i in np.where(inv.ravel() == u)[0]] for u in range(len(uniq)) if cnt[u] > 1]

# bootstrap Jaccard (clusterboot style)
B = 2000; jac = np.zeros((B, 3)); jac_cnt = np.zeros(3)
for b in range(B):
    idx = rng.choice(N, N, replace=True)
    Db = sq[np.ix_(idx, idx)]
    if len(set(map(tuple, X[idx]))) < 3: continue
    Lb = linkage(squareform(Db, checks=False), 'ward')
    pb = fcluster(Lb, 3, 'maxclust')
    ui = np.unique(idx)
    for c_ in range(1, 4):
        A = set(np.where(cl == c_)[0]) & set(ui)
        if not A: jac[b, c_-1] = np.nan; continue
        best = 0
        for k in range(1, 4):
            Bk = set(idx[pb == k])
            best = max(best, len(A & Bk)/len(A | Bk))
        jac[b, c_-1] = best
out['jaccard_mean'] = np.nanmean(jac, 0).round(2).tolist()

# ---------- PERMANOVA
def pseudoF(D, g):
    D2 = D**2; N = len(g); groups = np.unique(g); a = len(groups)
    SST = D2[np.triu_indices(N, 1)].sum()/N
    SSW = sum(D2[np.ix_(g == k, g == k)][np.triu_indices((g == k).sum(), 1)].sum()/(g == k).sum() for k in groups)
    SSA = SST-SSW
    return (SSA/(a-1))/(SSW/(N-a)), SSA/SST
res = {}
for nm, g in [('jalur', jalur), ('kew', kew), ('jenjang', jenjang), ('subs', subs)]:
    F, R2 = pseudoF(D, g)
    perm = np.array([pseudoF(D, rng.permutation(g))[0] for _ in range(9999)])
    p = (1+(perm >= F-1e-12).sum())/10000
    a = len(set(g)); R2adj = 1-(1-R2)*(N-1)/(N-a)
    res[nm] = [round(F, 2), round(R2, 3), round(R2adj, 3), round(p, 4)]
out['permanova'] = res

# restricted (subsystem-level) permutation for jalur: choose 2 of 6 subsystems as nonformal
Fobs, _ = pseudoF(D, jalur); Fs = []
for comb in itertools.combinations(range(6), 2):
    g = np.isin(subs, comb).astype(int); Fs.append(pseudoF(D, g)[0])
Fs = np.array(Fs)
out['restricted_jalur'] = dict(n=len(Fs), rank=int((Fs >= Fobs-1e-12).sum()), p=round((Fs >= Fobs-1e-12).mean(), 3),
                               Fsorted=np.sort(Fs)[::-1].round(2).tolist())
Fk, _ = pseudoF(D, kew); Fs2 = []
for comb in itertools.combinations(range(6), 3):
    g = np.isin(subs, comb).astype(int); Fs2.append(pseudoF(D, g)[0])
Fs2 = np.array(Fs2)
out['restricted_kew'] = dict(n=len(Fs2), p=round((Fs2 >= Fk-1e-12).mean(), 3))

# within-jalur vs within-jenjang mean distances
def mean_within(g):
    v = [D[i, j] for i in range(N) for j in range(i+1, N) if g[i] == g[j]]; return np.mean(v)
def mean_between(g):
    v = [D[i, j] for i in range(N) for j in range(i+1, N) if g[i] != g[j]]; return np.mean(v)
out['dist_within_jalur'] = round(mean_within(jalur), 3); out['dist_between_jalur'] = round(mean_between(jalur), 3)
out['dist_within_jenjang'] = round(mean_within(jenjang), 3); out['dist_between_jenjang'] = round(mean_between(jenjang), 3)
# example distances
i = names.index
out['d_SDKes_SMAKes'] = round(D[i('SD Kesetaraan'), i('SMA Kesetaraan')], 3)
out['d_SDKes_SDUmum'] = round(D[i('SD Kesetaraan'), i('SD Umum')], 3)
nn = np.argsort(D[i('SD Kesetaraan')])[1:6]
out['nn_SDKes'] = [(names[j], round(D[i('SD Kesetaraan'), j], 3)) for j in nn]
out['d_SMPKes_SMPUmum'] = round(D[i('SMP Kesetaraan'), i('SMP Umum')], 3)
out['d_SMPKes_MTs'] = round(D[i('SMP Kesetaraan'), i('MTs')], 3)
out['d_PKSDSetara_PKSMASetara'] = round(D[i('PK SD Setara'), i('PK SMA Setara')], 3)
out['d_PKSMPSetara_SMPPK'] = round(D[i('PK SMP Setara'), i('SMP Pendidikan Keagamaan')], 3)
out['d_PKSMPSetara_PKSDSetara'] = round(D[i('PK SMP Setara'), i('PK SD Setara')], 3)

# ---------- LDA
Xl = X[:, :6].astype(float); y = jalur
def lda(): return LDA(solver='lsqr', shrinkage='auto')
pred = []
for k in range(N):
    m = np.ones(N, bool); m[k] = False
    pred.append(lda().fit(Xl[m], y[m]).predict(Xl[k:k+1])[0])   # shrinkage re-estimated in each fold
pred = np.array(pred); acc = (pred == y).mean()
out['loocv_acc'] = round(acc, 3); out['misclass'] = [names[j] for j in np.where(pred != y)[0]]
p1 = y.mean(); out['Cpro'] = round(p1**2+(1-p1)**2, 3); out['Cmax'] = round(1-p1, 3); out['1.25Cmax'] = round(1.25*(1-p1), 3)
out['binom_p_vs_Cmax'] = round(stats.binom.sf(int(acc*N)-1, N, 1-p1), 3)
out['binom_p_vs_Cpro'] = round(stats.binom.sf(int(acc*N)-1, N, p1**2+(1-p1)**2), 3)
nc = int(round(acc*N)); out['PressQ'] = round((N-nc*2)**2/(N*1), 2)
# permutation test of LOOCV accuracy
def loo_acc(yy):
    pr = []
    for k in range(N):
        m = np.ones(N, bool); m[k] = False
        if len(set(yy[m])) < 2: pr.append(yy[m][0]); continue
        pr.append(lda().fit(Xl[m], yy[m]).predict(Xl[k:k+1])[0])
    return (np.array(pr) == yy).mean()
pa = np.array([loo_acc(rng.permutation(y)) for _ in range(999)])
out['perm_p_loocv'] = round((1+(pa >= acc-1e-12).sum())/1000, 3)
# structure coefficients: corr between X and discriminant score
full = lda().fit(Xl, y)
score = Xl @ full.coef_.ravel()
sc = np.array([abs(np.corrcoef(Xl[:, k], score)[0, 1]) for k in range(6)])
out['struct'] = dict(zip(ind[:6], sc.round(2).tolist()))
bs = []
for b in range(2000):
    idx = rng.choice(N, N, replace=True)
    if len(set(y[idx])) < 2 or (y[idx] == 1).sum() < 2: continue
    Xb = Xl[idx]
    if np.any(Xb.std(0) == 0): continue
    m_ = lda().fit(Xb, y[idx]); s_ = Xb @ m_.coef_.ravel()
    bs.append([np.corrcoef(Xb[:, k], s_)[0, 1]*np.sign(np.corrcoef(Xb[:, 5], s_)[0, 1]) for k in range(6)])
bs = np.array(bs)
out['struct_ci'] = {ind[k]: [round(np.percentile(abs(bs[:, k]), 2.5), 2), round(np.percentile(abs(bs[:, k]), 97.5), 2)] for k in range(6)}
out['struct_ci_signed'] = {ind[k]: [round(np.percentile(bs[:, k], 2.5), 2), round(np.percentile(bs[:, k], 97.5), 2)] for k in range(6)}
out['n_boot_lda'] = len(bs)
# Wilks
from numpy.linalg import det
W = sum(np.cov(Xl[y == k].T, bias=False)*((y == k).sum()-1) for k in (0, 1))
T = np.cov(Xl.T)*(N-1)
wl = det(W)/det(T); out['wilks'] = round(wl, 3)
chi = -(N-1-(6+2)/2)*np.log(wl); out['wilks_chi'] = round(chi, 2); out['wilks_p'] = round(stats.chi2.sf(chi, 6), 3)
out['can_corr'] = round(np.sqrt(1-wl), 2)

# exact permutation tests per indicator
combos = list(itertools.combinations(range(N), 6)); out['n_combos'] = len(combos)
pex = {}
for k in range(7):
    rk = stats.rankdata(X[:, k]); obs = rk[jalur == 0].mean()-rk[jalur == 1].mean()
    cnt_ = 0
    for cmb in combos:
        g = np.zeros(N, bool); g[list(cmb)] = True
        if abs(rk[~g].mean()-rk[g].mean()) >= abs(obs)-1e-12: cnt_ += 1
    pex[ind[k]] = round(cnt_/len(combos), 3)
out['p_exact'] = pex
print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
json.dump(out, open('hasil_replikasi.json', 'w'), indent=1, default=float)
