"""common_drills.py의 모든 함수를 scikit-learn · PyTorch · 손계산 값과 비교해 확인. 실행: python common_drills_check.py"""
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from common_drills import *

rng = np.random.default_rng(0)
ok = lambda name, cond: print(("통과 " if cond else "실패 ") + name)

# 12-2 지표
e_y, e_p = np.array([1, 2, 3, 13.0]), np.zeros(4) + np.array([0, 1, 2, 4])
ok(f"MAE·RMSE = sklearn ({mae(e_y, e_p):.3f}, {rmse(e_y, e_p):.3f})",
   np.isclose(mae(e_y, e_p), mean_absolute_error(e_y, e_p)) and np.isclose(rmse(e_y, e_p), np.sqrt(mean_squared_error(e_y, e_p))))
y = np.array([1] * 60 + [0] * 940); p = np.array([1] * 40 + [0] * 20 + [1] * 10 + [0] * 930)
pr, rc, f = precision_recall_f1(y, p)
tn, fp, fn, tp = confusion_matrix(y, p).ravel()
ok(f"정밀도·재현율·F1 = sklearn ({pr:.3f}, {rc:.3f}, {f:.3f}), 혼동행렬 = sklearn",
   np.allclose([pr, rc, f], [precision_score(y, p), recall_score(y, p), f1_score(y, p)]) and confusion(y, p) == (tp, fp, fn, tn))

# 12-3 선형 회귀
X = rng.normal(size=(200, 3)); yr = X @ np.array([2.0, -1.0, 0.5]) + 3 + 0.1 * rng.normal(size=200)
w_gd, b_gd = linreg_gd(X, yr); w_ne, b_ne = linreg_normal(X, yr)
ok(f"GD = 정규방정식 (w {np.round(w_gd, 3)}, b {b_gd:.3f})", np.allclose(w_gd, w_ne, atol=1e-3) and np.isclose(b_gd, b_ne, atol=1e-3))
w_r, b_r = linreg_normal(X, yr, lam=10.0); sk = Ridge(alpha=10.0).fit(X, yr)
ok("Ridge 정규방정식 = sklearn Ridge(alpha=10)", np.allclose(w_r, sk.coef_) and np.isclose(b_r, sk.intercept_))
w_rg, b_rg = linreg_gd(X, yr, lam=10.0 / 200, epochs=5000)     # GD의 MSE는 1/n 평균이라 λ를 n으로 나눠야 같은 문제
ok("Ridge GD(λ/n) = Ridge 정규방정식", np.allclose(w_rg, w_r, atol=1e-3))

# 12-4 로지스틱
yc = (X[:, 0] - X[:, 1] > 0).astype(float); wl, bl = logreg_gd(X, yc)
ok(f"로지스틱 정확도 {np.mean((sigmoid(X @ wl + bl) > 0.5) == yc):.3f}", np.mean((sigmoid(X @ wl + bl) > 0.5) == yc) > 0.97)

# 12-5 표준화
Xtr, Xte = X[:150] * 5 + 2, X[150:] * 5 + 2
sc = Standardizer().fit(Xtr); skc = StandardScaler().fit(Xtr)
ok("Standardizer = sklearn StandardScaler (학습 데이터로 fit, 테스트는 transform)", np.allclose(sc.transform(Xte), skc.transform(Xte)))

# 12-6 지니·분할
ok("지니(0.5, 0.5) = 0.5, 순수 = 0", np.isclose(gini(np.array([0, 1])), 0.5) and gini(np.array([1, 1])) == 0)
j, t, sc_ = best_split(np.round(X[:, :2], 2), (X[:, 1] > 0.3).astype(int))
ok(f"best_split이 진짜 기준(특성 1, 0.3 근처)을 찾음: 특성 {j}, 임계값 {t}", j == 1 and abs(t - 0.3) < 0.02 and sc_ == 0)

# 12-7 K-fold
folds = list(kfold_indices(10, 3))
ok(f"K-fold 크기 {[(len(a), len(b)) for a, b in folds]}, 검증 조각이 겹치지 않고 전체를 덮음",
   sorted(np.concatenate([b for _, b in folds]).tolist()) == list(range(10)))

# 12-8 그래디언트 부스팅
yb = np.sin(X[:, 0]) * 3 + X[:, 1] ** 2 + 0.1 * rng.normal(size=200)
f0, stumps = gradient_boosting(X, yb, n_trees=200, lr=0.1)
mse0, mse1 = np.mean((yb - yb.mean()) ** 2), np.mean((yb - gb_predict(f0, stumps, X)) ** 2)
ok(f"부스팅: 평균 예측 MSE {mse0:.3f} → stump 200개 MSE {mse1:.3f}", mse1 < 0.2 * mse0)

# 12-9 XGBoost (손계산: 제곱 오차 ½(y−ŷ)², 현재 예측 0 → g = −y, h = 1)
yy = np.array([1.0, 2.0, 3.0, 10.0]); g, h = -yy, np.ones(4)
w_all = xgb_leaf_weight(g, h, lam=1.0)                       # −(−16)/(4+1) = 3.2
gain = xgb_gain(g[:3], h[:3], g[3:], h[3:], lam=1.0, gamma=0.0)   # ½[36/4 + 100/2 − 256/5] = 3.9
ok(f"XGBoost 잎 가중치 {w_all:.2f}(손계산 3.2), 분할 이득 {gain:.2f}(손계산 3.9)", np.isclose(w_all, 3.2) and np.isclose(gain, 3.9))

# 12-10 softmax · CE
z = rng.normal(size=(4, 3)); yy3 = np.array([0, 2, 1, 0])
ce_t = torch.nn.CrossEntropyLoss()(torch.tensor(z), torch.tensor(yy3)).item()
ok(f"교차엔트로피 = torch ({cross_entropy(z, yy3):.6f})", np.isclose(cross_entropy(z, yy3), ce_t))
ok("softmax 큰 값에서도 안정 (1000, 1001)", np.all(np.isfinite(softmax(np.array([[1000.0, 1001.0]])))))

# 12-11 Adam
wa, m, v = np.array([5.0]), 0.0, 0.0
for tt in range(1, 11): wa, m, v = adam_step(wa, 2 * wa, m, v, tt, lr=0.05)
pt = torch.tensor([5.0], requires_grad=True); o = torch.optim.Adam([pt], lr=0.05)
for _ in range(10): o.zero_grad(); (pt ** 2).sum().backward(); o.step()
ok(f"Adam 10스텝 = torch.optim.Adam ({wa[0]:.6f} vs {pt.item():.6f})", np.isclose(wa[0], pt.item(), atol=1e-6))

# 12-12 역전파 = PyTorch autograd
Xn, yn = rng.normal(size=(8, 5)), rng.integers(0, 3, 8)
W1, b1, W2, b2 = rng.normal(size=(5, 6)), rng.normal(size=6), rng.normal(size=(6, 3)), rng.normal(size=3)
loss, grads = mlp_forward_backward(Xn, yn, W1, b1, W2, b2)
tp_ = [torch.tensor(a, requires_grad=True) for a in (W1, b1, W2, b2)]
lt = torch.nn.functional.cross_entropy(torch.relu(torch.tensor(Xn) @ tp_[0] + tp_[1]) @ tp_[2] + tp_[3], torch.tensor(yn))
lt.backward()
ok(f"2층 신경망 손실·기울기 4개 = torch autograd (손실 {loss:.4f})",
   np.isclose(loss, lt.item()) and all(np.allclose(gm, tt_.grad.numpy()) for gm, tt_ in zip(grads, tp_)))

# 12-13 차분·ACF
rw = np.cumsum(rng.normal(size=3000))
ok(f"랜덤워크 ACF(1) {acf(rw, 1):.3f} → 차분 후 {acf(diff(rw), 1):.3f}", acf(rw, 1) > 0.95 and abs(acf(diff(rw), 1)) < 0.05)
from statsmodels.tsa.stattools import acf as sm_acf
ok("acf = statsmodels acf", np.isclose(acf(rw, 3), sm_acf(rw, nlags=3)[3]))

# 12-14 AR 최소제곱 (참값 복원)
ar = np.zeros(3000)
for k in range(2, 3000): ar[k] = 0.5 + 0.7 * ar[k - 1] - 0.3 * ar[k - 2] + rng.normal()
c_h, ph_h = fit_ar(ar, 2)
ok(f"AR(2) 최소제곱 c {c_h:.3f}, φ {np.round(ph_h, 3)} (참값 0.5, [0.7, −0.3])", np.allclose(ph_h, [0.7, -0.3], atol=0.05))
