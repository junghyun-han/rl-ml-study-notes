"""1교시 주관식 대비 코드 (numpy만 사용). 노트 12장에 이 파일이 그대로 실린다.
검증: python common_drills_check.py
"""
import numpy as np


# ── 12-2 평가 지표 · 혼동행렬 ──────────────────────────────────────────
def mae(y, p):
    return np.mean(np.abs(y - p))


def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2))


def confusion(y, p):
    """이진 분류 혼동행렬 칸 4개 (y = 정답, p = 예측, 둘 다 0/1 배열)"""
    tp = np.sum((p == 1) & (y == 1))
    fp = np.sum((p == 1) & (y == 0))
    fn = np.sum((p == 0) & (y == 1))
    tn = np.sum((p == 0) & (y == 0))
    return tp, fp, fn, tn


def precision_recall_f1(y, p):
    tp, fp, fn, tn = confusion(y, p)
    prec = tp / (tp + fp) if tp + fp else 0.0          # 양성 예측이 없으면 0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


# ── 12-3 선형 회귀: 경사하강법 · 정규방정식 (Ridge 포함) ─────────────────
def linreg_gd(X, y, lr=0.1, epochs=1000, lam=0.0):
    """MSE + lam·Σw² 를 경사하강법으로 최소화 (lam > 0 이면 Ridge)"""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        err = X @ w + b - y                     # 예측 − 정답
        gw = (2 / n) * X.T @ err + 2 * lam * w  # MSE 기울기 + L2 벌점 기울기 (절편엔 벌점 없음)
        gb = (2 / n) * err.sum()
        w -= lr * gw
        b -= lr * gb
    return w, b


def linreg_normal(X, y, lam=0.0):
    """정규방정식: w = (XᵀX + λI)⁻¹ Xᵀy. 1 열을 붙여 절편을 첫 번째 가중치로."""
    n = X.shape[0]
    Xb = np.c_[np.ones(n), X]
    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0                               # 절편에는 벌점을 주지 않음
    wb = np.linalg.solve(Xb.T @ Xb + lam * I, Xb.T @ y)
    return wb[1:], wb[0]                        # (가중치, 절편)


# ── 12-4 로지스틱 회귀 ─────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logreg_gd(X, y, lr=0.5, epochs=500):
    """이진 교차엔트로피(BCE)를 경사하강법으로 최소화"""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)                  # 양성 확률
        gw = X.T @ (p - y) / n                  # BCE 기울기 (선형 회귀와 같은 모양)
        gb = np.mean(p - y)
        w -= lr * gw
        b -= lr * gb
    return w, b


# ── 12-5 표준화 (누수 없이: 학습 데이터로만 fit) ─────────────────────────
class Standardizer:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-12        # 0으로 나누기 방지
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


# ── 12-6 결정트리 분할: 지니 불순도 · 최적 분할 ─────────────────────────
def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    q = counts / counts.sum()
    return 1 - np.sum(q ** 2)


def best_split(X, y):
    """모든 특성 × 모든 임계값 중 자식 지니의 가중 평균이 가장 작은 분할"""
    best = (None, None, np.inf)                 # (특성 번호, 임계값, 점수)
    for j in range(X.shape[1]):
        for t in np.unique(X[:, j]):
            left, right = y[X[:, j] <= t], y[X[:, j] > t]
            if len(left) == 0 or len(right) == 0:
                continue
            score = len(left) / len(y) * gini(left) + len(right) / len(y) * gini(right)
            if score < best[2]:
                best = (j, t, score)
    return best


# ── 12-7 K-fold 나누기 ─────────────────────────────────────────────────
def kfold_indices(n, k, seed=0):
    idx = np.random.default_rng(seed).permutation(n)     # 시계열이면 섞지 않는다
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val


# ── 12-8 그래디언트 부스팅 (제곱 오차, 깊이 1 트리 = stump) ──────────────
def fit_stump(X, r):
    """잔차 r을 가장 잘 맞추는 '질문 하나짜리 트리': 왼쪽·오른쪽 각각 평균으로 예측"""
    best = (None, None, 0.0, 0.0, np.inf)       # (특성, 임계값, 왼쪽 값, 오른쪽 값, 제곱 오차)
    for j in range(X.shape[1]):
        for t in np.unique(X[:, j])[:-1]:       # 마지막 값은 오른쪽이 비므로 제외
            left = X[:, j] <= t
            lv, rv = r[left].mean(), r[~left].mean()
            sse = np.sum((r[left] - lv) ** 2) + np.sum((r[~left] - rv) ** 2)
            if sse < best[4]:
                best = (j, t, lv, rv, sse)
    return best[:4]


def predict_stump(stump, X):
    j, t, lv, rv = stump
    return np.where(X[:, j] <= t, lv, rv)


def gradient_boosting(X, y, n_trees=100, lr=0.1):
    """F₀ = 평균,  F_m = F_(m−1) + lr · (잔차 y − F_(m−1) 에 맞춘 stump)"""
    f0 = y.mean()
    F = np.full(len(y), f0)
    stumps = []
    for _ in range(n_trees):
        r = y - F                               # 제곱 오차의 음의 기울기 = 잔차
        s = fit_stump(X, r)
        F += lr * predict_stump(s, X)
        stumps.append(s)
    return f0, stumps


def gb_predict(f0, stumps, X, lr=0.1):
    return f0 + lr * sum(predict_stump(s, X) for s in stumps)


# ── 12-9 XGBoost 잎 가중치 · 분할 이득 ──────────────────────────────────
def xgb_leaf_weight(g, h, lam=1.0):
    """w* = −G / (H + λ)"""
    return -g.sum() / (h.sum() + lam)


def xgb_gain(g_left, h_left, g_right, h_right, lam=1.0, gamma=0.0):
    """Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] − γ"""
    GL, HL, GR, HR = g_left.sum(), h_left.sum(), g_right.sum(), h_right.sum()
    score = lambda G, H: G ** 2 / (H + lam)
    return 0.5 * (score(GL, HL) + score(GR, HR) - score(GL + GR, HL + HR)) - gamma


# ── 12-10 softmax · 교차엔트로피 ────────────────────────────────────────
def softmax(z):                                 # z: (배치, 클래스)
    z = z - z.max(axis=1, keepdims=True)        # 수치 안정 (결과는 같음)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(z, y):                        # y: 정답 클래스 번호 (배치,)
    p = softmax(z)
    return -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))


# ── 12-11 Adam 한 스텝 ─────────────────────────────────────────────────
def adam_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g                   # 1차: 기울기 이동평균
    v = b2 * v + (1 - b2) * g ** 2              # 2차: 기울기 제곱 이동평균
    m_hat = m / (1 - b1 ** t)                   # 편향 보정 (t는 1부터)
    v_hat = v / (1 - b2 ** t)
    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w, m, v


# ── 12-12 2층 신경망 순전파 · 역전파 (numpy) ─────────────────────────────
def mlp_forward_backward(X, y, W1, b1, W2, b2):
    """X(n,d) → ReLU 은닉층 → softmax 출력, 교차엔트로피 손실과 모든 기울기"""
    n = X.shape[0]
    # 순전파
    a1 = X @ W1 + b1                            # (n, h)
    h1 = np.maximum(0, a1)                      # ReLU
    z = h1 @ W2 + b2                            # (n, k) logit
    p = softmax(z)
    loss = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
    # 역전파 (연쇄법칙, 출력 → 입력 순서)
    dz = p.copy()
    dz[np.arange(n), y] -= 1                    # softmax + CE 의 기울기 = p − 정답(원-핫)
    dz /= n                                     # 평균 손실이므로 n으로 나눔
    dW2 = h1.T @ dz
    db2 = dz.sum(axis=0)
    dh1 = dz @ W2.T                             # 은닉층으로 전달
    da1 = dh1 * (a1 > 0)                        # ReLU 미분: 양수면 1, 아니면 0
    dW1 = X.T @ da1
    db1 = da1.sum(axis=0)
    return loss, (dW1, db1, dW2, db2)


# ── 12-13 차분 · ACF ──────────────────────────────────────────────────
def diff(y, lag=1):
    return y[lag:] - y[:-lag]                   # lag=12면 계절 차분


def acf(y, k):
    """시차 k의 표본 자기상관"""
    y = y - y.mean()
    return np.sum(y[k:] * y[:-k]) / np.sum(y * y)


# ── 12-14 AR(p) 최소제곱 적합과 예측 ────────────────────────────────────
def fit_ar(y, p):
    """y_t = c + φ₁y_(t−1) + … + φ_p y_(t−p) + ε 를 최소제곱으로 추정"""
    rows = [y[p - i - 1:len(y) - i - 1] for i in range(p)]     # i번째 열 = y_(t−1−i)
    X = np.column_stack([np.ones(len(y) - p)] + rows)
    coef, *_ = np.linalg.lstsq(X, y[p:], rcond=None)
    return coef[0], coef[1:]                    # (c, [φ₁ … φ_p])


def forecast_ar(y, c, phi, steps):
    hist = list(y[-len(phi):])
    out = []
    for _ in range(steps):
        nxt = c + sum(phi[i] * hist[-1 - i] for i in range(len(phi)))   # 예측값을 다시 입력으로
        out.append(nxt)
        hist.append(nxt)
    return np.array(out)
