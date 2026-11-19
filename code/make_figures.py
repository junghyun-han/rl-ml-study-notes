"""노트 그림 11장 생성 (code/results의 실제 실험 결과 사용). 실행: cd code && python make_figures.py
먼저 실험 스크립트들을 돌려 results/를 채워 둬야 한다(저장소에는 이미 들어 있음)."""
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": ["AppleGothic", "Malgun Gothic", "NanumGothic", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "fig"
RES = HERE / "results"
C = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]


def save(name):
    plt.tight_layout(); plt.savefig(OUT / f"{name}.png", dpi=160); plt.close()


# 1) 과소·적정·과적합
rng = np.random.default_rng(1)
x = np.sort(rng.uniform(0, 1, 15)); y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 15); xs = np.linspace(0, 1, 200)
fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
for a, deg, t in zip(ax, [1, 3, 12], ["과소적합 (1차)", "적정 (3차)", "과적합 (12차)"]):
    coef = np.polyfit(x, y, deg); a.scatter(x, y, s=14, color="#444"); a.plot(xs, np.sin(2 * np.pi * xs), "--", color="#999", lw=1)
    a.plot(xs, np.polyval(coef, xs), color=C[0], lw=2); a.set_ylim(-1.8, 1.8); a.set_title(t); a.set_xticks([]); a.set_yticks([])
save("fit")

# 2) L1 vs L2 제약
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.6))
w1, w2 = np.meshgrid(np.linspace(-1.5, 3, 300), np.linspace(-1.5, 3, 300))
loss = 1.2 * (w1 - 1.9) ** 2 + 0.6 * (w2 - 1.1) ** 2 + 0.8 * (w1 - 1.9) * (w2 - 1.1)
for a, kind in zip(ax, ["L1 (Lasso): 마름모", "L2 (Ridge): 원"]):
    a.contour(w1, w2, loss, levels=[0.3, 0.8, 1.6, 2.6, 3.8], colors=C[1], linewidths=1)
    if kind.startswith("L1"):
        a.fill([1, 0, -1, 0], [0, 1, 0, -1], color=C[0], alpha=0.25); a.plot(1, 0, "ko"); a.annotate("w₂ = 0 (모서리)", (1, 0), (1.3, -0.9), arrowprops=dict(arrowstyle="->"))
    else:
        t = np.linspace(0, 2 * np.pi, 200); a.fill(np.cos(t), np.sin(t), color=C[0], alpha=0.25)
        a.plot(0.87, 0.5, "ko"); a.annotate("둘 다 0 아님", (0.87, 0.5), (1.3, -0.9), arrowprops=dict(arrowstyle="->"))
    a.plot(1.9, 1.1, "r*", ms=10); a.text(1.95, 1.2, "규제 없는 최적점", fontsize=8)
    a.axhline(0, color="#aaa", lw=0.6); a.axvline(0, color="#aaa", lw=0.6); a.set_aspect("equal"); a.set_title(kind); a.set_xlabel("w₁"); a.set_ylabel("w₂")
save("l1l2")

# 3) ACF / PACF (AR(1), MA(1) 시뮬레이션)
def acf(y, k):
    y = y - y.mean(); return np.sum(y[k:] * y[:-k]) / np.sum(y * y)
def pacf(y, kmax):
    out = []
    for k in range(1, kmax + 1):
        X = np.column_stack([y[k - j - 1:len(y) - j - 1] for j in range(k)]); out.append(np.linalg.lstsq(np.c_[np.ones(len(X)), X], y[k:], rcond=None)[0][-1])
    return np.array(out)
n = 3000; e = rng.normal(size=n + 1)
ar = np.zeros(n); [ar.__setitem__(t, 0.7 * ar[t - 1] + e[t]) for t in range(1, n)]
ma = e[1:] + 0.7 * e[:-1]
fig, ax = plt.subplots(2, 2, figsize=(8, 4.4)); lags = np.arange(1, 11); band = 1.96 / np.sqrt(n)
for r_, (yy, nm) in enumerate([(ar, "AR(1), φ=0.7"), (ma, "MA(1), θ=0.7")]):
    for c_, (vals, kind) in enumerate([([acf(yy, k) for k in lags], "ACF"), (pacf(yy, 10), "PACF")]):
        a = ax[r_][c_]; a.bar(lags, vals, color=C[0] if c_ == 0 else C[2], width=0.5); a.axhspan(-band, band, color="#ddd", zorder=0)
        a.axhline(0, color="#666", lw=0.6); a.set_ylim(-0.6, 0.8); a.set_xticks(lags); a.set_title(f"{nm} — {kind}")
save("acf")

# 4) 활성화 함수
xx = np.linspace(-4, 4, 400); sig = 1 / (1 + np.exp(-xx))
from math import erf
gelu = np.array([v * 0.5 * (1 + erf(v / np.sqrt(2))) for v in xx])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for f, nm, c in [(sig, "Sigmoid", C[0]), (np.tanh(xx), "Tanh", C[1]), (np.maximum(0, xx), "ReLU", C[2]), (np.where(xx > 0, xx, 0.1 * xx), "Leaky ReLU (α=0.1)", C[3]), (gelu, "GELU", C[4])]:
    ax[0].plot(xx, f, label=nm, color=c, lw=1.8)
ax[0].set_ylim(-1.5, 3); ax[0].legend(fontsize=8); ax[0].set_title("함수"); ax[0].axhline(0, color="#aaa", lw=0.5)
for f, nm, c in [(sig * (1 - sig), "Sigmoid′ (최대 0.25)", C[0]), (1 - np.tanh(xx) ** 2, "Tanh′ (최대 1)", C[1]), ((xx > 0).astype(float), "ReLU′ (0 또는 1)", C[2])]:
    ax[1].plot(xx, f, label=nm, color=c, lw=1.8)
ax[1].legend(fontsize=8); ax[1].set_title("미분 (기울기)"); ax[1].set_ylim(-0.1, 1.2)
save("act")

# 5) 강화학습 루프 도식
fig, ax = plt.subplots(figsize=(7, 2.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
for (x0, lab, col) in [(0.6, "에이전트\n(정책 π)", C[0]), (6.4, "환경\n(시뮬레이터·로봇)", C[2])]:
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x0, 1.2), 3, 1.6, boxstyle="round,pad=0.1", fc=col, alpha=0.15, ec=col)); ax.text(x0 + 1.5, 2.0, lab, ha="center", va="center", fontsize=11)
ax.annotate("", (6.4, 2.5), (3.6, 2.5), arrowprops=dict(arrowstyle="-|>", lw=1.6)); ax.text(5, 2.75, r"행동 $a_t$", ha="center")
ax.annotate("", (3.6, 1.5), (6.4, 1.5), arrowprops=dict(arrowstyle="-|>", lw=1.6)); ax.text(5, 0.95, r"다음 상태 $s_{t+1}$, 보상 $r_{t+1}$", ha="center")
ax.text(5, 3.6, "목표: 누적 보상 G = r₁ + γr₂ + γ²r₃ + … 를 최대로 하는 π", ha="center", fontsize=10, color="#333")
save("rlloop")

# 6) PPO clip
r = np.linspace(0, 2, 400); eps = 0.2
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3))
for a, A, t in [(ax[0], 1, r"$\hat{A}>0$ (좋은 행동)"), (ax[1], -1, r"$\hat{A}<0$ (나쁜 행동)")]:
    a.plot(r, r * A, "--", color="#999", label=r"clip 없음  $r\hat{A}$"); a.plot(r, np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A), color=C[0], lw=2.2, label=r"$L^{CLIP}=\min(r\hat{A},\ \mathrm{clip}(r)\hat{A})$")
    for v in [1 - eps, 1, 1 + eps]: a.axvline(v, color="#ccc", lw=0.8)
    a.set_xlabel(r"확률 비율 $r=\pi_\theta/\pi_{old}$"); a.set_title(t); a.legend(fontsize=7.5, loc="upper left" if A > 0 else "lower left")
    a.text(1 + eps, a.get_ylim()[0] + 0.1, "1+ε", fontsize=8); a.text(1 - eps - 0.12, a.get_ylim()[0] + 0.1, "1-ε", fontsize=8)
save("ppoclip")

# 7) RRT (실행 결과 그림 복사)
shutil.copy(RES / "rrt.png", OUT / "rrt.png")

# 8) DQN: 타깃망 유무 학습 곡선
def smooth(xs, ys, w=20):
    if len(ys) < w: return xs, ys
    k = np.ones(w) / w; return xs[w - 1:], np.convolve(ys, k, mode="valid")
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3), sharey=True)
for a, (tag, t) in zip(ax, [("dqn", "Double DQN + 타깃망 (기본)"), ("dqn_nodouble", "타깃망만 (Double 끔)"), ("dqn_notarget", "타깃망 끔")]):
    for s in range(3):
        d = np.loadtxt(RES / f"{tag}_s{s}.txt"); xs, ys = smooth(d[:, 0], d[:, 1]); a.plot(xs, ys, color=C[s], lw=1.3, label=f"시드 {s}")
    a.set_title(t, fontsize=10); a.set_xlabel("환경 스텝")
ax[0].set_ylabel("에피소드 리턴 (20판 이동평균)"); ax[0].legend(fontsize=8)
save("curve_dqn")

# 9) 정책 기반 비교: REINFORCE / Actor-Critic / PPO (CartPole)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.1))
for f, nm, c in [("reinforce.txt", "REINFORCE", C[0]), ("actor_critic.txt", "Actor-Critic (1-step)", C[1])]:
    y_ = np.loadtxt(RES / f); xs, ys = smooth(np.arange(1, len(y_) + 1), y_); ax[0].plot(xs, ys, color=c, label=nm)
ax[0].set_xlabel("에피소드"); ax[0].set_ylabel("리턴 (20판 이동평균)"); ax[0].legend(fontsize=8); ax[0].set_title("판 단위 업데이트")
d = np.loadtxt(RES / "ppo_lr0.0003_clip0.2_s0.txt"); xs, ys = smooth(d[:, 0], d[:, 1]); ax[1].plot(xs, ys, color=C[2], label="PPO (lr 3e-4, ε 0.2)")
ax[1].set_xlabel("환경 스텝"); ax[1].legend(fontsize=8); ax[1].set_title("PPO")
save("curve_pg")

# 10) 연속 행동: SAC vs PPO (Pendulum)
fig, ax = plt.subplots(figsize=(6.5, 3.1))
d = np.loadtxt(RES / "sac.txt"); xs, ys = smooth(d[:, 0], d[:, 1], 5); ax.plot(xs, ys, color=C[3], label="SAC (학습 중 리턴, 5판 이동평균)")
d = np.loadtxt(RES / "ppo_continuous.txt"); ax.plot(d[:, 0], d[:, 1], color=C[2], label="PPO (평가 리턴)")
ax.set_xscale("log"); ax.set_xlabel("환경 스텝 (로그 눈금)"); ax.set_ylabel("에피소드 리턴"); ax.legend(fontsize=8); ax.set_title("Pendulum-v1: 0에 가까울수록 좋음")
save("curve_cont")
print("figs:", sorted(p.name for p in OUT.iterdir()))

# 11) 밴딧: greedy / ε / UCB
d = np.load(RES / "bandit.npz")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.1))
for (k, lab), c in zip([("greedy_(ε=0)", "greedy (ε=0)"), ("ε=0.01", "ε=0.01"), ("ε=0.1", "ε=0.1"), ("UCB_c=2", "UCB c=2")], C):
    r, b = d[k]
    ax[0].plot(np.arange(1, len(r) + 1), r, color=c, lw=1.2, label=lab); ax[1].plot(np.arange(1, len(b) + 1), b * 100, color=c, lw=1.2, label=lab)
ax[0].set_title("평균 보상"); ax[1].set_title("최적 팔을 고른 비율 (%)")
for a in ax: a.set_xlabel("당긴 횟수"); a.legend(fontsize=8)
save("curve_bandit")
print("bandit fig ok")
