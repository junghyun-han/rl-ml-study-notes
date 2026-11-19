"""easy_snippets.py 확인: 이미 검증한 코드(common_drills·10·12·09)와 손계산·PyTorch 결과와 비교"""
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import easy_snippets as E
import common_drills as C

ok = lambda name, cond: print(("통과 " if cond else "실패 ") + name)
load = lambda f: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(importlib.util.spec_from_file_location(f[:-3], f))
rng = np.random.default_rng(0); torch.manual_seed(0)

# E1~E8: common_drills와 같은 값
y = np.array([1] * 60 + [0] * 940); p = np.array([1] * 40 + [0] * 20 + [1] * 10 + [0] * 930)
e = np.array([1, 1, 1, 9.0])
ok(f"E1 MAE 3·RMSE 4.583·F1 0.727", np.isclose(E.mae(e, 0), 3) and np.isclose(E.rmse(e, 0), np.sqrt(21)) and np.isclose(E.f1(y, p), C.precision_recall_f1(y, p)[2]))
X = rng.normal(size=(200, 3)); yr = X @ np.array([2.0, -1.0, 0.5]) + 3 + 0.1 * rng.normal(size=200)
ok("E2 선형 회귀 GD = common_drills 정규방정식", np.allclose(E.linreg_gd(X, yr)[0], C.linreg_normal(X, yr)[0], atol=1e-3))
yc = (X[:, 0] - X[:, 1] > 0).astype(float); wl, bl = E.logreg_gd(X, yc)
ok("E3 로지스틱 = common_drills 로지스틱", np.allclose(wl, C.logreg_gd(X, yc)[0]))
z = rng.normal(size=(5, 4)); yy = rng.integers(0, 4, 5)
ok("E4 교차엔트로피 = torch", np.isclose(E.cross_entropy(z, yy), nn.CrossEntropyLoss()(torch.tensor(z), torch.tensor(yy)).item()))
w, m, v = np.array([5.0]), 0.0, 0.0
for t in range(1, 11): w, m, v = E.adam_step(w, 2 * w, m, v, t, lr=0.05)
ok(f"E5 Adam 10스텝 {w[0]:.6f} (common_drills 검증값 4.501748)", np.isclose(w[0], 4.501748, atol=1e-6))
Xs = np.round(X[:, :2], 2); ys = (X[:, 1] > 0.3).astype(int)
ok("E6 최적 분할 = common_drills", E.best_split(Xs, ys)[:2] == C.best_split(Xs, ys)[:2])
ok("E7 K-fold 크기 (6,4)(7,3)(7,3)", [(len(a), len(b)) for a, b in E.kfold(10, 3)] == [(6, 4), (7, 3), (7, 3)])
tr, te = E.standardize(X[:150] * 5 + 2, X[150:] * 5 + 2)
ok("E7 표준화: 학습 데이터 평균 0·표준편차 1", np.allclose(tr.mean(0), 0) and np.allclose(tr.std(0), 1))
rw = np.cumsum(rng.normal(size=3000))
ok("E8 차분·ACF = common_drills", np.isclose(E.acf(E.diff(rw), 1), C.acf(C.diff(rw), 1)))

# E9 역전파 = torch autograd
Xn, yn = rng.normal(size=(8, 5)), rng.integers(0, 3, 8)
Ws = [rng.normal(size=(5, 6)), rng.normal(size=6), rng.normal(size=(6, 3)), rng.normal(size=3)]
loss, grads = E.mlp_grads(Xn, yn, *Ws)
tw = [torch.tensor(a, requires_grad=True) for a in Ws]
lt = nn.functional.cross_entropy(torch.relu(torch.tensor(Xn) @ tw[0] + tw[1]) @ tw[2] + tw[3], torch.tensor(yn)); lt.backward()
ok("E9 역전파 손실·기울기 4개 = torch autograd", np.isclose(loss, lt.item()) and all(np.allclose(g, t.grad.numpy()) for g, t in zip(grads, tw)))

# E10 PyTorch 루프
Xt = torch.randn(600, 6); yt = (Xt[:, 0] + Xt[:, 1] > 0).long()
_, acc = E.train_torch(Xt, yt, 2)
ok(f"E10 PyTorch 학습 루프 정확도 {acc:.3f}", acc > 0.9)

# R1 Q-learning·SARSA 한 번 갱신 (손계산)
Q = np.zeros((3, 2)); Q[1] = [1.0, 2.0]
E.q_learning_update(Q, 0, 0, 1.0, 1, False, alpha=0.5, gamma=0.9)      # 0 + 0.5*(1 + 0.9*2 − 0) = 1.4
Q2 = np.zeros((3, 2)); Q2[1] = [1.0, 2.0]
E.sarsa_update(Q2, 0, 0, 1.0, 1, 0, False, alpha=0.5, gamma=0.9)      # 0 + 0.5*(1 + 0.9*1 − 0) = 0.95
ok(f"R1 Q-learning {Q[0, 0]:.2f}(손계산 1.40) · SARSA {Q2[0, 0]:.2f}(손계산 0.95)", np.isclose(Q[0, 0], 1.4) and np.isclose(Q2[0, 0], 0.95))

# R2 할인 리턴
ok("R2 할인 리턴 [1,1,1], γ=0.5 → [1.75, 1.5, 1]", np.allclose(E.discounted_returns([1, 1, 1], 0.5), [1.75, 1.5, 1.0]))

# R3 DQN 손실: 손으로 계산한 타깃과 비교
q = nn.Linear(4, 2); qt = nn.Linear(4, 2)
s, a, r, s2, d = torch.randn(5, 4), torch.randint(0, 2, (5,)), torch.randn(5), torch.randn(5, 4), torch.tensor([0, 0, 1, 0, 0.])
with torch.no_grad():
    y_ref = r + 0.99 * (1 - d) * qt(s2).gather(1, q(s2).argmax(1, keepdim=True)).squeeze(1)
    ref = nn.functional.smooth_l1_loss(q(s).gather(1, a.unsqueeze(1)).squeeze(1), y_ref)
ok("R3 DQN(Double) 손실 = 손으로 만든 타깃", torch.isclose(E.dqn_loss(q, qt, s, a, r, s2, d), ref))

# R4 Actor-Critic: Actor 손실 기울기는 Critic으로 안 흐름
actor, critic = nn.Linear(4, 2), nn.Linear(4, 1)
al, cl = E.actor_critic_losses(actor, critic, s, a, r, s2, d); al.backward()
ok("R4 Actor 손실이 Critic 가중치에 기울기를 남기지 않음(detach)", critic.weight.grad is None)

# R5 GAE: 끝난 스텝에서 끊김, λ=0이면 1-step TD
adv, ret = E.gae([1, 1, 1], [0.5, 0.2, 0.1], [0, 1, 0], 0.3, gamma=0.9, lam=0.0)
ok("R5 GAE(λ=0) = 1-step TD, 끝난 스텝은 다음 값 안 씀", np.allclose(adv, [1 + 0.9 * 0.2 - 0.5, 1 - 0.2, 1 + 0.9 * 0.3 - 0.1]))

# R6 PPO 손실 = 05 파일의 식
nl, ol = torch.randn(8, requires_grad=True), torch.randn(8)
advt, vals, rets, ent = torch.randn(8), torch.randn(8), torch.randn(8), torch.rand(8)
ratio = (nl - ol).exp(); an = (advt - advt.mean()) / (advt.std() + 1e-8)
ref = -torch.min(ratio * an, torch.clamp(ratio, 0.8, 1.2) * an).mean() + 0.5 * ((vals - rets) ** 2).mean() - 0.01 * ent.mean()
ok("R6 PPO 손실 = 05_ppo_discrete.py 식", torch.isclose(E.ppo_loss(nl, ol, advt, vals, rets, ent), ref))

# R7 SAC: rsample로 정책까지 기울기, tanh 로그확률 = 직접 계산, Polyak
mu_net = nn.Linear(3, 2)
pi = lambda st: E.tanh_gaussian(mu_net(st)[:, :1], mu_net(st)[:, 1:], scale=2.0)
Qn = lambda: (lambda net: (lambda st, at: net(torch.cat([st, at], -1)).squeeze(-1)))(nn.Linear(4, 1))
q1, q2, q1t, q2t = Qn(), Qn(), Qn(), Qn()
st, at, rt, s2t, dt = torch.randn(6, 3), torch.rand(6, 1) * 4 - 2, torch.randn(6), torch.randn(6, 3), torch.zeros(6)
ql, pl, lp = E.sac_losses(pi, q1, q2, q1t, q2t, st, at, rt, s2t, dt, alpha=0.2)
pl.backward()
ok("R7 SAC 정책 손실 → 정책 가중치에 기울기 흐름(rsample)", mu_net.weight.grad is not None and mu_net.weight.grad.abs().sum() > 0)
torch.manual_seed(1); mu, ls = torch.zeros(1, 1), torch.zeros(1, 1)
act, lpp = E.tanh_gaussian(mu, ls, scale=2.0); torch.manual_seed(1); u = torch.randn(1, 1)
ref_lp = (-0.5 * u ** 2 - 0.5 * np.log(2 * np.pi)).sum() - torch.log(2 * (1 - torch.tanh(u) ** 2) + 1e-6).sum()
ok("R7 tanh 가우시안 log π = 직접 계산(N(0,1) 밀도 − log 보정)", torch.isclose(lpp.sum(), ref_lp) and torch.isclose(act, 2 * torch.tanh(u)).all())
la = torch.zeros(1, requires_grad=True); E.alpha_loss(la, torch.full((4,), 2.0), -1.0).backward()   # 엔트로피 ≈ −2 < 목표 −1
lb = torch.zeros(1, requires_grad=True); E.alpha_loss(lb, torch.full((4,), 0.0), -1.0).backward()   # 엔트로피 ≈ 0 > 목표 −1
ok("R7 α 손실: 엔트로피 < 목표면 α↑(기울기 음수), 엔트로피 > 목표면 α↓(기울기 양수)", la.grad.item() < 0 and lb.grad.item() > 0)
n1, n1t = nn.Linear(2, 1), nn.Linear(2, 1); before = n1t.weight.clone(); E.soft_update(n1, n1t, 0.005)
ok("R7 Polyak: θ′ = 0.995θ′ + 0.005θ", torch.allclose(n1t.weight, 0.995 * before + 0.005 * n1.weight))

# R8 가치 반복 = 10_value_iteration.py
fl = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
V, pi_ = E.value_iteration(fl.unwrapped.P, 16, 4)
vi = load("10_value_iteration.py"); V_ref, pi_ref, _ = vi.value_iteration()
ok(f"R8 가치 반복 = 10번 파일 (V 차이 {np.abs(V - V_ref).max():.1e}, 정책 같음)", np.abs(V - V_ref).max() < 1e-6 and list(pi_) == list(pi_ref))

# R9 UCB · 증분 평균
Qb, Nb = np.zeros(3), np.zeros(3); got = [[], [], []]
for t in range(1, 50):
    k = E.ucb_action(Qb, Nb, t); rv = rng.normal([0.1, 0.5, 0.9][k], 0.1); E.update_mean(Qb, Nb, k, rv); got[k].append(rv)
ok("R9 UCB가 안 당긴 팔부터 당기고, Q = 실제 표본 평균", all(len(g_) > 0 for g_ in got) and all(np.isclose(Qb[k], np.mean(got[k])) for k in range(3)))

# R10 잠재 기반 성형
ok("R10 성형 보상: 목표 쪽 한 칸 +, 끝 상태 Φ=0", np.isclose(E.shaped_reward(0, -0.3, -0.2, False, 1.0), 0.1) and np.isclose(E.shaped_reward(1, -0.1, -5, True, 1.0), 1.1))

# R11 RRT: 09_rrt.py의 지도·충돌 검사로 경로 찾기
rr = load("09_rrt.py")
path = E.rrt(rr.START, rr.GOAL, rr.edge_free, [0, 0], [10, 10])
ok(f"R11 RRT가 09번 지도에서 경로를 찾고 모든 선분이 충돌 없음 (길이 {rr.path_length(np.array(path)):.2f})",
   path is not None and all(rr.edge_free(path[i], path[i + 1]) for i in range(len(path) - 1)))

# R12 A* = 12_astar.py
ast = load("12_astar.py"); S_, G_ = ast.find("S"), ast.find("G")
pa = E.astar(S_, G_, ast.neighbors, lambda v: ast.manhattan(v, G_))
ok(f"R12 A* 경로 {len(pa) - 1}칸 = 12번 파일 {len(ast.astar(S_, G_, ast.manhattan)[0]) - 1}칸", len(pa) == len(ast.astar(S_, G_, ast.manhattan)[0]))
