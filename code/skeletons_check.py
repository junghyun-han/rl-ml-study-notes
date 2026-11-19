"""3교시 노트 16장 '빈 파일 뼈대' 코드가 실제로 돌아가는지 확인"""
import random, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, gymnasium as gym
from torch.distributions import Normal

random.seed(0); np.random.seed(0); torch.manual_seed(0)
# ── 16-1 DQN 뼈대 (노트와 같은 코드) ──
env = gym.make("CartPole-v1"); nS, nA = env.observation_space.shape[0], env.action_space.n
mk = lambda: nn.Sequential(nn.Linear(nS, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, nA))
q, qt = mk(), mk(); qt.load_state_dict(q.state_dict())
opt = torch.optim.Adam(q.parameters(), lr=1e-3); buf = collections.deque(maxlen=50000)
s, _ = env.reset(seed=0)
for t in range(50000):
    eps = max(0.05, 1 - t / 15000)
    a = random.randrange(nA) if random.random() < eps else q(torch.as_tensor(s)).argmax().item()
    s2, r, term, trunc, _ = env.step(a); buf.append((s, a, r, s2, float(term)))
    s = env.reset()[0] if (term or trunc) else s2
    if len(buf) >= 1000:
        S, A, R, S2, D = map(lambda x: torch.as_tensor(np.array(x), dtype=torch.float32), zip(*random.sample(buf, 128)))
        qsa = q(S).gather(1, A.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            y = R + 0.99 * (1 - D) * qt(S2).gather(1, q(S2).argmax(1, keepdim=True)).squeeze(1)  # Double
        loss = F.smooth_l1_loss(qsa, y); opt.zero_grad(); loss.backward(); opt.step()
    if t % 500 == 0: qt.load_state_dict(q.state_dict())
scores = []
for i in range(10):
    o, _ = env.reset(seed=100 + i); done, tot = False, 0
    while not done:
        with torch.no_grad(): o, r, te, tr, _ = env.step(q(torch.as_tensor(o)).argmax().item())
        tot += r; done = te or tr
    scores.append(tot)
print("16-1 DQN 뼈대 평가 10판 평균:", np.mean(scores))

# ── 16-2 GAE + clip 손실 (임의 데이터로 모양·값 확인) ──
T, gamma, lam, clip = 6, 0.99, 0.95, 0.2
rew = torch.ones(T); done = torch.tensor([0, 0, 1, 0, 0, 0.]); val = torch.rand(T); last_v = torch.tensor(0.5)
adv, gae = torch.zeros(T), 0.0
for t in reversed(range(T)):
    nv = last_v if t == T - 1 else val[t + 1]
    delta = rew[t] + gamma * nv * (1 - done[t]) - val[t]
    gae = delta + gamma * lam * (1 - done[t]) * gae
    adv[t] = gae
ret = adv + val
assert torch.isclose(adv[2], rew[2] - val[2])               # 끝난 스텝: 다음 값 끌어오지 않음
V = nn.Linear(3, 1); obs_mb = torch.randn(T, 3)
new_logp, old_logp, entropy = torch.randn(T, requires_grad=True), torch.randn(T), torch.rand(T)
adv_mb, ret_mb = adv, ret
ratio = (new_logp - old_logp).exp()
a_n = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
pg = -torch.min(ratio * a_n, ratio.clamp(1 - clip, 1 + clip) * a_n).mean()
vl = ((V(obs_mb).squeeze(-1) - ret_mb) ** 2).mean()
loss = pg + 0.5 * vl - 0.01 * entropy.mean(); loss.backward()
print("16-2 GAE·clip 손실 OK, loss =", round(loss.item(), 4))

# ── 16-3, 16-4 SAC 네 줄 + tanh log_prob ──
nS, nA, B = 3, 1, 8
scale, bias = torch.tensor([2.0]), torch.tensor([0.0])
body = nn.Linear(nS, 2 * nA)
def pi(s):
    mu, log_std = body(s).chunk(2, -1)
    dist = Normal(mu, log_std.clamp(-20, 2).exp())
    u = dist.rsample(); y_ = torch.tanh(u); a = y_ * scale + bias
    logp = dist.log_prob(u).sum(-1) - torch.log(scale * (1 - y_.pow(2)) + 1e-6).sum(-1)
    return a, logp
Qnet = lambda: (lambda net: (lambda s, a: net(torch.cat([s, a], -1)).squeeze(-1)))(nn.Linear(nS + nA, 1))
q1, q2, q1t, q2t = Qnet(), Qnet(), Qnet(), Qnet()
s, a, r, s2, d = torch.randn(B, nS), torch.rand(B, nA) * 4 - 2, torch.randn(B), torch.randn(B, nS), torch.zeros(B)
g, alpha, tau, target_ent = 0.99, 0.2, 0.005, -1.0
log_alpha = torch.zeros(1, requires_grad=True)
with torch.no_grad():
    a2, lp2 = pi(s2); y = r + g * (1 - d) * (torch.min(q1t(s2, a2), q2t(s2, a2)) - alpha * lp2)
q_loss = F.mse_loss(q1(s, a), y) + F.mse_loss(q2(s, a), y)
a_pi, lp = pi(s)
pi_loss = (alpha * lp - torch.min(q1(s, a_pi), q2(s, a_pi))).mean()
alpha_loss = -(log_alpha.exp() * (lp.detach() + target_ent)).mean()
q_loss.backward(); pi_loss.backward(); alpha_loss.backward()
assert body.weight.grad is not None and body.weight.grad.abs().sum() > 0   # rsample → 정책까지 기울기
n1, n1t = nn.Linear(2, 1), nn.Linear(2, 1); before = n1t.weight.clone()
for p, pt in zip(n1.parameters(), n1t.parameters()): pt.data.mul_(1 - tau).add_(tau * p.data)
assert torch.allclose(n1t.weight, (1 - tau) * before + tau * n1.weight)
print("16-3·16-4 SAC 네 줄 + tanh log_prob OK (정책 기울기 흐름 확인)")

# ── 16-5 RRT 핵심 ──
rng = np.random.default_rng(0); goal, lo, hi, step = np.array([9., 9.]), np.array([0., 0.]), np.array([10., 10.]), 0.5
nodes, parent = [np.array([1., 1.])], [-1]
edge_free = lambda p, q_: True
for _ in range(200):
    x_rand = goal if rng.random() < 0.1 else rng.uniform(lo, hi)
    i = np.argmin(np.linalg.norm(np.array(nodes) - x_rand, axis=1)); x_near = nodes[i]
    d_ = x_rand - x_near; dist = np.linalg.norm(d_)
    if dist > 1e-9:                                   # 같은 점이면 0으로 나누기 방지
        x_new = x_near + d_ / dist * min(step, dist)
        if edge_free(x_near, x_new): nodes.append(x_new); parent.append(i)
assert not np.isnan(np.array(nodes)).any()
print("16-5 RRT 핵심 OK, 노드", len(nodes), "개, 목표까지 최소 거리", round(min(np.linalg.norm(np.array(nodes) - goal, axis=1)), 3))

# ── 16-6 가치 반복 뼈대 (10_value_iteration.py와 같은 답인지) ──
import gymnasium as gym, heapq, importlib.util
fl = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
P, nS, nA, g = fl.unwrapped.P, 16, 4, 0.99
V = np.zeros(nS)
while True:
    delta = 0.0
    for s in range(nS):
        q = [sum(p * (r + g * V[s2] * (not d)) for p, s2, r, d in P[s][a]) for a in range(nA)]
        delta = max(delta, abs(max(q) - V[s])); V[s] = max(q)
    if delta < 1e-8: break
pi = [int(np.argmax([sum(p * (r + g * V[s2] * (not d)) for p, s2, r, d in P[s][a]) for a in range(nA)])) for s in range(nS)]
spec = importlib.util.spec_from_file_location("vi", "10_value_iteration.py"); vi = importlib.util.module_from_spec(spec); spec.loader.exec_module(vi)
V_ref, pi_ref, _ = vi.value_iteration()
print("16-6 가치 반복 뼈대: V 차이", f"{np.abs(V - V_ref).max():.1e}", "| 정책 같음", list(pi) == list(pi_ref))

# ── 16-7 ε-greedy · UCB · 증분 평균 (한 문제에서 동작 확인) ──
rng = np.random.default_rng(0); K, c = 10, 2.0; q_true = rng.normal(0, 1, K)
for name in ["eps", "ucb"]:
    Q, N, eps, hits, got = np.zeros(K), np.zeros(K), 0.1, 0, [[] for _ in range(K)]
    for t in range(1, 2001):
        if name == "eps":
            a = int(rng.integers(K)) if rng.random() < eps else int(np.argmax(Q))
        else:
            a = int(np.argmin(N)) if (N == 0).any() else int(np.argmax(Q + c * np.sqrt(np.log(t) / N)))
        r = rng.normal(q_true[a], 1)
        N[a] += 1; Q[a] += (r - Q[a]) / N[a]
        got[a].append(r)
        hits += (t > 1000) and (a == int(np.argmax(q_true)))
    same = all(np.isclose(Q[k], np.mean(got[k])) for k in range(K) if got[k])
    print(f"16-7 {name}: 뒤 1000번 중 최적 팔 {hits / 10:.1f}%, 증분 평균 Q = 실제 표본 평균: {same}")

# ── 16-8 A* 뼈대 (12_astar.py와 같은 길이인지) ──
spec = importlib.util.spec_from_file_location("ast", "12_astar.py"); ast_ = importlib.util.module_from_spec(spec); spec.loader.exec_module(ast_)
S, G = ast_.find("S"), ast_.find("G"); neighbors = ast_.neighbors; h = lambda v: ast_.manhattan(v, G)
heap, gd, parent, closed = [(h(S), 0, S)], {S: 0}, {S: None}, set()
while heap:
    f, gc, u = heapq.heappop(heap)
    if u in closed: continue
    closed.add(u)
    if u == G: break
    for v in neighbors(u):
        if gc + 1 < gd.get(v, float("inf")):
            gd[v], parent[v] = gc + 1, u
            heapq.heappush(heap, (gc + 1 + h(v), gc + 1, v))
ref_path, _ = ast_.astar(S, G, ast_.manhattan)
print("16-8 A* 뼈대: 경로 길이", gd[G], "| 원본", len(ref_path) - 1)
