"""07. SAC (Soft Actor-Critic) — Pendulum-v1 (연속 행동)
요즘 표준 구현(후속 논문 1812.05905 방식): Q 두 개 + 타깃 Q(Polyak) + 온도 α 자동 조절.
원 논문(1801.01290)과의 차이: 원 논문은 V 네트워크와 타깃 V를 따로 두고, α 대신 보상 크기(reward scale)를 조절.
실행: python 07_sac_pendulum.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Normal

TOTAL, LEARN_START, BATCH, BUFFER = 15_000, 1_000, 256, 100_000
GAMMA, TAU, LR = 0.99, 0.005, 3e-4                  # 원 논문 표 1과 같은 값
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

env = gym.make("Pendulum-v1")
obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
act_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2, dtype=torch.float32)  # 2
act_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2, dtype=torch.float32)   # 0


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    """가우시안 정책 + tanh로 범위 제한"""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, s):
        h = self.body(s)
        mu, log_std = self.mu(h), self.log_std(h).clamp(-20, 2)   # 표준편차가 너무 작거나 크지 않게
        dist = Normal(mu, log_std.exp())
        u = dist.rsample()                         # 재매개변수화: u = μ + σ·ε  (기울기가 μ, σ로 흐름)
        y = torch.tanh(u)
        action = y * act_scale + act_bias
        # tanh 변환에 따른 확률 보정: log π(a|s) = log N(u) − Σ log(scale·(1 − tanh²u))
        log_prob = dist.log_prob(u).sum(-1) - torch.log(act_scale * (1 - y.pow(2)) + 1e-6).sum(-1)
        mean_action = torch.tanh(mu) * act_scale + act_bias          # 평가용(탐험 없음)
        return action, log_prob, mean_action


actor = Actor()
q1, q2 = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)
q1_t, q2_t = mlp(obs_dim + act_dim, 1), mlp(obs_dim + act_dim, 1)
q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
opt_actor = torch.optim.Adam(actor.parameters(), lr=LR)
opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LR)
log_alpha = torch.zeros(1, requires_grad=True)       # α = exp(log_alpha) > 0 이 되도록
opt_alpha = torch.optim.Adam([log_alpha], lr=LR)
target_entropy = -float(act_dim)                     # 목표 엔트로피 = −(행동 차원 수)

# 리플레이 버퍼 (numpy 배열)
S = np.zeros((BUFFER, obs_dim), np.float32); A = np.zeros((BUFFER, act_dim), np.float32)
R = np.zeros(BUFFER, np.float32); S2 = np.zeros((BUFFER, obs_dim), np.float32); T = np.zeros(BUFFER, np.float32)
ptr, size = 0, 0

obs, _ = env.reset(seed=SEED)
ep_ret, log = 0.0, []
for step in range(1, TOTAL + 1):
    if step < LEARN_START:
        action = np.random.uniform(env.action_space.low, env.action_space.high)   # 처음엔 무작위로 버퍼 채우기
    else:
        with torch.no_grad():
            action = actor(torch.as_tensor(obs).unsqueeze(0))[0][0].numpy()
    next_obs, r, terminated, truncated, _ = env.step(action)
    S[ptr], A[ptr], R[ptr], S2[ptr], T[ptr] = obs, action, r, next_obs, float(terminated)
    ptr, size = (ptr + 1) % BUFFER, min(size + 1, BUFFER)
    obs, ep_ret = next_obs, ep_ret + r
    if terminated or truncated:
        log.append((step, ep_ret))
        if len(log) % 10 == 0:
            print(f"step {step:6d} | 최근 10판 평균 {np.mean([x for _, x in log[-10:]]):8.1f} | α {log_alpha.exp().item():.3f}")
        obs, _ = env.reset()
        ep_ret = 0.0

    if step >= LEARN_START:
        idx = np.random.randint(0, size, BATCH)
        s, a, r_b, s2, d = (torch.as_tensor(x[idx]) for x in (S, A, R, S2, T))
        alpha = log_alpha.exp().detach()

        # ① Q 학습: y = r + γ(1−d)[ min(Q1', Q2')(s', a') − α log π(a'|s') ],  a' ~ π(·|s')
        with torch.no_grad():
            a2, logp2, _ = actor(s2)
            sa2 = torch.cat([s2, a2], 1)
            q_next = torch.min(q1_t(sa2), q2_t(sa2)).squeeze(1) - alpha * logp2
            y = r_b + GAMMA * (1 - d) * q_next
        sa = torch.cat([s, a], 1)
        q_loss = F.mse_loss(q1(sa).squeeze(1), y) + F.mse_loss(q2(sa).squeeze(1), y)
        opt_q.zero_grad(); q_loss.backward(); opt_q.step()

        # ② 정책 학습: 최소화  E[ α log π(a|s) − min(Q1, Q2)(s, a) ],  a는 rsample (재매개변수화)
        a_pi, logp, _ = actor(s)
        sa_pi = torch.cat([s, a_pi], 1)
        actor_loss = (alpha * logp - torch.min(q1(sa_pi), q2(sa_pi)).squeeze(1)).mean()
        opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()

        # ③ 온도 α 자동 조절: 엔트로피가 목표보다 낮으면 α↑ (더 탐험), 높으면 α↓
        alpha_loss = -(log_alpha.exp() * (logp.detach() + target_entropy)).mean()
        opt_alpha.zero_grad(); alpha_loss.backward(); opt_alpha.step()

        # ④ 타깃 Q를 천천히 따라가게 (Polyak 평균): θ' ← τθ + (1−τ)θ'
        with torch.no_grad():
            for net, tgt in ((q1, q1_t), (q2, q2_t)):
                for p_, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - TAU).add_(TAU * p_)

np.savetxt("results/sac.txt", np.array(log))

# 평가: 평균 행동으로 5판
scores = []
for i in range(5):
    o, _ = env.reset(seed=500 + i); done, tot = False, 0.0
    while not done:
        with torch.no_grad():
            a_ = actor(torch.as_tensor(o).unsqueeze(0))[2][0].numpy()
        o, r, te, tr, _ = env.step(a_); tot += r; done = te or tr
    scores.append(tot)
print(f"평가(평균 행동) 5판 평균: {np.mean(scores):.1f}  (Pendulum은 0에 가까울수록 좋음, 대략 −200 안팎이면 잘 세움)")
