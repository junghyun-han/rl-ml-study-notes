"""모의 실습 M2: 연속 행동 환경에서 SAC로 성능 올리기 (Pendulum, MuJoCo 등 환경 이름만 바꿔 사용)
실행 예: python mock_sac.py --env Pendulum-v1 --steps 15000 --seed 0
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Normal


def evaluate(policy_fn, env_id, episodes=10, seed=1000):
    """탐험 없는 정책으로 여러 판 평가 → (평균, 표준편차, 평균 길이). 학습과 다른 시드 사용"""
    env = gym.make(env_id)
    scores, lengths = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done, total, n = False, 0.0, 0
        while not done:
            obs, r, term, trunc, _ = env.step(policy_fn(obs))
            total += r; n += 1
            done = term or trunc
        scores.append(total); lengths.append(n)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(lengths))


p = argparse.ArgumentParser()
p.add_argument("--env", default="Pendulum-v1")
p.add_argument("--steps", type=int, default=15_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--save", default="")          # 판 리턴 기록 파일(곡선 그리기용)
args = p.parse_args()

GAMMA, TAU, LR, BATCH, START, BUF = 0.99, 0.005, 3e-4, 256, 1000, 100_000
torch.manual_seed(args.seed); np.random.seed(args.seed)
env = gym.make(args.env)
n_obs, n_act = env.observation_space.shape[0], env.action_space.shape[0]
low, high = env.action_space.low, env.action_space.high
scale = torch.tensor((high - low) / 2, dtype=torch.float32); bias = torch.tensor((high + low) / 2, dtype=torch.float32)
mlp = lambda i, o: nn.Sequential(nn.Linear(i, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, o))


class Actor(nn.Module):
    def __init__(self):
        super().__init__(); self.net = mlp(n_obs, 2 * n_act)

    def forward(self, s):
        mu, log_std = self.net(s).chunk(2, -1)
        dist = Normal(mu, log_std.clamp(-20, 2).exp())
        u = dist.rsample(); y = torch.tanh(u)
        logp = dist.log_prob(u).sum(-1) - torch.log(scale * (1 - y.pow(2)) + 1e-6).sum(-1)
        return y * scale + bias, logp, torch.tanh(mu) * scale + bias


actor = Actor()
q1, q2, q1t, q2t = (mlp(n_obs + n_act, 1) for _ in range(4))
q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
opt_pi = torch.optim.Adam(actor.parameters(), lr=LR)
opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=LR)
log_alpha = torch.zeros(1, requires_grad=True); opt_a = torch.optim.Adam([log_alpha], lr=LR); target_ent = -float(n_act)
S = np.zeros((BUF, n_obs), np.float32); A = np.zeros((BUF, n_act), np.float32)
Rw = np.zeros(BUF, np.float32); S2 = np.zeros((BUF, n_obs), np.float32); T = np.zeros(BUF, np.float32)
ptr = size = 0

obs, _ = env.reset(seed=args.seed); env.action_space.seed(args.seed)
ep_ret, returns = 0.0, []
for step in range(1, args.steps + 1):
    if step < START:
        act = env.action_space.sample()                                   # 처음엔 무작위로 버퍼 채우기
    else:
        with torch.no_grad():
            act = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0][0].numpy()
    nxt, r, term, trunc, _ = env.step(act)
    S[ptr], A[ptr], Rw[ptr], S2[ptr], T[ptr] = obs, act, r, nxt, float(term)   # terminated만 끝으로
    ptr, size = (ptr + 1) % BUF, min(size + 1, BUF)
    obs, ep_ret = nxt, ep_ret + r
    if term or trunc:
        returns.append(ep_ret); ep_ret = 0.0; obs, _ = env.reset()
    if step >= START:
        i = np.random.randint(0, size, BATCH)
        s, a, rb, s2, d = (torch.as_tensor(x[i]) for x in (S, A, Rw, S2, T))
        alpha = log_alpha.exp().detach()
        with torch.no_grad():                                                # ① Q 목표
            a2, lp2, _ = actor(s2); sa2 = torch.cat([s2, a2], 1)
            y = rb + GAMMA * (1 - d) * (torch.min(q1t(sa2), q2t(sa2)).squeeze(1) - alpha * lp2)
        sa = torch.cat([s, a], 1)
        lq = F.mse_loss(q1(sa).squeeze(1), y) + F.mse_loss(q2(sa).squeeze(1), y)
        opt_q.zero_grad(); lq.backward(); opt_q.step()
        an, lp, _ = actor(s); san = torch.cat([s, an], 1)                     # ② 정책
        lpi = (alpha * lp - torch.min(q1(san), q2(san)).squeeze(1)).mean()
        opt_pi.zero_grad(); lpi.backward(); opt_pi.step()
        la = -(log_alpha.exp() * (lp.detach() + target_ent)).mean()           # ③ 온도 α
        opt_a.zero_grad(); la.backward(); opt_a.step()
        with torch.no_grad():                                                # ④ Polyak
            for net, tgt in ((q1, q1t), (q2, q2t)):
                for p_, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - TAU).add_(TAU * p_)
    if step % 3000 == 0:
        print(f"step {step:6d} | 최근 5판 학습 리턴 {np.mean(returns[-5:]):8.1f} | α {log_alpha.exp().item():.3f}")

# ⑤ 탐험 없이 평가 (평균 행동)
pol = lambda o: actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))[2][0].detach().numpy()
mean, std, length = evaluate(pol, args.env, episodes=10)
print(f"FINAL seed {args.seed}: 평가 10판 {mean:.1f} ± {std:.1f} (평균 길이 {length:.0f})")
if args.save:
    np.savetxt(args.save, np.array(returns))
