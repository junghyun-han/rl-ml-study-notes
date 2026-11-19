"""06. PPO 연속 행동 버전 — Pendulum-v1
05_ppo_discrete.py와 달라지는 곳만 바꿈: Categorical → Normal(가우시안), log_prob·entropy를 행동 차원으로 합산, 행동 범위 자르기
실행: python 06_ppo_continuous.py
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Normal

TOTAL, N_STEPS, EPOCHS, MINIBATCH = 200_000, 2048, 10, 64
GAMMA, LAM, CLIP, LR, ENT_COEF, VF_COEF, MAX_GRAD = 0.99, 0.95, 0.2, 3e-4, 0.0, 0.5, 0.5
SEED = 0
torch.manual_seed(SEED)

env = gym.make("Pendulum-v1")
env = gym.wrappers.NormalizeReward(env, gamma=GAMMA)    # 보상 크기 정규화 (학습 안정)
obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
low, high = env.action_space.low, env.action_space.high  # [-2, 2]


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, act_dim))
        self.log_std = nn.Parameter(torch.zeros(act_dim))     # 상태와 무관한 학습 가능한 표준편차
        self.critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None):
        dist = Normal(self.mu(x), self.log_std.exp())
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1)   # 차원별 값을 합산


model = ActorCritic()
opt = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
b_obs, b_act = torch.zeros(N_STEPS, obs_dim), torch.zeros(N_STEPS, act_dim)
b_logp, b_rew, b_done, b_val = (torch.zeros(N_STEPS) for _ in range(4))

obs, _ = env.reset(seed=SEED)
log, step = [], 0
for update in range(TOTAL // N_STEPS):
    # ① 롤아웃 모으기
    for t in range(N_STEPS):
        o = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            a, logp, _ = model.act(o)
            v = model.value(o)
        a_env = np.clip(a.numpy(), low, high)              # 환경에는 범위 안으로 자른 값을 보냄
        next_obs, r, terminated, truncated, info = env.step(a_env)
        step += 1
        if truncated and not terminated:
            with torch.no_grad():
                r += GAMMA * model.value(torch.as_tensor(next_obs, dtype=torch.float32)).item()
        b_obs[t], b_act[t], b_logp[t], b_val[t] = o, a, logp, v   # 버퍼에는 자르기 전 a (log_prob 일치)
        b_rew[t], b_done[t] = r, float(terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    # ② GAE (05와 같음)
    with torch.no_grad():
        last_v = model.value(torch.as_tensor(obs, dtype=torch.float32))
    adv, gae = torch.zeros(N_STEPS), 0.0
    for t in reversed(range(N_STEPS)):
        next_v = last_v if t == N_STEPS - 1 else b_val[t + 1]
        nonterm = 1.0 - b_done[t]
        delta = b_rew[t] + GAMMA * next_v * nonterm - b_val[t]
        gae = delta + GAMMA * LAM * nonterm * gae
        adv[t] = gae
    ret = adv + b_val

    # ③ PPO 업데이트 (05와 같음)
    for epoch in range(EPOCHS):
        perm = torch.randperm(N_STEPS)
        for start in range(0, N_STEPS, MINIBATCH):
            idx = perm[start:start + MINIBATCH]
            _, new_logp, entropy = model.act(b_obs[idx], b_act[idx])
            ratio = (new_logp - b_logp[idx]).exp()
            mb_adv = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
            policy_loss = -torch.min(ratio * mb_adv, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * mb_adv).mean()
            value_loss = ((model.value(b_obs[idx]) - ret[idx]) ** 2).mean()
            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy.mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD); opt.step()

    # 평가: 정규화 안 된 원래 보상으로, 평균 행동(탐험 없음) 3판
    ev_env, scores = gym.make("Pendulum-v1"), []
    for i in range(3):
        o2, _ = ev_env.reset(seed=100 + i); done, tot = False, 0.0
        while not done:
            with torch.no_grad():
                a_ = model.mu(torch.as_tensor(o2, dtype=torch.float32)).numpy()
            o2, r_, te, tr, _ = ev_env.step(np.clip(a_, low, high)); tot += r_; done = te or tr
        scores.append(tot)
    log.append((step, np.mean(scores)))
    if (update + 1) % 10 == 0:
        print(f"update {update + 1:3d} | step {step:6d} | 평가 평균 {np.mean(scores):8.1f} | std {model.log_std.exp().mean().item():.3f}")

np.savetxt("results/ppo_continuous.txt", np.array(log))
