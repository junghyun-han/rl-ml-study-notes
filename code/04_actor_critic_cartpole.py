"""04. Actor-Critic (1-step TD 이점) — CartPole-v1
Actor = 정책 π(a|s), Critic = 상태 가치 V(s).  이점 A ≈ δ = r + γV(s') − V(s)
실행: python 04_actor_critic_cartpole.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical

GAMMA, LR_A, LR_C, EPISODES, SEED = 0.99, 1e-3, 5e-3, 800, 0
torch.manual_seed(SEED)
env = gym.make("CartPole-v1")
actor = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))
critic = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 1))
opt_a = torch.optim.Adam(actor.parameters(), lr=LR_A)
opt_c = torch.optim.Adam(critic.parameters(), lr=LR_C)

returns_log = []
for ep in range(EPISODES):
    obs, _ = env.reset(seed=SEED + ep)
    S, A, R, S2, T = [], [], [], [], []
    while True:
        with torch.no_grad():
            a = Categorical(logits=actor(torch.as_tensor(obs))).sample().item()
        obs2, r, terminated, truncated, _ = env.step(a)
        S.append(obs); A.append(a); R.append(r); S2.append(obs2); T.append(float(terminated))
        obs = obs2
        if terminated or truncated:
            break

    S, S2 = torch.as_tensor(np.array(S)), torch.as_tensor(np.array(S2))
    A, R, T = torch.as_tensor(A), torch.as_tensor(R, dtype=torch.float32), torch.as_tensor(T)

    v = critic(S).squeeze(1)                                   # V(s)
    with torch.no_grad():
        target = R + GAMMA * critic(S2).squeeze(1) * (1 - T)   # TD 타깃 (끝난 상태면 미래 0)
    critic_loss = F.mse_loss(v, target)                        # Critic: V(s)를 타깃에 맞춤
    advantage = (target - v).detach()                          # δ = 이점 추정 (Actor 쪽엔 상수로)
    log_prob = Categorical(logits=actor(S)).log_prob(A)
    actor_loss = -(log_prob * advantage).mean()                # 이점이 +면 그 행동 확률↑

    opt_c.zero_grad(); critic_loss.backward(); opt_c.step()
    opt_a.zero_grad(); actor_loss.backward(); opt_a.step()

    returns_log.append(R.sum().item())
    if (ep + 1) % 100 == 0:
        print(f"episode {ep + 1:4d} | 최근 50판 평균 {np.mean(returns_log[-50:]):6.1f}")

np.savetxt("results/actor_critic.txt", np.array(returns_log))
