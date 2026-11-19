"""03. REINFORCE (몬테카를로 정책 경사) — CartPole-v1
한 판을 끝까지 해 본 뒤, 잘 된 행동의 확률은 올리고 못 된 행동의 확률은 내린다.
실행: python 03_reinforce_cartpole.py
"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical

GAMMA, LR, EPISODES, SEED = 0.99, 1e-3, 800, 0
torch.manual_seed(SEED)
env = gym.make("CartPole-v1")
policy = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))   # 출력 = 행동별 logit
opt = torch.optim.Adam(policy.parameters(), lr=LR)

returns_log = []
for ep in range(EPISODES):
    obs, _ = env.reset(seed=SEED + ep)
    log_probs, rewards = [], []
    while True:
        dist = Categorical(logits=policy(torch.as_tensor(obs)))   # logit → 확률 분포
        a = dist.sample()                                         # 확률대로 행동 뽑기 (탐험이 저절로 됨)
        log_probs.append(dist.log_prob(a))                        # log π(a|s) 저장 (기울기 연결됨)
        obs, r, terminated, truncated, _ = env.step(a.item())
        rewards.append(r)
        if terminated or truncated:
            break

    # 뒤에서부터 할인 누적 보상 G_t = r_t + γ G_(t+1)
    G, Gs = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        Gs.insert(0, G)
    Gs = torch.tensor(Gs)
    Gs = (Gs - Gs.mean()) / (Gs.std() + 1e-8)      # 정규화 = 기준선(baseline) 역할 → 분산 감소

    loss = -(torch.stack(log_probs) * Gs).sum()     # 경사 "상승"을 위해 음수를 붙여 최소화
    opt.zero_grad()
    loss.backward()
    opt.step()

    returns_log.append(sum(rewards))
    if (ep + 1) % 100 == 0:
        print(f"episode {ep + 1:4d} | 최근 50판 평균 {np.mean(returns_log[-50:]):6.1f}")

np.savetxt("results/reinforce.txt", np.array(returns_log))
