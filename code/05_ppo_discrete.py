"""05. PPO (clip) + GAE — CartPole-v1 (이산 행동)
순서: ① 정책으로 N스텝 모으기 → ② GAE로 이점 계산 → ③ 같은 데이터로 K epoch 미니배치 업데이트(clip)
실행: python 05_ppo_discrete.py            (학습률 바꾸기: --lr 3e-3)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=100_000)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--clip", type=float, default=0.2)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

N_STEPS, EPOCHS, MINIBATCH = 2048, 10, 64         # PPO 논문 MuJoCo 표와 같은 값
GAMMA, LAM, CLIP = 0.99, 0.95, args.clip
ENT_COEF, VF_COEF, MAX_GRAD = 0.01, 0.5, 0.5

torch.manual_seed(args.seed); np.random.seed(args.seed)
env = gym.make("CartPole-v1")
obs_dim, n_act = env.observation_space.shape[0], env.action_space.n


def layer(i, o, std=np.sqrt(2)):
    """직교 초기화 (PPO 구현에서 흔히 쓰는 방식). 정책 마지막 층은 0.01로 작게 → 처음엔 거의 균등 확률"""
    l = nn.Linear(i, o)
    nn.init.orthogonal_(l.weight, std)
    nn.init.zeros_(l.bias)
    return l


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(layer(obs_dim, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, n_act, 0.01))
        self.critic = nn.Sequential(layer(obs_dim, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, 1, 1.0))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def act(self, x, action=None):
        dist = Categorical(logits=self.actor(x))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


model = ActorCritic()
opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

# 롤아웃 저장 공간
b_obs = torch.zeros(N_STEPS, obs_dim)
b_act = torch.zeros(N_STEPS, dtype=torch.long)
b_logp, b_rew, b_done, b_val = (torch.zeros(N_STEPS) for _ in range(4))

obs, _ = env.reset(seed=args.seed)
ep_ret, log, step = 0.0, [], 0
n_updates = args.steps // N_STEPS
for update in range(n_updates):
    # ① 현재 정책(π_old)으로 N_STEPS 모으기
    for t in range(N_STEPS):
        o = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            a, logp, _ = model.act(o)
            v = model.value(o)
        next_obs, r, terminated, truncated, _ = env.step(a.item())
        step += 1
        ep_ret += r
        if truncated and not terminated:
            # 시간 제한으로 잘린 것은 "끝"이 아님 → 다음 상태 가치를 보상에 더해 부트스트랩
            with torch.no_grad():
                r += GAMMA * model.value(torch.as_tensor(next_obs, dtype=torch.float32)).item()
        b_obs[t], b_act[t], b_logp[t], b_val[t] = o, a, logp, v
        b_rew[t], b_done[t] = r, float(terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            log.append((step, ep_ret))
            ep_ret = 0.0
            obs, _ = env.reset()

    # ② GAE(λ): δ_t = r_t + γV(s_(t+1)) − V(s_t),  Â_t = δ_t + γλ Â_(t+1)
    with torch.no_grad():
        last_v = model.value(torch.as_tensor(obs, dtype=torch.float32))
    adv = torch.zeros(N_STEPS)
    gae = 0.0
    for t in reversed(range(N_STEPS)):
        next_v = last_v if t == N_STEPS - 1 else b_val[t + 1]
        nonterm = 1.0 - b_done[t]                      # 판이 끝났으면 다음 판 값을 끌어오지 않음
        delta = b_rew[t] + GAMMA * next_v * nonterm - b_val[t]
        gae = delta + GAMMA * LAM * nonterm * gae
        adv[t] = gae
    ret = adv + b_val                                  # Critic 학습 목표 (= λ-return)

    # ③ 같은 데이터로 EPOCHS번, 미니배치로 나눠 업데이트
    kls, clipfracs = [], []
    for epoch in range(EPOCHS):
        perm = torch.randperm(N_STEPS)
        for start in range(0, N_STEPS, MINIBATCH):
            idx = perm[start:start + MINIBATCH]
            _, new_logp, entropy = model.act(b_obs[idx], b_act[idx])
            log_ratio = new_logp - b_logp[idx]
            ratio = log_ratio.exp()                                    # r_t(θ) = π_θ / π_old
            mb_adv = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)   # 이점 정규화

            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()               # L^CLIP (최대화 → 음수)
            value_loss = ((model.value(b_obs[idx]) - ret[idx]) ** 2).mean()
            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy.mean()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD)
            opt.step()

            with torch.no_grad():                                       # 진단용 지표
                kls.append(((ratio - 1) - log_ratio).mean().item())     # 근사 KL
                clipfracs.append(((ratio - 1).abs() > CLIP).float().mean().item())

    ev = 1 - torch.var(ret - b_val) / (torch.var(ret) + 1e-8)          # 설명 분산 (1에 가까울수록 Critic 좋음)
    recent = np.mean([r for _, r in log[-20:]]) if log else 0
    print(f"update {update + 1:3d} | step {step:6d} | 최근 20판 {recent:6.1f} | "
          f"KL {np.mean(kls):.4f} | clipfrac {np.mean(clipfracs):.3f} | entropy {entropy.mean().item():.3f} | EV {ev:.2f}")

np.savetxt(f"results/ppo_lr{args.lr:g}_clip{args.clip:g}_s{args.seed}.txt", np.array(log))
