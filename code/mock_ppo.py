"""모의 실습 M1: 이산 행동 환경에서 PPO로 성능 올리기 (시험장에서 그대로 칠 수 있는 한 파일 풀이)
실행 예: python mock_ppo.py --env CartPole-v1 --steps 60000 --seed 0
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical
from exam_workflow import evaluate

p = argparse.ArgumentParser()
p.add_argument("--env", default="CartPole-v1")
p.add_argument("--steps", type=int, default=60_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--ent", type=float, default=0.01)
args = p.parse_args()

N_STEPS, EPOCHS, MB, GAMMA, LAM, CLIP, VF, MAX_GRAD = 2048, 10, 64, 0.99, 0.95, 0.2, 0.5, 0.5
torch.manual_seed(args.seed); np.random.seed(args.seed)
env = gym.make(args.env)
n_obs, n_act = env.observation_space.shape[0], env.action_space.n


def layer(i, o, std=np.sqrt(2)):
    l = nn.Linear(i, o); nn.init.orthogonal_(l.weight, std); nn.init.zeros_(l.bias); return l


actor = nn.Sequential(layer(n_obs, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, n_act, 0.01))
critic = nn.Sequential(layer(n_obs, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, 1, 1.0))
opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr, eps=1e-5)

obs, _ = env.reset(seed=args.seed)
ep_ret, returns, step = 0.0, [], 0
for update in range(args.steps // N_STEPS):
    # ① 지금 정책으로 N_STEPS 모으기
    O, A, LP, R, D, V = [], [], [], [], [], []
    for _ in range(N_STEPS):
        o = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = Categorical(logits=actor(o)); a = dist.sample(); v = critic(o).squeeze()
        nxt, r, term, trunc, _ = env.step(a.item()); step += 1; ep_ret += r
        if trunc and not term:                       # 시간 잘림 → 미래 가치를 보상에 더해 부트스트랩
            with torch.no_grad():
                r += GAMMA * critic(torch.as_tensor(nxt, dtype=torch.float32)).item()
        O.append(o); A.append(a); LP.append(dist.log_prob(a)); R.append(r); D.append(float(term or trunc)); V.append(v)
        obs = nxt
        if term or trunc:
            returns.append(ep_ret); ep_ret = 0.0; obs, _ = env.reset()
    O, A, LP, V = torch.stack(O), torch.stack(A), torch.stack(LP), torch.stack(V)
    R, D = torch.tensor(R), torch.tensor(D)
    # ② GAE (뒤에서부터)
    with torch.no_grad():
        last_v = critic(torch.as_tensor(obs, dtype=torch.float32)).squeeze()
    adv, g = torch.zeros(N_STEPS), 0.0
    for t in reversed(range(N_STEPS)):
        nv = last_v if t == N_STEPS - 1 else V[t + 1]
        delta = R[t] + GAMMA * nv * (1 - D[t]) - V[t]
        g = delta + GAMMA * LAM * (1 - D[t]) * g
        adv[t] = g
    ret = adv + V
    # ③ 같은 데이터로 EPOCHS번, 미니배치 업데이트
    for _ in range(EPOCHS):
        for idx in torch.randperm(N_STEPS).split(MB):
            dist = Categorical(logits=actor(O[idx]))
            ratio = (dist.log_prob(A[idx]) - LP[idx]).exp()
            a_n = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
            pg = -torch.min(ratio * a_n, ratio.clamp(1 - CLIP, 1 + CLIP) * a_n).mean()
            vl = ((critic(O[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
            loss = pg + VF * vl - args.ent * dist.entropy().mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), MAX_GRAD); opt.step()
    if (update + 1) % 5 == 0:
        print(f"step {step:6d} | 최근 20판 학습 리턴 {np.mean(returns[-20:]):6.1f}")

# ④ 탐험 없이 평가 (argmax 행동, 학습과 다른 시드)
mean, std, length = evaluate(lambda o: int(actor(torch.as_tensor(o, dtype=torch.float32)).argmax()), args.env, episodes=10)
print(f"FINAL seed {args.seed}: 평가 10판 {mean:.1f} ± {std:.1f} (평균 길이 {length:.0f})")
