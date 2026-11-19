"""02. DQN (+ Double DQN) — CartPole-v1
핵심 3가지: ① Q 신경망  ② 경험 재생 버퍼  ③ 타깃 네트워크
실행: python 02_dqn_cartpole.py            (Double DQN 끄기: --no-double, 타깃망 끄기: --no-target)
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=50_000)
p.add_argument("--no-double", action="store_true")
p.add_argument("--no-target", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--target-every", type=int, default=500)
args = p.parse_args()

# ── 하이퍼파라미터 ──
GAMMA, LR, BATCH = 0.99, args.lr, 128
BUFFER, LEARN_START, TARGET_EVERY = 50_000, 1_000, args.target_every
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 15_000

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
env = gym.make("CartPole-v1")
obs_dim, n_act = env.observation_space.shape[0], env.action_space.n


class QNet(nn.Module):
    """상태 → 행동마다 Q값 하나씩 (출력 개수 = 행동 개수)"""
    def __init__(self, obs_dim, n_act, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_act))

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """(s, a, r, s', done)을 원형 배열에 저장하고 무작위로 꺼냄"""
    def __init__(self, cap, obs_dim):
        self.s = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.s2 = np.zeros((cap, obs_dim), np.float32)
        self.d = np.zeros(cap, np.float32)
        self.cap, self.i, self.size = cap, 0, 0

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.cap               # 다 차면 가장 오래된 것부터 덮어씀
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (torch.as_tensor(self.s[idx]), torch.as_tensor(self.a[idx]), torch.as_tensor(self.r[idx]),
                torch.as_tensor(self.s2[idx]), torch.as_tensor(self.d[idx]))


q = QNet(obs_dim, n_act)
q_target = QNet(obs_dim, n_act)
q_target.load_state_dict(q.state_dict())                # 처음엔 똑같이 복사
opt = torch.optim.Adam(q.parameters(), lr=LR)
buf = ReplayBuffer(BUFFER, obs_dim)

obs, _ = env.reset(seed=args.seed)
ep_ret, ep_returns, log = 0.0, [], []
for step in range(1, args.steps + 1):
    # ① ε-greedy 행동 선택 (ε는 선형으로 줄임)
    eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY_STEPS)
    if random.random() < eps:
        action = random.randrange(n_act)            # 시드 고정된 난수로 (재현 가능)
    else:
        with torch.no_grad():
            action = int(q(torch.as_tensor(obs)).argmax())

    # ② 환경 한 스텝 + 버퍼 저장
    next_obs, r, terminated, truncated, _ = env.step(action)
    buf.add(obs, action, r, next_obs, float(terminated))   # 시간 잘림(truncated)은 끝이 아님 → 0으로 저장
    obs, ep_ret = next_obs, ep_ret + r
    if terminated or truncated:
        ep_returns.append(ep_ret)
        log.append((step, ep_ret))
        obs, _ = env.reset()
        ep_ret = 0.0

    # ③ 학습: 버퍼에서 무작위 배치
    if step >= LEARN_START:
        s, a, r_b, s2, d = buf.sample(BATCH)
        q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)          # 내가 한 행동의 Q(s,a)
        with torch.no_grad():
            tnet = q if args.no_target else q_target
            if args.no_double:
                next_q = tnet(s2).max(1).values                      # DQN: 타깃망이 고르고 평가
            else:
                a2 = q(s2).argmax(1, keepdim=True)                   # Double: 온라인망이 고르고
                next_q = tnet(s2).gather(1, a2).squeeze(1)           #         타깃망이 평가
            y = r_b + GAMMA * (1 - d) * next_q                        # 끝난 상태면 미래 가치 0
        loss = F.smooth_l1_loss(q_sa, y)                              # Huber 손실
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()

    # ④ 타깃망 주기적 복사 (hard update)
    if not args.no_target and step % TARGET_EVERY == 0:
        q_target.load_state_dict(q.state_dict())

    if step % 5000 == 0:
        recent = np.mean(ep_returns[-20:]) if ep_returns else 0
        print(f"step {step:6d} | 최근 20판 평균 {recent:6.1f} | ε {eps:.2f}")

tag = "dqn" + ("_nodouble" if args.no_double else "") + ("_notarget" if args.no_target else "") + f"_s{args.seed}"
np.savetxt(f"results/{tag}.txt", np.array(log))

# ⑤ 평가: 탐험 없이(greedy) 10판
eval_env = gym.make("CartPole-v1")
scores = []
for i in range(10):
    o, _ = eval_env.reset(seed=1000 + i)
    done, total = False, 0.0
    while not done:
        with torch.no_grad():
            o, r, te, tr, _ = eval_env.step(int(q(torch.as_tensor(o)).argmax()))
        total += r
        done = te or tr
    scores.append(total)
print(f"평가(탐험 없음) 10판 평균: {np.mean(scores):.1f} (최대 500)")
