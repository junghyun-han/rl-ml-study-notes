"""3교시 예상 문제 전수 목록(06)에 실리는 추가 코드. 검증: python extra_snippets_check.py"""
import numpy as np
import torch
import torch.nn as nn


# ── X1 Dueling DQN 머리: Q = V + (A − A의 평균) ──
class DuelingQ(nn.Module):
    def __init__(self, obs_dim, n_act, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU())
        self.v = nn.Linear(h, 1)           # 상태 가치 V(s)
        self.a = nn.Linear(h, n_act)       # 행동별 이점 A(s, a)

    def forward(self, x):
        z = self.body(x)
        a = self.a(z)
        return self.v(z) + a - a.mean(dim=1, keepdim=True)


# ── X2 n-step 목표값 ──
def n_step_target(rewards, gamma, bootstrap_value, done):
    """rewards = [r_t, …, r_(t+n−1)], bootstrap_value = n스텝 뒤 상태의 가치 추정(max Q 또는 V)"""
    G = sum(gamma ** k * r for k, r in enumerate(rewards))
    if not done:
        G += gamma ** len(rewards) * bootstrap_value
    return G


# ── X3 행동 복제(behavior cloning): 시연 (상태, 행동)으로 지도학습 ──
def behavior_cloning(states, actions, n_act, epochs=200, lr=1e-2):
    policy = nn.Sequential(nn.Linear(states.shape[1], 64), nn.ReLU(), nn.Linear(64, n_act))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()          # 연속 행동이면 nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(policy(states), actions)
        loss.backward()
        opt.step()
    return policy


# ── X4 GRPO 그룹 상대 이점: 같은 과제를 여러 번 시도한 보상들을 그룹 안에서 정규화 ──
def grpo_advantage(group_rewards):
    r = np.asarray(group_rewards, dtype=float)
    return (r - r.mean()) / (r.std() + 1e-8)


# ── X5 하위 목표 숙련도·보상 가중치 (Agentic-VLA 보상 합성 식) ──
def update_subgoal_weights(c, success, alpha=0.9):
    """c_k ← α·c_k + (1 − α)·성공(0/1),  가중치 w_k = 1 − c_k"""
    c = alpha * c + (1 - alpha) * success
    return c, 1 - c


# ── X6 경로 다듬기(shortcut): 두 점을 직선으로 이어도 충돌이 없으면 중간 점을 지움 ──
def shortcut(path, edge_free, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    path = [np.asarray(p, dtype=float) for p in path]
    for _ in range(iters):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i >= 2 and edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


# ── X7 학습 곡선 이동평균 (출렁이는 리턴을 부드럽게 보기) ──
def moving_average(x, w=20):
    x = np.asarray(x, dtype=float)
    return np.convolve(x, np.ones(w) / w, mode="valid")
