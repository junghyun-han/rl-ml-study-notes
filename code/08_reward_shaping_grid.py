"""08. 보상 설계 비교 — 같은 격자, 같은 Q-learning, 보상만 4가지
  sparse    : 도착 +1, 나머지 0                      → 신호가 드물어 느림
  step      : 매 스텝 −0.01, 도착 +1                 → 빨리 도착하도록 재촉
  potential : sparse + γΦ(s') − Φ(s), Φ = −거리      → 빨라지고 최적 정책은 그대로 (Ng 외 1999)
  near_bonus: sparse + 목표 1칸 이내면 매 스텝 +0.1   → 보상 해킹: 도착 안 하고 근처에서 맴돎
실행: python 08_reward_shaping_grid.py
"""
import numpy as np

N, MAX_STEPS, GAMMA = 8, 100, 0.99
GOAL = (N - 1, N - 1)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def dist(s):
    return abs(s[0] - GOAL[0]) + abs(s[1] - GOAL[1])


def step_env(s, a):
    r_, c_ = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    s2 = (min(max(r_, 0), N - 1), min(max(c_, 0), N - 1))
    return s2, s2 == GOAL                                  # (다음 상태, 도착=종료)


def reward(mode, s, s2, done):
    base = 1.0 if done else 0.0
    if mode == "sparse":
        return base
    if mode == "step":
        return base - 0.01
    if mode == "potential":
        phi = lambda x: -dist(x) / (2 * (N - 1))          # 거리를 0~−1로 맞춘 잠재함수
        phi_next = 0.0 if done else phi(s2)                # 끝난 상태의 Φ = 0
        return base + GAMMA * phi_next - phi(s)
    if mode == "near_bonus":
        return base + (0.1 if (not done and dist(s2) <= 1) else 0.0)


def run(mode, episodes=300, alpha=0.5, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N, N, 4))
    first_success = None
    for ep in range(episodes):
        s = (0, 0)
        for t in range(MAX_STEPS):
            a = int(rng.integers(4)) if rng.random() < eps else int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            s2, done = step_env(s, a)
            r = reward(mode, s, s2, done)
            target = r + GAMMA * (0.0 if done else Q[s2].max())   # 시간 잘림은 끝이 아니므로 부트스트랩
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2
            if done:
                if first_success is None:
                    first_success = ep
                break
    # 학습 끝난 뒤 탐험 없이 한 판: 도착했나, 몇 스텝?
    s, steps, reached = (0, 0), 0, False
    for t in range(MAX_STEPS):
        s, done = step_env(s, int(np.argmax(Q[s])))
        steps += 1
        if done:
            reached = True
            break
    return first_success, reached, steps


if __name__ == "__main__":
    print(f"{N}×{N} 격자, 최단 거리 {2 * (N - 1)}스텝, 시드 5개")
    for mode in ["sparse", "step", "potential", "near_bonus"]:
        res = [run(mode, seed=s) for s in range(5)]
        fs = [r[0] for r in res]
        print(f"{mode:10s} | 처음 도착한 판: {fs} | 학습 후 도착: {[r[1] for r in res]} | 학습 후 스텝: {[r[2] for r in res]}")
