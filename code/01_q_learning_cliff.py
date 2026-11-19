"""01. 표 형태 Q-learning vs SARSA (CliffWalking, Sutton & Barto 예제 6.6)
실행: python 01_q_learning_cliff.py
"""
import numpy as np
import gymnasium as gym


def eps_greedy(Q, s, eps, rng):
    """확률 eps로 무작위 행동(탐험), 나머지는 Q가 가장 큰 행동(활용)."""
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    return int(np.argmax(Q[s]))


def train(method, episodes=1000, alpha=0.1, gamma=1.0, eps=0.1, seed=0):
    env = gym.make("CliffWalking-v1")            # 4×12 격자, 매 스텝 −1, 절벽 −100(출발점으로)
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))   # 상태 48 × 행동 4
    returns = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        a = eps_greedy(Q, s, eps, rng)
        G, done = 0.0, False
        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            G += r
            done = terminated or truncated
            a2 = eps_greedy(Q, s2, eps, rng)       # 다음에 실제로 할 행동
            if method == "sarsa":                  # on-policy: 실제로 할 행동 a2의 Q
                target = r + gamma * Q[s2, a2] * (not terminated)
            else:                                  # Q-learning, off-policy: 가장 좋은 행동의 Q
                target = r + gamma * np.max(Q[s2]) * (not terminated)
            Q[s, a] += alpha * (target - Q[s, a])  # TD 오차만큼 조금 이동
            s, a = s2, a2
        returns.append(G)
    return Q, np.array(returns)


def greedy_path(Q):
    """학습된 Q로 탐험 없이 갔을 때의 길을 격자에 그림."""
    env = gym.make("CliffWalking-v1")
    s, _ = env.reset(seed=0)
    grid = [["." for _ in range(12)] for _ in range(4)]
    for c in range(1, 11):
        grid[3][c] = "C"                           # 절벽
    for _ in range(100):
        r_, c_ = divmod(s, 12)
        grid[r_][c_] = "*"
        s, _, term, trunc, _ = env.step(int(np.argmax(Q[s])))
        if term or trunc:
            break
    grid[3][11] = "G"
    return "\n".join("".join(row) for row in grid)


if __name__ == "__main__":
    for m in ["sarsa", "qlearning"]:
        Q, R = train(m)
        np.savetxt(f"results/cliff_{m}.txt", R)
        print(f"[{m}] 마지막 100판 평균 보상(탐험 포함): {R[-100:].mean():.1f}")
        print(greedy_path(Q), "\n")
