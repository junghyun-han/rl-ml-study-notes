"""11. 탐험-활용: 10개 팔 밴딧에서 greedy · ε-greedy · UCB 비교 (Sutton & Barto 2장 실험과 같은 설정)
실행: python 11_bandit.py
"""
import numpy as np

K, STEPS, RUNS = 10, 1000, 300                    # 팔 10개, 1000번 당기기, 300번 반복 평균


def argmax_random(x, rng):
    """최댓값이 여러 개면 그중 무작위 (항상 첫 번째를 고르면 편향)"""
    return int(rng.choice(np.flatnonzero(x == x.max())))


def run(method, eps=0.0, c=2.0, seed=0):
    rng = np.random.default_rng(seed)
    avg_reward, best_ratio = np.zeros(STEPS), np.zeros(STEPS)
    for _ in range(RUNS):
        q_true = rng.normal(0, 1, K)              # 팔마다 진짜 평균 보상 (에이전트는 모름)
        best = int(np.argmax(q_true))
        Q, N = np.zeros(K), np.zeros(K)           # 추정 가치, 당긴 횟수
        for t in range(1, STEPS + 1):
            if method == "ucb":
                if (N == 0).any():
                    a = int(np.flatnonzero(N == 0)[0])                # 안 당겨 본 팔부터 한 번씩
                else:
                    a = argmax_random(Q + c * np.sqrt(np.log(t) / N), rng)   # 덜 당긴 팔에 가산점
            else:
                a = int(rng.integers(K)) if rng.random() < eps else argmax_random(Q, rng)
            r = rng.normal(q_true[a], 1)          # 보상 = 진짜 평균 + 잡음
            N[a] += 1
            Q[a] += (r - Q[a]) / N[a]             # 증분 평균: 새 평균 = 옛 평균 + (새 값 − 옛 평균)/n
            avg_reward[t - 1] += r
            best_ratio[t - 1] += (a == best)
    return avg_reward / RUNS, best_ratio / RUNS


if __name__ == "__main__":
    results = {}
    for name, kw in [("greedy (ε=0)", dict(method="eps", eps=0.0)), ("ε=0.01", dict(method="eps", eps=0.01)),
                     ("ε=0.1", dict(method="eps", eps=0.1)), ("UCB c=2", dict(method="ucb", c=2.0))]:
        r, b = run(**kw)
        results[name] = (r, b)
        print(f"{name:13s} | 마지막 100번 평균 보상 {r[-100:].mean():.3f} | 최적 팔 고른 비율 {b[-100:].mean() * 100:5.1f}%")
    np.savez("results/bandit.npz", **{k.replace(" ", "_"): np.stack(v) for k, v in results.items()})
