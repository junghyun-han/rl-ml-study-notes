"""10. 가치 반복 · 정책 반복 (전이 확률을 아는 MDP를 동적 계획법으로 풀기) — FrozenLake-v1 (4×4, 미끄러움)
환경의 전이 확률을 그대로 쓴다:  env.unwrapped.P[s][a] = [(확률, 다음 상태, 보상, 끝났나), ...]
실행: python 10_value_iteration.py
"""
import numpy as np
import gymnasium as gym

GAMMA, THETA = 0.99, 1e-8                         # 할인율, 수렴 판정 기준

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
P = env.unwrapped.P
nS, nA = env.observation_space.n, env.action_space.n   # 16, 4 (0=왼 1=아래 2=오른 3=위)


def q_from_v(V, s):
    """상태 s에서 행동별 Q(s, a) = Σ p · [r + γ V(s′)]  (끝나는 전이면 미래 가치 0)"""
    q = np.zeros(nA)
    for a in range(nA):
        for prob, s2, r, done in P[s][a]:
            q[a] += prob * (r + GAMMA * V[s2] * (not done))
    return q


def value_iteration():
    V = np.zeros(nS)
    it = 0
    while True:
        it += 1
        delta = 0.0
        for s in range(nS):
            v_new = q_from_v(V, s).max()                  # 벨만 최적 방정식: 행동 중 최댓값
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new                                  # 제자리 갱신 (바로 다음 상태 계산에 씀)
        if delta < THETA:                                 # 가장 크게 바뀐 값도 아주 작으면 수렴
            break
    pi = np.array([q_from_v(V, s).argmax() for s in range(nS)])   # V*에서 greedy 정책 뽑기
    return V, pi, it


def policy_evaluation(pi, V):
    """정책 π를 고정하고 V^π 계산: 벨만 기대 방정식을 수렴할 때까지 반복"""
    while True:
        delta = 0.0
        for s in range(nS):
            v_new = q_from_v(V, s)[pi[s]]                 # 결정적 정책: Σ_a π(a|s) Q(s,a) = Q(s, π(s))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < THETA:
            return V


def policy_iteration():
    pi = np.zeros(nS, dtype=int)                          # 아무 정책(전부 '왼쪽')에서 시작
    V = np.zeros(nS)
    rounds = 0
    while True:
        rounds += 1
        V = policy_evaluation(pi, V)                      # ① 평가
        new_pi = np.array([q_from_v(V, s).argmax() for s in range(nS)])   # ② 개선 (greedy)
        if np.array_equal(new_pi, pi):                    # 정책이 더 안 바뀌면 최적
            return V, pi, rounds
        pi = new_pi


def success_rate(pi, episodes=2000, seed=0):
    """정책대로 실제로 해 보고 목표 도착 비율 재기"""
    e = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
    wins = 0.0
    for i in range(episodes):
        s, _ = e.reset(seed=seed + i)
        done = False
        while not done:
            s, r, term, trunc, _ = e.step(int(pi[s]))
            done = term or trunc
        wins += r                                         # 목표 도착 때만 r = 1
    return wins / episodes


if __name__ == "__main__":
    V1, pi1, it = value_iteration()
    V2, pi2, rounds = policy_iteration()
    arrows = np.array(["←", "↓", "→", "↑"])
    desc = ["".join(c.decode() for c in row) for row in env.unwrapped.desc]
    print("지도 (S 출발, F 얼음, H 구멍, G 목표):", " / ".join(desc))
    print(f"가치 반복: {it}번 반복 / 정책 반복: {rounds}번 (평가 → 개선)")
    print(f"두 방법 V 최대 차이 {np.abs(V1 - V2).max():.1e}, 정책 같음: {np.array_equal(pi1, pi2)}")
    print("V*:\n", V1.reshape(4, 4).round(3))
    print("π*:\n", arrows[pi1].reshape(4, 4))
    print(f"최적 정책 성공률(2000판): {success_rate(pi1):.3f}  /  무작위가 아닌 '전부 오른쪽' 정책: {success_rate(np.full(nS, 2)):.3f}")
