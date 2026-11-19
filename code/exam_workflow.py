"""3교시 진행 순서(06 문서 0장)에서 쓰는 도구 함수.
환경 확인 → 무작위 기준선 → 정책 평가 → 시드 요약 → 실험 기록
"""
import time
import numpy as np
import gymnasium as gym


# ── W1 환경 확인: 이산/연속, 판 길이, 1,000스텝 걸리는 시간 ──
def inspect_env(env_id):
    env = gym.make(env_id)
    env.reset(seed=0)
    env.action_space.seed(0)
    kind = "이산" if isinstance(env.action_space, gym.spaces.Discrete) else "연속"
    print("관측 공간:", env.observation_space)
    print("행동 공간:", env.action_space, "→", kind)
    print("판 최대 길이:", env.spec.max_episode_steps)
    t0 = time.time()
    for _ in range(1000):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    print(f"1,000스텝(무작위 행동) {time.time() - t0:.2f}초")
    return kind


# ── W2 정책 평가: 탐험 없는 정책으로 여러 판, 평균·표준편차·평균 길이 ──
def evaluate(policy_fn, env_id, episodes=10, seed=1000):
    env = gym.make(env_id)
    scores, lengths = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done, total, n = False, 0.0, 0
        while not done:
            obs, r, term, trunc, _ = env.step(policy_fn(obs))
            total += r
            n += 1
            done = term or trunc
        scores.append(total)
        lengths.append(n)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(lengths))


# ── W3 무작위 기준선: 학습 전 "바닥 점수" ──
def random_baseline(env_id, episodes=10):
    space = gym.make(env_id).action_space
    space.seed(0)
    return evaluate(lambda obs: space.sample(), env_id, episodes)


# ── W4 시드 여러 개 결과 요약 ──
def summarize(results):
    r = np.asarray(results, dtype=float)
    return f"{r.mean():.1f} ± {r.std():.1f} (시드 {len(r)}개)"


# ── W5 실험 기록: 한 줄씩 표로 남기기 ──
def log_result(table, name, change, score_text, note=""):
    table.append((name, change, score_text, note))
    print(f"| {name} | {change} | {score_text} | {note} |")
