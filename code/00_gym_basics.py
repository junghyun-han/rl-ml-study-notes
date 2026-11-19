"""00. gymnasium 기본 API + 내 환경 만들기 + 보상 Wrapper
실행: python 00_gym_basics.py
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

# ── 1) 이미 있는 환경을 무작위 행동으로 한 판 돌리기 ──────────────────
env = gym.make("CartPole-v1")
print("관측 공간:", env.observation_space)       # Box(4,) : 카트 위치·속도, 막대 각도·각속도
print("행동 공간:", env.action_space)            # Discrete(2) : 0=왼쪽, 1=오른쪽
obs, info = env.reset(seed=0)                    # reset은 (관측, 정보) 두 개를 돌려줌
total, steps = 0.0, 0
while True:
    action = env.action_space.sample()           # 무작위 행동
    obs, reward, terminated, truncated, info = env.step(action)   # step은 5개를 돌려줌
    total += reward
    steps += 1
    if terminated or truncated:                  # terminated=실패/성공으로 끝, truncated=시간 제한으로 잘림
        break
print(f"무작위 정책: {steps}스텝, 보상 합 {total}")
env.close()


# ── 2) 내 환경을 gym.Env 규칙대로 만들기 ─────────────────────────────
class GridEnv(gym.Env):
    """n×n 격자. (0,0)에서 출발해 (n-1,n-1)에 도착하면 성공."""

    def __init__(self, n=5, max_steps=50):
        super().__init__()
        self.n, self.max_steps = n, max_steps
        self.observation_space = spaces.Box(0, n - 1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.moves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])   # 위·아래·왼·오
        self.goal = np.array([n - 1, n - 1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                  # self.np_random 시드 설정
        self.pos = np.array([0, 0])
        self.t = 0
        return self.pos.astype(np.float32), {}

    def step(self, action):
        self.pos = np.clip(self.pos + self.moves[action], 0, self.n - 1)   # 벽 밖으로 못 나감
        self.t += 1
        terminated = bool((self.pos == self.goal).all())                  # 목표 도착 = 진짜 끝
        truncated = self.t >= self.max_steps                               # 시간 초과 = 잘림
        reward = 1.0 if terminated else -0.01                              # 도착 +1, 매 스텝 −0.01
        return self.pos.astype(np.float32), reward, terminated, truncated, {}


check_env(GridEnv())                              # 규칙(반환 형식·공간)을 지켰는지 자동 검사
print("GridEnv check_env 통과")


# ── 3) Wrapper로 보상만 바꾸기 (환경 코드는 그대로) ──────────────────
class DistanceShaping(gym.Wrapper):
    """잠재 기반 보상 설계: r' = r + γΦ(s') − Φ(s),  Φ(s) = −(목표까지 맨해튼 거리)"""

    def __init__(self, env, gamma=0.99, scale=0.1):
        super().__init__(env)
        self.gamma, self.scale = gamma, scale

    def phi(self, obs):
        return -self.scale * np.abs(obs - self.env.unwrapped.goal).sum()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_phi = self.phi(obs)
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        phi_next = 0.0 if terminated else self.phi(obs)      # 끝난 상태의 Φ는 0
        shaped = r + self.gamma * phi_next - self.last_phi
        self.last_phi = phi_next
        info["original_reward"] = r                           # 원래 보상은 따로 기록 (평가용)
        return obs, shaped, terminated, truncated, info


env = DistanceShaping(GridEnv())
obs, _ = env.reset(seed=0)
obs, r, term, trunc, info = env.step(3)                       # 오른쪽 = 목표 쪽으로 한 칸
print(f"목표 쪽으로 한 칸: 설계 보상 {r:+.3f} (원래 {info['original_reward']:+.2f})")
obs, r, term, trunc, info = env.step(2)                       # 왼쪽 = 목표 반대
print(f"반대로 한 칸:     설계 보상 {r:+.3f} (원래 {info['original_reward']:+.2f})")

# 상태가 필요 없는 보상 변환은 RewardWrapper가 더 간단
class ScaleReward(gym.RewardWrapper):
    def reward(self, reward):
        return 0.1 * reward
