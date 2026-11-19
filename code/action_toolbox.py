"""상황별 행동 코드집에서 쓰는 도구 (모두 실행 확인: python action_toolbox_check.py)
정규화 켜기 · 평가용 환경(정규화 값 복사·고정) · 학습률 바꾸기/줄이기 · PPO 근사 KL · 모델 저장/불러오기 · 시드 여러 개 실행 · 학습 곡선
"""
import copy
import re
import subprocess
import sys
import numpy as np
import torch
import gymnasium as gym


# ── T1 환경 만들기: 관측·보상 정규화, 행동 자르기 켜고 끄기 ──
def make_env(env_id, norm_obs=False, norm_reward=False, clip_action=False, gamma=0.99):
    env = gym.make(env_id)
    if clip_action:
        env = gym.wrappers.ClipAction(env)                      # 연속 행동을 범위 안으로 자름
    if norm_obs:
        env = gym.wrappers.NormalizeObservation(env)            # 관측을 이동 평균·분산으로 정규화
    if norm_reward:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)    # 보상 크기 맞추기(학습용)
    return env


def find_wrapper(env, cls):
    """감싼 층들을 안쪽으로 따라가며 cls 종류의 Wrapper를 찾음"""
    e = env
    while True:
        if isinstance(e, cls):
            return e
        if not hasattr(e, "env"):
            return None
        e = e.env


# ── T2 평가용 환경: 관측 정규화 값을 복사해 고정, 보상 정규화는 끔(원래 점수로 평가) ──
def make_eval_env(env_id, train_env, clip_action=False):
    src = find_wrapper(train_env, gym.wrappers.NormalizeObservation)
    ev = make_env(env_id, norm_obs=src is not None, clip_action=clip_action)
    if src is not None:
        dst = find_wrapper(ev, gym.wrappers.NormalizeObservation)
        dst.obs_rms = copy.deepcopy(src.obs_rms)                # 학습 때 쌓은 평균·분산 복사
        dst.update_running_mean = False                         # 평가 중에는 갱신하지 않음
    return ev


def evaluate_env(env, policy_fn, episodes=10, seed=1000):
    """이미 만든 환경 객체로 평가 (정규화된 관측을 정책에 넣고, 점수는 원래 보상으로)"""
    scores = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy_fn(obs))
            total += r
            done = term or trunc
        scores.append(total)
    return float(np.mean(scores)), float(np.std(scores))


# ── T3 학습률 바꾸기 / 학습하며 선형으로 줄이기 ──
def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def linear_decay(opt, total_updates):
    """매 업데이트 끝에 sched.step() 호출 → 학습률이 처음 값에서 0까지 선형으로 감소"""
    return torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=total_updates)


# ── T4 PPO 근사 KL: 미니배치마다 계산해 목표를 넘으면 그 업데이트의 남은 epoch를 멈춤 ──
def approx_kl(new_logp, old_logp):
    log_ratio = new_logp - old_logp
    return ((log_ratio.exp() - 1) - log_ratio).mean().item()


# ── T5 가장 좋은 모델 저장·불러오기 ──
def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


# ── T6 시드 여러 개 실행해 FINAL 줄에서 평가 점수 모으기 ──
def run_seeds(script, args, seeds):
    scores = []
    for s in seeds:
        out = subprocess.run([sys.executable, script, *args, "--seed", str(s)], capture_output=True, text=True).stdout
        m = re.search(r"FINAL seed \d+: 평가 10판 (-?[\d.]+)", out)
        scores.append(float(m.group(1)))
        print(f"seed {s}: {scores[-1]}")
    r = np.array(scores)
    print(f"평균 {r.mean():.1f} ± {r.std():.1f} (시드 {len(r)}개)")
    return scores


# ── T7 학습 곡선 그리기 (시드별 판 리턴 → 이동평균) ──
def plot_curves(curves, path, w=20, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, ret in curves.items():
        ret = np.asarray(ret, dtype=float)
        k = min(w, len(ret))
        plt.plot(np.convolve(ret, np.ones(k) / k, mode="valid"), label=name)
    plt.xlabel("episode"); plt.ylabel(f"return (moving avg {w})"); plt.title(title); plt.legend()
    plt.savefig(path, dpi=120); plt.close()
