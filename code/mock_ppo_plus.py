"""모의 실습 M7: PPO 개선판 한 파일 — 이산·연속 행동 자동 구분, Gym·MuJoCo 모두. 개선 옵션을 하나씩 켜며 비교한다.
실행 예: python mock_ppo_plus.py --env CartPole-v1 --steps 60000
        python mock_ppo_plus.py --env InvertedPendulum-v5 --steps 100000 --norm-obs --norm-reward --lr-decay --target-kl 0.02 --save-best
"""
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical, Normal

p = argparse.ArgumentParser()
p.add_argument("--env", default="CartPole-v1")
p.add_argument("--steps", type=int, default=100_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--ent", type=float, default=0.0)          # 엔트로피 계수 (엔트로피가 금방 0이면 올림)
p.add_argument("--epochs", type=int, default=10)          # 같은 데이터로 학습하는 횟수 (KL이 크면 내림)
p.add_argument("--clip", type=float, default=0.2)         # clip ε
p.add_argument("--norm-obs", action="store_true")         # 관측 정규화
p.add_argument("--norm-reward", action="store_true")      # 보상 정규화 (학습용, 평가는 원래 보상)
p.add_argument("--lr-decay", action="store_true")         # 학습률 선형 감소 → 끝에 0
p.add_argument("--target-kl", type=float, default=0.0)    # 0보다 크면 KL 조기 종료
p.add_argument("--save-best", action="store_true")        # 학습 리턴이 가장 좋았던 시점의 모델로 평가
args = p.parse_args()

N_STEPS, MB, GAMMA, LAM, VF, MAX_GRAD = 2048, 64, 0.99, 0.95, 0.5, 0.5
torch.manual_seed(args.seed); np.random.seed(args.seed)


def make_env(train):
    """원래 점수 기록 → (관측 정규화) → (학습 때만 보상 정규화). 관측 정규화 층도 돌려줌"""
    env = gym.wrappers.RecordEpisodeStatistics(gym.make(args.env))
    obs_norm = None
    if args.norm_obs:
        env = obs_norm = gym.wrappers.NormalizeObservation(env)
    if args.norm_reward and train:
        env = gym.wrappers.NormalizeReward(env, gamma=GAMMA)
    return env, obs_norm


env, obs_norm = make_env(train=True)
disc = isinstance(env.action_space, gym.spaces.Discrete)            # 이산이면 Categorical, 연속이면 Normal
n_obs = env.observation_space.shape[0]
n_out = env.action_space.n if disc else env.action_space.shape[0]
if not disc:
    low, high = env.action_space.low, env.action_space.high


def layer(i, o, std=np.sqrt(2)):
    l = nn.Linear(i, o); nn.init.orthogonal_(l.weight, std); nn.init.zeros_(l.bias); return l


actor = nn.Sequential(layer(n_obs, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, n_out, 0.01))
critic = nn.Sequential(layer(n_obs, 64), nn.Tanh(), layer(64, 64), nn.Tanh(), layer(64, 1, 1.0))
log_std = nn.Parameter(torch.zeros(n_out))                          # 연속 행동일 때만 사용
params = list(actor.parameters()) + list(critic.parameters()) + ([] if disc else [log_std])
opt = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
n_updates = args.steps // N_STEPS
sched = torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=n_updates) if args.lr_decay else None


def policy(o):
    """분포 하나 돌려줌. log_prob·entropy는 연속이면 행동 차원으로 합산"""
    out = actor(o)
    return Categorical(logits=out) if disc else Normal(out, log_std.exp())


def logp_ent(dist, a):
    if disc:
        return dist.log_prob(a), dist.entropy()
    return dist.log_prob(a).sum(-1), dist.entropy().sum(-1)


obs, _ = env.reset(seed=args.seed)
returns, step, best, best_state = [], 0, -np.inf, None
for update in range(n_updates):
    # ① 지금 정책으로 N_STEPS 모으기
    O, A, LP, R, D, V = [], [], [], [], [], []
    for _ in range(N_STEPS):
        o = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = policy(o); a = dist.sample(); lp, _ = logp_ent(dist, a); v = critic(o).squeeze()
        act = a.item() if disc else np.clip(a.numpy(), low, high)      # 환경에 넣을 때만 범위로 자름
        nxt, r, term, trunc, info = env.step(act); step += 1
        if trunc and not term:                                          # 시간 잘림 → 미래 가치로 부트스트랩
            with torch.no_grad():
                r += GAMMA * critic(torch.as_tensor(nxt, dtype=torch.float32)).item()
        O.append(o); A.append(a); LP.append(lp); R.append(float(r)); D.append(float(term or trunc)); V.append(v)
        obs = nxt
        if term or trunc:
            returns.append(float(info["episode"]["r"]))                 # 정규화 전 원래 점수
            obs, _ = env.reset()
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
    # ③ 미니배치 업데이트 + 진단값(KL·clip 비율·엔트로피) 기록, KL 조기 종료
    kls, clips, ents, stop = [], [], [], False
    for _ in range(args.epochs):
        for idx in torch.randperm(N_STEPS).split(MB):
            new_lp, ent = logp_ent(policy(O[idx]), A[idx])
            log_ratio = new_lp - LP[idx]; ratio = log_ratio.exp()
            kl = ((ratio - 1) - log_ratio).mean().item()               # 근사 KL
            if args.target_kl > 0 and kl > 1.5 * args.target_kl:       # 너무 많이 바뀌면 이번 업데이트 중단
                stop = True; break
            a_n = (adv[idx] - adv[idx].mean()) / (adv[idx].std() + 1e-8)
            pg = -torch.min(ratio * a_n, ratio.clamp(1 - args.clip, 1 + args.clip) * a_n).mean()
            vl = ((critic(O[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
            loss = pg + VF * vl - args.ent * ent.mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, MAX_GRAD); opt.step()
            kls.append(kl); clips.append(((ratio - 1).abs() > args.clip).float().mean().item()); ents.append(ent.mean().item())
        if stop:
            break
    if sched:
        sched.step()
    recent = np.mean(returns[-20:]) if returns else float("nan")
    if args.save_best and returns and recent > best:                    # 최고 시점 저장 (관측 정규화 값 포함)
        best = recent
        best_state = (copy.deepcopy(actor.state_dict()), log_std.detach().clone(), copy.deepcopy(obs_norm.obs_rms) if obs_norm else None)
    if (update + 1) % 5 == 0:
        print(f"step {step:7d} | 최근 20판 리턴 {recent:8.1f} | KL {np.mean(kls) if kls else 0:.4f} | clip {np.mean(clips) if clips else 0:.3f} | "
              f"엔트로피 {np.mean(ents) if ents else 0:.3f} | lr {opt.param_groups[0]['lr']:.1e}{' | KL 조기 종료' if stop else ''}")

# ④ 탐험 없이 평가 (이산 argmax / 연속 평균 행동), 관측 정규화 값은 복사·고정, 보상은 원래 점수
rms = obs_norm.obs_rms if obs_norm else None
if best_state:
    actor.load_state_dict(best_state[0]); rms = best_state[2]
    print(f"최고 시점 모델로 평가 (학습 리턴 {best:.1f})")
ev, ev_norm = make_env(train=False)
if ev_norm:
    ev_norm.obs_rms = copy.deepcopy(rms); ev_norm.update_running_mean = False
scores, lengths = [], []
for i in range(10):
    o, _ = ev.reset(seed=1000 + i); done, total, n = False, 0.0, 0
    while not done:
        with torch.no_grad():
            out = actor(torch.as_tensor(o, dtype=torch.float32))
        act = int(out.argmax()) if disc else np.clip(out.numpy(), low, high)
        o, r, term, trunc, _ = ev.step(act); total += r; n += 1; done = term or trunc
    scores.append(total); lengths.append(n)
print(f"FINAL seed {args.seed}: 평가 10판 {np.mean(scores):.1f} ± {np.std(scores):.1f} (평균 길이 {np.mean(lengths):.0f})")
