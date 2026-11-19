import importlib.util
import numpy as np
import torch
import gymnasium as gym
import extra_snippets as X

ok = lambda n, c: print(("통과 " if c else "실패 ") + n)
torch.manual_seed(0); rng = np.random.default_rng(0)

q = X.DuelingQ(4, 3); out = q(torch.randn(5, 4))
z = q.body(torch.randn(5, 4)); a = q.a(z)
ok("X1 Dueling: 출력 모양 (5,3), 행동별 이점의 평균이 0이 되도록 빼서 Q의 평균 = V", out.shape == (5, 3) and torch.allclose((q.v(z) + a - a.mean(1, keepdim=True)).mean(1), q.v(z).squeeze(1)))

ok("X2 n-step: [1,1,1], γ=0.5, 뒤 가치 8 → 1+0.5+0.25+0.125×8 = 2.75", np.isclose(X.n_step_target([1, 1, 1], 0.5, 8.0, False), 2.75) and np.isclose(X.n_step_target([1, 1, 1], 0.5, 8.0, True), 1.75))

# X3: CartPole에서 간단한 전문가 규칙(막대가 기우는 쪽으로 카트 밀기)의 시연을 따라 배우기
env = gym.make("CartPole-v1"); expert = lambda o: int(o[2] + 0.5 * o[3] > 0)
S, A = [], []
for ep in range(20):
    o, _ = env.reset(seed=ep); done = False
    while not done:
        a_ = expert(o); S.append(o); A.append(a_); o, r, te, tr, _ = env.step(a_); done = te or tr
pol = X.behavior_cloning(torch.as_tensor(np.array(S)), torch.as_tensor(A), 2)
def score(policy_fn, n=5):
    tot = []
    for i in range(n):
        o, _ = env.reset(seed=100 + i); done, t = False, 0
        while not done:
            o, r, te, tr, _ = env.step(policy_fn(o)); t += r; done = te or tr
        tot.append(t)
    return np.mean(tot)
s_exp = score(expert); s_bc = score(lambda o: int(pol(torch.as_tensor(o)).argmax()))
ok(f"X3 행동 복제: 전문가 {s_exp:.0f}점 → 따라 배운 정책 {s_bc:.0f}점 (시연 {len(S)}스텝)", s_bc >= 0.9 * s_exp)

adv = X.grpo_advantage([1, 0, 0, 1, 1, 0, 0, 0])
ok(f"X4 GRPO 이점: 평균 0·표준편차 1, 성공=+{adv[0]:.2f}, 실패={adv[1]:.2f}", np.isclose(adv.mean(), 0) and np.isclose(adv.std(), 1, atol=1e-6) and adv[0] > 0 > adv[1])

c = np.array([0.0, 0.0])
for _ in range(20): c, w = X.update_subgoal_weights(c, np.array([1, 0]))
ok(f"X5 하위 목표: 늘 성공하는 목표 가중치 {w[0]:.2f}↓, 늘 실패하는 목표 {w[1]:.2f}(=1)", w[0] < 0.2 and np.isclose(w[1], 1))

spec = importlib.util.spec_from_file_location("rr", "09_rrt.py"); rr = importlib.util.module_from_spec(spec); spec.loader.exec_module(rr)
p, *_ = rr.rrt(); sp = X.shortcut(p, rr.edge_free)
ok(f"X6 경로 다듬기: RRT 길이 {rr.path_length(p):.2f} → {rr.path_length(np.array(sp)):.2f}, 다듬은 선분도 모두 충돌 없음",
   rr.path_length(np.array(sp)) < rr.path_length(p) and all(rr.edge_free(sp[i], sp[i + 1]) for i in range(len(sp) - 1)))

ok("X7 이동평균: [0..9], w=5 → 길이 6, 첫 값 2", len(X.moving_average(range(10), 5)) == 6 and np.isclose(X.moving_average(range(10), 5)[0], 2))
