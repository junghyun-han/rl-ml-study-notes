"""실습 중 자주 만나는 오류를 일부러 내서 실제 메시지(첫 줄)를 확인한다. 실행: python error_messages_check.py"""
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym


def show(name, fn):
    try:
        fn()
        print(f"[{name}] 오류 없음")
    except Exception as e:
        print(f"[{name}] {type(e).__name__}: {str(e).splitlines()[0][:160]}")


net = nn.Linear(4, 2)
show("E01 float64 관측을 그대로 넣음", lambda: net(torch.as_tensor(np.zeros(4))))
show("E02 입력 크기 불일치", lambda: net(torch.zeros(3)))
def e03():
    with torch.no_grad():
        loss = net(torch.zeros(4)).sum()
    loss.backward()
show("E03 no_grad 안에서 만든 손실을 backward", e03)
def e04():
    loss = net(torch.zeros(4)).sum(); loss.backward(); loss.backward()
show("E04 같은 그래프로 backward 두 번", e04)
show("E05 gather 인덱스가 float", lambda: torch.zeros(3, 2).gather(1, torch.zeros(3, 1)))
show("E06 CrossEntropy 정답이 int32", lambda: nn.CrossEntropyLoss()(torch.zeros(3, 2), torch.zeros(3, dtype=torch.int32)))
show("E07 CrossEntropy 정답이 float(클래스 번호)", lambda: nn.CrossEntropyLoss()(torch.zeros(3, 2), torch.tensor([0., 1., 0.])))
show("E08 환경 이름 오타", lambda: gym.make("CartPol-v1"))
show("E09 없는 버전", lambda: gym.make("CartPole-v3"))
show("E10 reset 전에 step", lambda: gym.make("CartPole-v1").step(0))
show("E11 원소 여러 개인 텐서에 .item()", lambda: torch.zeros(2).item())
show("E12 기울기 있는 텐서에 .numpy()", lambda: net(torch.zeros(4)).numpy())
def e13():
    p = nn.Parameter(torch.zeros(2)); p.mul_(0.5)
show("E13 no_grad 없이 파라미터 제자리 수정(Polyak)", e13)
def e14():
    env = gym.make("CartPole-v1"); env.reset(seed=0); env.step(torch.tensor(1))
show("E14 이산 행동을 텐서로 넘김(.item() 누락)", e14)
def e15():
    env = gym.make("CartPole-v1"); env.reset(seed=0); env.step(np.int64(5))
show("E15 이산 행동 범위 밖", e15)
show("E16 log(0) → 손실이 -inf/NaN", lambda: print("   값:", torch.log(torch.tensor(0.0)).item(), "/", (torch.log(torch.tensor(0.0)) * 0).item()))
