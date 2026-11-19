# 3. 강화학습 실습 전수

강화학습 실습(PPO·SAC·DQN을 Gym·MuJoCo 환경에서 돌려 성능 높이기, 보상 설계, 경로 탐색)에 쓸 수 있는 것을 전부 모은 폴더. 코드는 저장소의 `../code/`에 있고 모두 실행해 확인했다.

## 보는 순서
| 순서 | 파일 | 언제 보나 |
|---|---|---|
| 1 | [`07_RL실험_1쪽_요약.md`](07_RL실험_1쪽_요약.md) | 전체 흐름을 한 장으로 |
| 2 | [`11_RL실험_상황별_행동_코드집.md`](11_RL실험_상황별_행동_코드집.md) | **문제가 이렇게 나오면 무엇을 어떤 순서로, 어떤 코드로 하나** (PPO·Gym, PPO·MuJoCo, SAC, 안 배울 때, 무너질 때, 시드 차이, 곡선, 비교, 보상, DQN, 시간 부족) |
| 3 | [`12_RL실험_코드_하나씩.md`](12_RL실험_코드_하나씩.md) | **코드를 한 파일씩 통째로** 보고 그대로 칠 때: 코드 19개 + 외워 쓰는 뼈대 8개, 코드마다 언제 쓰나·실행·실제 결과 |
| 4 | [`08_RL실험_답안_후보_전수.md`](08_RL실험_답안_후보_전수.md) | 답안으로 쓸 것 전부: 모의 실습 6개 완전 풀이(실제 출력), 보고 틀, 코드 후보, 설명 문장 후보 |
| 5 | [`06_RL실험_진행순서_연습문제.md`](06_RL실험_진행순서_연습문제.md) | 실험 진행 순서와 예상 문제 전수 목록 |
| 6 | [`05_RL실험_상황별_대처표.md`](05_RL실험_상황별_대처표.md) | 증상별 대처, 알고리즘 고르기, 120분 진행 계획 |
| 7 | [`09_RL실험_에러메시지_대처표.md`](09_RL실험_에러메시지_대처표.md) | 오류 메시지가 떴을 때(실제 메시지 16개 → 원인 → 고치기) |
| 8 | [`10_RL실험_하이퍼파라미터_체크리스트.md`](10_RL실험_하이퍼파라미터_체크리스트.md) | 어떤 값을 시도할지(출처 표시), 제출 전 체크리스트 |
| 9 | [`03_강화학습_실습노트.md`](03_강화학습_실습노트.md) | 실습 코드 13개 전체와 줄별 설명, 실제 학습 곡선 |

## 이 폴더에서 쓰는 코드 (`../code/`)
| 코드 | 내용 |
|---|---|
| `mock_ppo.py` | 이산 행동 PPO 한 파일 풀이 (`--env --steps --seed --lr --ent --save`) |
| `mock_sac.py` | 연속 행동 SAC 한 파일 풀이, MuJoCo도 환경 이름만 바꿔 사용 (`--env --steps --seed --save`) |
| `06_ppo_continuous.py` | 연속 행동 PPO (MuJoCo는 `gym.make` 두 곳의 환경 이름을 바꿈) |
| `action_toolbox.py` | 상황별 도구 T1~T7: 정규화, 평가 환경, 학습률 조절, KL 조기 종료, 모델 저장, 시드 여러 개, 곡선 (`action_toolbox_check.py`로 확인) |
| `exam_workflow.py` | 진행 도구 W1~W5: 환경 확인, 평가, 무작위 기준선, 요약, 기록표 |
| `easy_snippets.py` / `extra_snippets.py` | 구현 문제 코드 R1~R12 / X1~X7 |
| `error_messages_check.py` | 에러 메시지 대처표의 오류를 일부러 내는 코드 |
| `00`~`12` | 실습노트의 전체 코드 |

## 실행
```bash
pip install torch "gymnasium[classic-control,toy-text,mujoco]" numpy matplotlib
cd ../code
python mock_ppo.py --env CartPole-v1 --steps 60000 --seed 0
python mock_sac.py --env InvertedPendulum-v5 --steps 30000 --seed 0
python action_toolbox_check.py
```
