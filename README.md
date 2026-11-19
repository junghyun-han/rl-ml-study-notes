# 머신러닝 · 딥러닝 · 강화학습 공부 노트

처음 보는 사람도 읽을 수 있게 쉬운 말로 정리한 노트와, 모두 실행해 확인한 코드.

| 파일 | 내용 |
|---|---|
| `00_쉬운설명본_ML_DL_RL.md` | 개념 55개 쉬운 설명(한 줄 → 쉽게 풀면 → 용어·식 → 시험 포인트), 서술형 코드 22개, 주관식 22문제, 계산 20문제, 자가점검 78문항 |
| `01_ML_DL_공부노트.md` | 선형 회귀·규제·트리·XGBoost·평가 지표·ARIMA·DNN/CNN/RNN·최적화 + 주관식 코드 |
| `02_강화학습_PhysicalAI_이론노트.md` | MDP·벨만·TD·DQN·PPO·SAC·보상 설계·안정성 해석·RRT·최신 로봇 학습 논문 |
| `03_강화학습_실습노트.md` | 실습 코드 전부와 줄별 설명, 실제 학습 곡선 해석 |
| `04_연습문제.md` | 객관식·주관식 연습문제와 해설 |
| `05_RL실험_상황별_대처표.md` | 강화학습 실험에서 언제 무엇을 할지 |
| `06_RL실험_진행순서_연습문제.md` | 실험 진행 순서(그대로 따라 하기)와 실습 연습문제 전수 목록 |
| `07_RL실험_1쪽_요약.md` | 1쪽 요약 |
| `08_RL실험_답안_후보_전수.md` | 실습 답안 후보 전부: 모의 실습 6개 완전 풀이(CartPole PPO, Pendulum SAC, MuJoCo InvertedPendulum SAC, 보상 설계, RRT, 로그 해석) + 코드 후보 + 설명 문장 후보 |
| `code/` | 실행 코드 23개 + 그림 생성 `make_figures.py` (`*_check.py` = 검증 스크립트) |
| `code/results/` | 실험 결과 원본(학습 곡선 기록, 로그) — 노트의 숫자·그림의 근거 |
| `fig/` | 노트 그림 11장 |

## 실행
```bash
pip install torch "gymnasium[classic-control,toy-text]" numpy matplotlib statsmodels scikit-learn
cd code
python 05_ppo_discrete.py          # 예: CartPole PPO
python mock_ppo.py --env CartPole-v1 --steps 60000 --seed 0        # 모의 실습 M1
python mock_sac.py --env InvertedPendulum-v5 --steps 30000 --seed 0  # M3 (pip install "gymnasium[mujoco]")
python common_drills_check.py      # 1교시 코드 검증
python easy_snippets_check.py      # 쉬운 설명본 코드 검증
python extra_snippets_check.py     # 실습 추가 코드 검증
```

## 노트 안에서 부르는 이름 ↔ 파일
| 노트 안의 이름 | 파일 |
|---|---|
| 쉬운 설명본 (E1~E10, R1~R12 서술형 코드) | `00_쉬운설명본_ML_DL_RL.md` |
| 1교시 노트 (예: 12-3) | `01_ML_DL_공부노트.md` |
| 2교시 노트 | `02_강화학습_PhysicalAI_이론노트.md` |
| 3교시 실습노트 (예: 16-1 뼈대) | `03_강화학습_실습노트.md` |
| 예상문제 | `04_연습문제.md` |
| 대처표 | `05_RL실험_상황별_대처표.md` |
| 진행 순서·예상 문제 전수 (X1~X7, W1~W5) | `06_RL실험_진행순서_연습문제.md` |
| 답안 후보 전수 (M1~M6) | `08_RL실험_답안_후보_전수.md` |
| 코드 이름 `00`~`12` | `code/00_*.py` ~ `code/12_*.py` |

## 실험 다시 돌리고 그림 다시 그리기
```bash
cd code
python 02_dqn_cartpole.py --seed 0        # 결과는 results/에 저장
python make_figures.py                    # ../fig/ 에 그림 11장
```
