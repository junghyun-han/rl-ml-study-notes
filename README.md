# 머신러닝 · 딥러닝 · 강화학습 공부 노트

처음 보는 사람도 읽을 수 있게 쉬운 말로 정리한 노트와, 모두 실행해 확인한 코드.

## 폴더
| 폴더 | 내용 |
|---|---|
| [`1_머신러닝_딥러닝/`](1_머신러닝_딥러닝/) | 선형 회귀·규제, 트리·XGBoost, 평가 지표, ARIMA, DNN·CNN·RNN, 최적화, PyTorch — 쉬운 설명본, 공부노트, 연습문제 |
| [`2_강화학습_이론/`](2_강화학습_이론/) | MDP·벨만, TD, DQN, PPO·GAE, SAC, 보상 설계, 안정성 해석, RRT·A\*, 최신 로봇 학습 논문 — 쉬운 설명본, 이론노트, 연습문제 |
| [`3_강화학습_실습_전수/`](3_강화학습_실습_전수/) | PPO·SAC·DQN을 Gym·MuJoCo에서 돌려 성능 높이기, 보상 설계, 경로 탐색 — 상황별 행동, 코드 하나씩, 답안 후보, 에러 대처 등 |
| `code/` | 실행 코드 전부 (`*_check.py` = 검증 스크립트), `code/results/` = 실험 결과 원본(노트 숫자·그림의 근거) |
| `fig/` | 노트 그림 11장 (`code/make_figures.py`로 다시 그림) |

각 폴더의 README에 보는 순서가 있다.

## 실행
```bash
pip install torch "gymnasium[classic-control,toy-text,mujoco]" numpy matplotlib statsmodels scikit-learn
cd code
python mock_ppo.py --env CartPole-v1 --steps 60000 --seed 0          # 이산 PPO
python mock_sac.py --env InvertedPendulum-v5 --steps 30000 --seed 0  # 연속 SAC (MuJoCo)
python common_drills_check.py      # 머신러닝·딥러닝 코드 검증
python easy_snippets_check.py      # 쉬운 설명본 코드 검증
python extra_snippets_check.py     # 실습 추가 코드 검증
python action_toolbox_check.py     # 상황별 도구 T1~T7 검증
```

## 노트 안에서 부르는 이름 ↔ 파일
| 노트 안의 이름 | 파일 |
|---|---|
| 쉬운 설명본 (E1~E10, Q1~Q10) | `1_머신러닝_딥러닝/00_쉬운설명본_ML_DL.md` |
| 쉬운 설명본 (R1~R12, Q11~Q22) | `2_강화학습_이론/00_쉬운설명본_강화학습.md` |
| 1교시 노트 (예: 12-3) | `1_머신러닝_딥러닝/01_ML_DL_공부노트.md` |
| 2교시 노트 | `2_강화학습_이론/02_강화학습_PhysicalAI_이론노트.md` |
| 예상문제 1-x, 1-S | `1_머신러닝_딥러닝/04_연습문제_ML_DL.md` |
| 예상문제 2-x, 2-S | `2_강화학습_이론/04_연습문제_강화학습.md` |
| 예상문제 P1~P10 | `3_강화학습_실습_전수/04_연습문제_실습과제.md` |
| 3교시 실습노트 (예: 16-1 뼈대) | `3_강화학습_실습_전수/03_강화학습_실습노트.md` |
| 대처표 | `3_강화학습_실습_전수/05_RL실험_상황별_대처표.md` |
| 진행 순서·예상 문제 전수 (X1~X7, W1~W5) | `3_강화학습_실습_전수/06_RL실험_진행순서_연습문제.md` |
| 1쪽 요약 | `3_강화학습_실습_전수/07_RL실험_1쪽_요약.md` |
| 답안 후보 전수 (M1~M6) | `3_강화학습_실습_전수/08_RL실험_답안_후보_전수.md` |
| 에러 메시지 대처표 (E01~E16) | `3_강화학습_실습_전수/09_RL실험_에러메시지_대처표.md` |
| 하이퍼파라미터·체크리스트 | `3_강화학습_실습_전수/10_RL실험_하이퍼파라미터_체크리스트.md` |
| 상황별 행동 코드집 (T1~T7) | `3_강화학습_실습_전수/11_RL실험_상황별_행동_코드집.md` |
| 코드 하나씩 (C1~C20) | `3_강화학습_실습_전수/12_RL실험_코드_하나씩.md` |
| 코드 이름 `00`~`12` | `code/00_*.py` ~ `code/12_*.py` |

## 실험 다시 돌리고 그림 다시 그리기
```bash
cd code
python 02_dqn_cartpole.py --seed 0        # 결과는 results/에 저장
python make_figures.py                    # ../fig/ 에 그림 11장
```
