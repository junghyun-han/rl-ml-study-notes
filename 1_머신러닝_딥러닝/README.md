# 1. 머신러닝 · 딥러닝

선형 회귀·규제, 트리·앙상블·XGBoost, 평가 지표, 시계열(ARIMA), DNN·CNN·RNN, 활성화·Dropout·최적화, PyTorch 학습 루프.

## 보는 순서
| 순서 | 파일 | 언제 보나 |
|---|---|---|
| 1 | [`00_쉬운설명본_ML_DL.md`](00_쉬운설명본_ML_DL.md) | 처음 보는 사람용 쉬운 설명: 개념(한 줄 → 쉽게 풀면 → 용어·식 → 시험 포인트), 서술형 코드 E1~E10, 주관식 Q1~Q10, 계산 문제 14개, 가리고 답하기 1~36, 헷갈리는 짝 |
| 2 | [`01_ML_DL_공부노트.md`](01_ML_DL_공부노트.md) | 더 자세한 공부노트: 식 유도, numpy·PyTorch 코드 전체, 참고 자료 지도, 용어 정리 |
| 3 | [`04_연습문제_ML_DL.md`](04_연습문제_ML_DL.md) | 외울 공식 한 장, 객관식 1-1~1-31, 주관식 1-S1~1-S13 |

## 코드 (`../code/`)
| 코드 | 내용 | 검증 |
|---|---|---|
| `common_drills.py` | numpy로 직접 짜는 지표·회귀·로지스틱·softmax·Adam·지니·K-fold·ACF·역전파 | `common_drills_check.py` (19개, sklearn·torch·statsmodels와 대조) |
| `common_arima.py` | ARIMA 절차 | 위와 같음 |
| `common_torch_loop.py` | PyTorch 학습 루프 | 위와 같음 |
| `easy_snippets.py` (E1~E10) | 서술형 대비 짧은 코드 | `easy_snippets_check.py` |

```bash
pip install torch numpy matplotlib statsmodels scikit-learn
cd ../code && python common_drills_check.py && python easy_snippets_check.py
```
