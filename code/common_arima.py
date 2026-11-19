"""1교시 6-8 · 12-14: ARIMA를 손으로(AR 최소제곱) 그리고 statsmodels로
실행: python common_arima.py
"""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from common_drills import fit_ar, forecast_ar

rng = np.random.default_rng(0)

# ① AR(2) 시계열 만들기:  y_t = 1.0 + 0.6·y_(t−1) − 0.2·y_(t−2) + ε_t
n, c, phi = 2000, 1.0, np.array([0.6, -0.2])
y = np.zeros(n)
for t in range(2, n):
    y[t] = c + phi[0] * y[t - 1] + phi[1] * y[t - 2] + rng.normal()

# ② 손으로: 최소제곱 AR(2)
c_hat, phi_hat = fit_ar(y, 2)
print(f"손으로 AR(2): c = {c_hat:.3f}, φ = {np.round(phi_hat, 3)}   (참값 c = 1.0, φ = [0.6, -0.2])")

# ③ statsmodels: 같은 AR(2) = ARIMA(2, 0, 0)
res = ARIMA(y, order=(2, 0, 0)).fit()
mu = res.params[0]                                  # 주의: ARIMA의 const는 절편 c가 아니라 평균 μ (AutoReg는 절편)
print(f"statsmodels: 평균 μ = {mu:.3f} → 절편 c = μ(1 − φ₁ − φ₂) = {mu * (1 - res.params[1] - res.params[2]):.3f}, "
      f"φ = {np.round(res.params[1:3], 3)}")
print("다음 3스텝 예측 — 손으로:", np.round(forecast_ar(y, c_hat, phi_hat, 3), 3),
      "/ statsmodels:", np.round(res.forecast(3), 3))

# ④ 계절성 있는 월별 시계열: 추세 + 12개월 주기 + 잡음 → SARIMA
months = np.arange(144)
sales = 100 + 0.5 * months + 10 * np.sin(2 * np.pi * months / 12) + rng.normal(0, 2, 144)
print(f"ADF p값 — 원래 {adfuller(sales)[1]:.3f}, 1차+계절 차분 후 {adfuller(np.diff(sales[12:] - sales[:-12]))[1]:.3f}")
y_train, y_test = sales[:132], sales[132:]
model = ARIMA(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
pred = model.forecast(steps=12)
print(f"SARIMA(1,1,1)(1,1,1,12) 12개월 예측 MAE = {np.mean(np.abs(pred - y_test)):.2f}  (잡음 표준편차 2)")
