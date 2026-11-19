"""1교시 12-15: PyTorch 학습 루프 (데이터 만들기 · DataLoader · 학습 · 검증 · 조기 종료까지 단독 실행)
실행: python common_torch_loop.py
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

# ① 예시 데이터: 특성 10개, 클래스 3개 (실전에서는 numpy 배열을 torch.tensor로 바꿔 쓰면 됨)
X = torch.randn(1200, 10)
y = (X[:, 0] + X[:, 1] > 0).long() + (X[:, 2] > 1).long()      # 정답 0, 1, 2 (정수, long)
X_tr, y_tr, X_va, y_va = X[:1000], y[:1000], X[1000:], y[1000:]

# ② DataLoader: 미니배치로 잘라 주고, 학습용은 매 epoch마다 섞음
train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=256)

# ③ 모델 · 손실 · 옵티마이저
d, n_cls, epochs = 10, 3, 30
model = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_cls))
loss_fn = nn.CrossEntropyLoss()                 # softmax를 안에 포함 → 모델 끝에 softmax 넣지 않음
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

best_val, best_state, patience, bad = float("inf"), None, 5, 0
for epoch in range(epochs):
    # ④ 학습
    model.train()                               # Dropout·BatchNorm 학습 모드
    for xb, yb in train_loader:
        opt.zero_grad()                         # 기울기 초기화 (안 하면 누적됨)
        loss = loss_fn(model(xb), yb)           # 순전파 + 손실
        loss.backward()                         # 역전파
        opt.step()                              # 가중치 업데이트

    # ⑤ 검증
    model.eval()                                # 평가 모드 (Dropout 끔)
    total, correct, val_loss = 0, 0, 0.0
    with torch.no_grad():                       # 기울기 계산 끔
        for xb, yb in val_loader:
            out = model(xb)
            val_loss += loss_fn(out, yb).item() * len(yb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)
    val_loss /= total

    # ⑥ 조기 종료: 검증 손실이 patience번 연속 안 좋아지면 멈추고 가장 좋던 가중치로
    if val_loss < best_val:
        best_val, bad = val_loss, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= patience:
            break

model.load_state_dict(best_state)
print(f"멈춘 epoch {epoch + 1}, 최저 검증 손실 {best_val:.4f}, 마지막 검증 정확도 {correct / total:.3f}")
