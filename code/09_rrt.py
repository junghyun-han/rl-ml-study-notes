"""09. RRT와 RRT* — 2D 평면 경로 탐색 (numpy만 사용)
RRT : 무작위 점을 찍고 → 가장 가까운 노드에서 → 그쪽으로 한 걸음(step) → 충돌 없으면 트리에 추가
RRT*: + 주변 반경 안에서 비용이 가장 싼 부모 고르기 + 주변 노드 다시 잇기(rewire) → 경로가 점점 짧아짐
실행: python 09_rrt.py        (그림 저장: results/rrt.png)
"""
import numpy as np

W, H = 10.0, 10.0
OBSTACLES = [(2, 0, 1, 6), (5, 4, 1, 6), (7.5, 0, 1, 6)]   # 사각형 (x, y, 너비, 높이)
START, GOAL = np.array([0.5, 0.5]), np.array([9.5, 9.5])


def point_free(p, margin=0.1):
    if not (0 <= p[0] <= W and 0 <= p[1] <= H):
        return False
    for (x, y, w, h) in OBSTACLES:
        if x - margin <= p[0] <= x + w + margin and y - margin <= p[1] <= y + h + margin:
            return False
    return True


def edge_free(p, q, res=0.05):
    """두 점 사이 선분을 잘게 나눠 모든 점이 장애물 밖인지 확인"""
    n = max(2, int(np.ceil(np.linalg.norm(q - p) / res)))
    return all(point_free(p + (q - p) * t) for t in np.linspace(0, 1, n))


def path_from(nodes, parent, i):
    path = []
    while i != -1:
        path.append(nodes[i])
        i = parent[i]
    return np.array(path[::-1])


def path_length(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def rrt(step=0.5, goal_bias=0.1, goal_tol=0.5, max_iter=5000, seed=0):
    rng = np.random.default_rng(seed)
    nodes, parent = [START.copy()], [-1]
    for it in range(max_iter):
        # ① 샘플: 가끔(goal_bias)은 목표점 자체를 찍어서 목표 쪽으로 끌어당김
        x_rand = GOAL.copy() if rng.random() < goal_bias else rng.uniform([0, 0], [W, H])
        # ② 가장 가까운 노드
        arr = np.array(nodes)
        i_near = int(np.argmin(np.linalg.norm(arr - x_rand, axis=1)))
        x_near = arr[i_near]
        # ③ 그쪽으로 최대 step만큼 이동 (steer)
        d = x_rand - x_near
        dlen = np.linalg.norm(d)
        if dlen < 1e-9:
            continue
        x_new = x_near + d / dlen * min(step, dlen)
        # ④ 충돌 검사 후 추가
        if not edge_free(x_near, x_new):
            continue
        nodes.append(x_new)
        parent.append(i_near)
        # ⑤ 목표에 충분히 가까우면 연결하고 끝
        if np.linalg.norm(x_new - GOAL) <= goal_tol and edge_free(x_new, GOAL):
            nodes.append(GOAL.copy())
            parent.append(len(nodes) - 2)
            return path_from(nodes, parent, len(nodes) - 1), nodes, parent, it + 1
    return None, nodes, parent, max_iter


def rrt_star(step=0.5, goal_bias=0.1, goal_tol=0.5, radius=1.2, max_iter=3000, seed=0):
    rng = np.random.default_rng(seed)
    nodes, parent, cost = [START.copy()], [-1], [0.0]
    goal_idx = None
    for it in range(max_iter):
        x_rand = GOAL.copy() if rng.random() < goal_bias else rng.uniform([0, 0], [W, H])
        arr = np.array(nodes)
        dists = np.linalg.norm(arr - x_rand, axis=1)
        i_near = int(np.argmin(dists))
        d = x_rand - arr[i_near]
        dlen = np.linalg.norm(d)
        if dlen < 1e-9:
            continue
        x_new = arr[i_near] + d / dlen * min(step, dlen)
        if not edge_free(arr[i_near], x_new):
            continue
        # ① 부모 고르기: 반경 안 이웃 중 (그 노드까지 비용 + 거리)가 가장 작은 것
        near = np.flatnonzero(np.linalg.norm(arr - x_new, axis=1) <= radius)
        best_p, best_c = i_near, cost[i_near] + np.linalg.norm(x_new - arr[i_near])
        for j in near:
            c = cost[j] + np.linalg.norm(x_new - arr[j])
            if c < best_c and edge_free(arr[j], x_new):
                best_p, best_c = int(j), c
        nodes.append(x_new); parent.append(best_p); cost.append(best_c)
        new_i = len(nodes) - 1
        # ② 다시 잇기(rewire): 이웃이 x_new를 거쳐 가는 게 더 싸면 부모를 x_new로 바꿈
        for j in near:
            c = best_c + np.linalg.norm(arr[j] - x_new)
            if c < cost[j] and edge_free(x_new, arr[j]):
                parent[j] = new_i
                cost[j] = c            # (간단 버전: 자손 비용은 즉시 갱신하지 않음)
        # ③ 목표 연결: 찾은 뒤에도 멈추지 않고 계속 → 더 짧은 길로 개선
        if np.linalg.norm(x_new - GOAL) <= goal_tol and edge_free(x_new, GOAL):
            c_goal = best_c + np.linalg.norm(GOAL - x_new)
            if goal_idx is None or c_goal < cost[goal_idx]:
                if goal_idx is None:
                    nodes.append(GOAL.copy()); parent.append(new_i); cost.append(c_goal)
                    goal_idx = len(nodes) - 1
                else:
                    parent[goal_idx], cost[goal_idx] = new_i, c_goal
    path = path_from(nodes, parent, goal_idx) if goal_idx is not None else None
    return path, nodes, parent


if __name__ == "__main__":
    p1, n1, par1, iters = rrt()
    p2, n2, par2 = rrt_star()
    print(f"RRT : {iters}번 반복, 노드 {len(n1)}개, 경로 길이 {path_length(p1):.2f}")
    print(f"RRT*: 3000번 반복, 노드 {len(n2)}개, 경로 길이 {path_length(p2):.2f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        plt.rcParams["font.family"] = "AppleGothic"
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        for ax, (path, nodes, parent, title) in zip(axes, [(p1, n1, par1, "RRT"), (p2, n2, par2, "RRT*")]):
            for (x, y, w, h) in OBSTACLES:
                ax.add_patch(Rectangle((x, y), w, h, color="#888"))
            for i, pi in enumerate(parent):
                if pi >= 0:
                    ax.plot([nodes[i][0], nodes[pi][0]], [nodes[i][1], nodes[pi][1]], color="#9cc3e6", lw=0.6)
            ax.plot(path[:, 0], path[:, 1], color="#d62728", lw=2.2)
            ax.plot(*START, "go", ms=8); ax.plot(*GOAL, "r*", ms=12)
            ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
            ax.set_title(f"{title}: 노드 {len(nodes)}개, 경로 {path_length(path):.2f}")
        plt.tight_layout(); plt.savefig("results/rrt.png", dpi=130)
        print("그림 저장: results/rrt.png")
    except ImportError:
        pass
