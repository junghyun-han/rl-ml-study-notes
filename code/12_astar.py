"""12. A* 격자 경로 탐색 (+ 휴리스틱 0 = Dijkstra와 비교) — 표준 라이브러리 heapq만 사용
실행: python 12_astar.py
"""
import heapq

GRID = [
    "S....#....",
    ".##..#.##.",
    "..#....#..",
    "..#.####..",
    "..#....#..",
    "..####.#..",
    ".......#.G",
]
ROWS, COLS = len(GRID), len(GRID[0])


def find(ch):
    for r, row in enumerate(GRID):
        if ch in row:
            return (r, row.index(ch))


def neighbors(node):
    r, c = node
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:                  # 상하좌우 4방향
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and GRID[nr][nc] != "#":
            yield (nr, nc)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])        # 4방향 이동에서 실제 비용보다 크지 않음(admissible)


def astar(start, goal, h):
    open_heap = [(h(start, goal), 0, start)]           # (f = g + h, g, 노드) — f가 작은 것부터 꺼냄
    g = {start: 0}                                      # 출발점에서 각 노드까지 지금까지 찾은 최소 비용
    parent = {start: None}
    closed = set()                                      # 확정된(다시 볼 필요 없는) 노드
    expanded = 0
    while open_heap:
        f, g_cur, node = heapq.heappop(open_heap)
        if node in closed:                              # 더 비싼 옛 기록이면 건너뜀
            continue
        closed.add(node)
        expanded += 1
        if node == goal:
            break
        for nb in neighbors(node):
            ng = g_cur + 1                              # 한 칸 이동 비용 1
            if ng < g.get(nb, float("inf")):            # 더 싼 길을 찾았으면 갱신
                g[nb] = ng
                parent[nb] = node
                heapq.heappush(open_heap, (ng + h(nb, goal), ng, nb))
    if goal not in parent:
        return None, expanded
    path, node = [], goal
    while node is not None:                             # 부모를 따라 거꾸로 올라감
        path.append(node)
        node = parent[node]
    return path[::-1], expanded


if __name__ == "__main__":
    S, G = find("S"), find("G")
    path_a, exp_a = astar(S, G, manhattan)
    path_d, exp_d = astar(S, G, lambda a, b: 0)        # h = 0 → Dijkstra
    print(f"A*      : 경로 {len(path_a) - 1}칸, 펼친 노드 {exp_a}개")
    print(f"Dijkstra: 경로 {len(path_d) - 1}칸, 펼친 노드 {exp_d}개")
    grid = [list(row) for row in GRID]
    for (r, c) in path_a[1:-1]:
        grid[r][c] = "*"
    print("\n".join("".join(row) for row in grid))
