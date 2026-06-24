import copy

DIRS = [(1,0),(-1,0),(0,1),(0,-1)]


def solve_flow(board, endpoints):
    rows, cols = len(board), len(board[0])
    colors = list(endpoints.keys())

    # grid tracks occupancy
    grid = [[None for _ in range(cols)] for _ in range(rows)]

    # place endpoints
    for color, pts in endpoints.items():
        for r, c in pts:
            grid[r][c] = color

    paths = {color: [] for color in colors}


    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols


    def is_free(r, c, color):
        return grid[r][c] in [None, color]


    # heuristic: solve hardest first (furthest endpoints)
    def sort_colors():
        def dist(color):
            (r1, c1), (r2, c2) = endpoints[color]
            return abs(r1 - r2) + abs(c1 - c2)
        return sorted(colors, key=dist, reverse=True)


    ordered_colors = sort_colors()


    def dfs_color(i):
        if i == len(ordered_colors):
            # success only if grid is fully filled or valid
            return True

        color = ordered_colors[i]
        start, end = endpoints[color]

        visited = set()

        def dfs(r, c, path):
            if (r, c) == end:
                paths[color] = path[:]
                return dfs_color(i + 1)

            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc

                if not in_bounds(nr, nc):
                    continue
                if (nr, nc) in visited:
                    continue
                if not is_free(nr, nc, color):
                    continue

                visited.add((nr, nc))
                path.append((nr, nc))

                prev = grid[nr][nc]
                grid[nr][nc] = color

                if dfs(nr, nc, path):
                    return True

                grid[nr][nc] = prev
                path.pop()
                visited.remove((nr, nc))

            return False

        sr, sc = start
        visited.add((sr, sc))
        return dfs(sr, sc, [start])


    success = dfs_color(0)

    if not success:
        print("Warning: no full solution found (puzzle may be unsolvable or ordering issue)")

    return paths