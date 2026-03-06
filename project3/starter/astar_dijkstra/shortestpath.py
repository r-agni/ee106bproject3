import numpy as np
import heapq
from utils.map_utils import worldtocell, celltoworld, celltonumber, numbertocell


def shortestpath(map_data, start, goal):
    """Find the shortest path from start to goal using Dijkstra's algorithm.

    Parameters
    ----------
    map_data : dict
        Map data returned by load_map().
    start : np.ndarray, shape (2,)
        Start position in world coordinates [x, y].
    goal : np.ndarray, shape (2,)
        Goal position in world coordinates [x, y].

    Returns
    -------
    np.ndarray, shape (M, 2)
        Path from start to goal. Each row is [x, y]. Returns empty array if
        no path is found.
    """
    ## DO NOT MODIFY
    nodenumber = map_data['nodenumber']
    leftbound = map_data['boundary'][:2]
    blockflag = map_data['blockflag']
    resolution = map_data['resolution']
    xy_res = resolution[0]
    segment = map_data['segment']
    mx, my = int(segment[0]), int(segment[1])
    num_nodes = len(nodenumber)

    start_cell = worldtocell(leftbound, resolution, start)
    goal_cell = worldtocell(leftbound, resolution, goal)

    def in_bounds(cell):
        return 0 <= int(cell[0]) < mx and 0 <= int(cell[1]) < my

    if not in_bounds(start_cell) or not in_bounds(goal_cell):
        return np.array([]).reshape(0, 2)

    start_node = celltonumber(segment, start_cell)
    goal_node = celltonumber(segment, goal_cell)

    if blockflag[start_node] == 1 or blockflag[goal_node] == 1:
        return np.array([]).reshape(0, 2)

    dist = np.full(num_nodes, np.inf)
    visited = np.zeros(num_nodes, dtype=bool)
    predecessor = np.full(num_nodes, -1, dtype=int)

    dist[start_node] = 0.0
    pq = [(0.0, int(start_node))]  # (g, node)
    step_cost = float(xy_res)

    while pq:
        g_curr, curr = heapq.heappop(pq)
        if visited[curr]:
            continue

        visited[curr] = True
        if curr == goal_node:
            break

        cell = numbertocell(segment, curr)
        cx, cy = int(cell[0]), int(cell[1])

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < mx and 0 <= ny < my):
                continue

            nbr = celltonumber(segment, np.array([nx, ny]))
            if visited[nbr] or blockflag[nbr] == 1:
                continue

            cand = g_curr + step_cost
            if cand < dist[nbr]:
                dist[nbr] = cand
                predecessor[nbr] = curr
                heapq.heappush(pq, (cand, int(nbr)))

    if not visited[goal_node]:
        return np.array([]).reshape(0, 2)

    path_nodes = []
    node = int(goal_node)
    while node != -1:
        path_nodes.append(node)
        if node == start_node:
            break
        node = int(predecessor[node])

    if not path_nodes or path_nodes[-1] != start_node:
        return np.array([]).reshape(0, 2)

    path_nodes.reverse()
    path_world = []
    for n in path_nodes:
        cell = numbertocell(segment, n)
        path_world.append(celltoworld(leftbound, resolution, cell))

    path = np.array(path_world, dtype=float).reshape(-1, 2)
    return path
