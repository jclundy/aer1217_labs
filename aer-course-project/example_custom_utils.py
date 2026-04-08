import numpy as np


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

    def dist(self, other):
        return np.hypot(self.x - other.x, self.y - other.y)


class RRTStar:
    def __init__(self, obstacles, bounds, step=0.3, r_rewire=0.7, goal_bias=0.15, goal_tol=0.15):
        """
        obstacles : list of (cx, cy, radius)
        bounds    : (x_min, x_max, y_min, y_max)
        """
        self.obstacles = obstacles
        self.x_min, self.x_max, self.y_min, self.y_max = bounds
        self.step = step
        self.r_rewire = r_rewire
        self.goal_bias = goal_bias
        self.goal_tol = goal_tol
        self.nodes = []

    def plan(self, start, goal, max_iter=3000):
        root = Node(*start)
        self.nodes = [root]
        goal_node = Node(*goal)

        for _ in range(max_iter):
            sample = goal_node if np.random.rand() < self.goal_bias else Node(
                np.random.uniform(self.x_min, self.x_max),
                np.random.uniform(self.y_min, self.y_max)
            )
            nearest_idx = min(range(len(self.nodes)), key=lambda i: self.nodes[i].dist(sample))
            nearest = self.nodes[nearest_idx]
            new = self._steer(nearest, sample)

            if not self._edge_free(nearest, new):
                continue

            neighbors = [i for i, n in enumerate(self.nodes) if n.dist(new) <= self.r_rewire]
            best_idx, best_cost = self._best_parent(new, neighbors, nearest_idx)
            new.parent = best_idx
            new.cost = best_cost
            self.nodes.append(new)
            new_idx = len(self.nodes) - 1
            self._rewire(new, new_idx, neighbors)

            if new.dist(goal_node) <= self.goal_tol and self._edge_free(new, goal_node):
                goal_node.parent = new_idx
                goal_node.cost = new.cost + new.dist(goal_node)
                self.nodes.append(goal_node)
                return self._trace()

        print("[RRTStar] max iterations reached, returning best partial path")
        closest = min(range(len(self.nodes)), key=lambda i: self.nodes[i].dist(goal_node))
        self.nodes.append(Node(*goal))
        self.nodes[-1].parent = closest
        return self._trace()

    def _steer(self, frm, to):
        d = frm.dist(to)
        if d <= self.step:
            return Node(to.x, to.y)
        r = self.step / d
        return Node(frm.x + r * (to.x - frm.x), frm.y + r * (to.y - frm.y))

    def _edge_free(self, a, b):
        return not any(self._line_hits(a.x, a.y, b.x, b.y, cx, cy, r)
                       for cx, cy, r in self.obstacles)

    def _line_hits(self, x1, y1, x2, y2, cx, cy, r):
        dx, dy = x2 - x1, y2 - y1
        l = np.hypot(dx, dy)
        if l == 0:
            return np.hypot(x1 - cx, y1 - cy) <= r
        t = np.clip(((cx - x1) * dx + (cy - y1) * dy) / l**2, 0, 1)
        return np.hypot(x1 + t*dx - cx, y1 + t*dy - cy) <= r

    def _best_parent(self, node, neighbors, fallback):
        best_idx = fallback
        best_cost = self.nodes[fallback].cost + self.nodes[fallback].dist(node)
        for i in neighbors:
            c = self.nodes[i].cost + self.nodes[i].dist(node)
            if c < best_cost and self._edge_free(self.nodes[i], node):
                best_idx, best_cost = i, c
        return best_idx, best_cost

    def _rewire(self, node, node_idx, neighbors):
        for i in neighbors:
            n = self.nodes[i]
            c = node.cost + node.dist(n)
            if c < n.cost and self._edge_free(node, n):
                n.parent, n.cost = node_idx, c

    def _trace(self):
        path, idx = [], len(self.nodes) - 1
        while idx is not None:
            n = self.nodes[idx]
            path.append([n.x, n.y])
            idx = n.parent
        return path[::-1]