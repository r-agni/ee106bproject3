from dataclasses import dataclass, field
import numpy as np
import casadi as ca

from optimization.obstacles import CircularObstacle

@dataclass
class PlannerParams:
    """Parameters for the minimum-time planner."""
    ## TODO: for getting a viable trajectory on the Turtlebot you may need to edit these values or add entirely new parameters
    N: int = 100
    v_min: float = -0.1
    v_max: float = 1.0
    omega_min: float = -2.0
    omega_max: float = 2.0
    max_lin_accel: float = 0.18
    max_ang_accel: float = 1.2
    omega_smooth_weight: float = 0.5
    dt_min: float = 0.01
    dt_max: float = 1.0
    obstacle_buffer: float = 2.0


@dataclass
class TrackingParams:
    """Parameters for the quadratic-cost tracking planner."""
    ## TODO: for getting a viable trajectory on the Turtlebot you may need to edit these values or add entirely new parameters
    N: int = 100
    dt: float = 1.5
    v_min: float = -0.1
    v_max: float = 1.0
    omega_min: float = -2.0
    omega_max: float = 2.0
    max_lin_accel: float = 0.18
    max_ang_accel: float = 1.2
    obstacle_buffer: float = 2.0
    Q: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0, 0.5]))
    R: np.ndarray = field(default_factory=lambda: np.diag([1.0, 0.5]))
    P: np.ndarray = field(default_factory=lambda: np.diag([8.0, 8.0, 4.0]))
    dR: np.ndarray = field(default_factory=lambda: np.diag([8.0, 1.5]))


@dataclass
class PlannerResult:
    success: bool
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    theta: np.ndarray = field(default_factory=lambda: np.array([]))
    v: np.ndarray = field(default_factory=lambda: np.array([]))
    omega: np.ndarray = field(default_factory=lambda: np.array([]))
    dt: float = 0.0
    total_time: float = 0.0
    solver_stats: dict = field(default_factory=dict)


def _add_acceleration_constraints(
    opti: ca.Opti,
    U: ca.MX,
    N: int,
    dt,
    max_lin_accel: float,
    max_ang_accel: float,
) -> None:
    """Bound control-rate changes to match controller-side slew limits."""
    if N <= 0:
        return

    lin_accel = max(0.0, float(max_lin_accel))
    ang_accel = max(0.0, float(max_ang_accel))
    v_step = lin_accel * dt
    omega_step = ang_accel * dt

    # Start from rest and enforce bounded increments thereafter.
    opti.subject_to(opti.bounded(-v_step, U[0, 0], v_step))
    opti.subject_to(opti.bounded(-omega_step, U[1, 0], omega_step))
    for k in range(N - 1):
        dv = U[0, k + 1] - U[0, k]
        domega = U[1, k + 1] - U[1, k]
        opti.subject_to(opti.bounded(-v_step, dv, v_step))
        opti.subject_to(opti.bounded(-omega_step, domega, omega_step))


def _omega_smoothness_cost(U: ca.MX, N: int):
    """Quadratic penalty on omega changes to reduce command chattering."""
    if N <= 0:
        return 0.0

    cost = U[1, 0] * U[1, 0]
    for k in range(N - 1):
        domega = U[1, k + 1] - U[1, k]
        cost += domega * domega
    return cost


# ──────────────────────────────────────────────────────────────────────────────
# Option 1: Minimum-time planner
# ──────────────────────────────────────────────────────────────────────────────

class UnicyclePlanner:
    """Minimum-time trajectory planner for a unicycle robot.

    Decision variables:
        X : (3, N+1) — state trajectory [x; y; theta] at each node
        U : (2, N)   — control inputs [v; omega] at each interval
        T : scalar   — total trajectory time
        dt = T / N   — derived per-step timestep
    """

    def __init__(self, params: PlannerParams | None = None):
        self.params = params or PlannerParams()

    def solve(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        obstacles: list[CircularObstacle] | None = None,
    ) -> PlannerResult:
        p = self.params
        N = p.N
        obstacles = obstacles or []

        opti = ca.Opti()

        ## Decision variables
        X = opti.variable(3, N + 1)  # [x; y; theta] at each node
        U = opti.variable(2, N)      # [v; omega] at each interval
        T = opti.variable()           # total trajectory time
        dt = T / N                    # derived timestep

        ## Objective — minimize total time while smoothing omega changes
        omega_smooth_weight = max(0.0, float(p.omega_smooth_weight))
        omega_smooth_cost = _omega_smoothness_cost(U, N)
        opti.minimize(T + omega_smooth_weight * omega_smooth_cost)

        ## Dynamics constraints — Euler integration of unicycle model
        for k in range(N):
            x_k = X[0, k]
            y_k = X[1, k]
            theta_k = X[2, k]
            v_k = U[0, k]
            omega_k = U[1, k]

            opti.subject_to(X[0, k + 1] == x_k + dt * v_k * ca.cos(theta_k))
            opti.subject_to(X[1, k + 1] == y_k + dt * v_k * ca.sin(theta_k))
            opti.subject_to(X[2, k + 1] == theta_k + dt * omega_k)

        ## Boundary constraints — pin start and goal states
        start_vec = np.array(start, dtype=float)
        goal_vec = np.array(goal, dtype=float)
        opti.subject_to(X[:, 0] == start_vec)
        opti.subject_to(X[:, N] == goal_vec)

        ## Control bounds — bound v and omega
        opti.subject_to(opti.bounded(p.v_min, U[0, :], p.v_max))
        opti.subject_to(opti.bounded(p.omega_min, U[1, :], p.omega_max))
        _add_acceleration_constraints(opti, U, N, dt, p.max_lin_accel, p.max_ang_accel)

        ## Time bounds — bound total time T (Force to be positive)
        opti.subject_to(opti.bounded(N * p.dt_min, T, N * p.dt_max))

        ## Obstacle avoidance — keep all nodes outside each obstacle
        for obs in obstacles:
            clearance_sq = (obs.radius + p.obstacle_buffer) ** 2
            for k in range(N + 1):
                dx = X[0, k] - obs.cx
                dy = X[1, k] - obs.cy
                opti.subject_to(dx * dx + dy * dy >= clearance_sq)

        ## Initial guess for T and trajectory variables
        opti.set_initial(T, N * p.dt_max)
        alpha = np.linspace(0.0, 1.0, N + 1)
        X_init = np.outer(start_vec, (1.0 - alpha)) + np.outer(goal_vec, alpha)
        opti.set_initial(X, X_init)
        U_init = np.zeros((2, N))
        distance = float(np.hypot(goal_vec[0] - start_vec[0], goal_vec[1] - start_vec[1]))
        nominal_v = distance / max(N * p.dt_max, 1e-6)
        U_init[0, :] = np.clip(nominal_v, p.v_min, p.v_max)
        opti.set_initial(U, U_init)

        opti.solver(
            "ipopt",
            {"expand": True},
            {"max_iter": 3000, "print_level": 5},
        )

        try:
            sol = opti.solve()
            T_sol = float(sol.value(T))
            return PlannerResult(
                success=True,
                x=np.array(sol.value(X[0, :])).flatten(),
                y=np.array(sol.value(X[1, :])).flatten(),
                theta=np.array(sol.value(X[2, :])).flatten(),
                v=np.array(sol.value(U[0, :])).flatten(),
                omega=np.array(sol.value(U[1, :])).flatten(),
                dt=T_sol / N,
                total_time=T_sol,
                solver_stats=sol.stats(),
            )
        except RuntimeError as e:
            print(f"Solver failed: {e}")
            debug = opti.debug
            T_dbg = float(debug.value(T))
            return PlannerResult(
                success=False,
                x=np.array(debug.value(X[0, :])).flatten(),
                y=np.array(debug.value(X[1, :])).flatten(),
                theta=np.array(debug.value(X[2, :])).flatten(),
                v=np.array(debug.value(U[0, :])).flatten(),
                omega=np.array(debug.value(U[1, :])).flatten(),
                dt=T_dbg / N,
                total_time=T_dbg,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Option 2: Quadratic tracking-cost planner
# ──────────────────────────────────────────────────────────────────────────────

class UnicycleTrackingPlanner:
    """Fixed-horizon quadratic tracking cost planner.

    Decision variables:
        X : (3, N+1) — state trajectory [x; y; theta] at each node
        U : (2, N)   — control inputs [v; omega] at each interval

    """

    def __init__(self, params: TrackingParams | None = None):
        self.params = params or TrackingParams()

    def solve(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        obstacles: list[CircularObstacle] | None = None,
    ) -> PlannerResult:
        p = self.params
        N = p.N
        dt = p.dt
        obstacles = obstacles or []

        opti = ca.Opti()

        X = opti.variable(3, N + 1)
        U = opti.variable(2, N)

        ## Objective — quadratic tracking cost with terminal penalty
        Q = ca.DM(p.Q)
        R = ca.DM(p.R)
        P = ca.DM(p.P)
        dR = ca.DM(p.dR)
        xf = ca.DM(np.array(goal, dtype=float).reshape(3, 1))

        cost = 0
        for k in range(N):
            e_k = X[:, k] - xf
            u_k = U[:, k]
            cost += ca.mtimes([e_k.T, Q, e_k]) + ca.mtimes([u_k.T, R, u_k])
        for k in range(N - 1):
            du_k = U[:, k + 1] - U[:, k]
            cost += ca.mtimes([du_k.T, dR, du_k])
        e_N = X[:, N] - xf
        cost += ca.mtimes([e_N.T, P, e_N])
        opti.minimize(cost)

        ## Dynamics constraints — Euler integration (dt is fixed here)
        for k in range(N):
            x_k = X[0, k]
            y_k = X[1, k]
            theta_k = X[2, k]
            v_k = U[0, k]
            omega_k = U[1, k]

            opti.subject_to(X[0, k + 1] == x_k + dt * v_k * ca.cos(theta_k))
            opti.subject_to(X[1, k + 1] == y_k + dt * v_k * ca.sin(theta_k))
            opti.subject_to(X[2, k + 1] == theta_k + dt * omega_k)

        ## Boundary constraints — pin start and goal states
        start_vec = np.array(start, dtype=float)
        goal_vec = np.array(goal, dtype=float)
        opti.subject_to(X[:, 0] == start_vec)
        opti.subject_to(X[:, N] == goal_vec)

        ## Control bounds — bound v and omega
        opti.subject_to(opti.bounded(p.v_min, U[0, :], p.v_max))
        opti.subject_to(opti.bounded(p.omega_min, U[1, :], p.omega_max))
        _add_acceleration_constraints(opti, U, N, dt, p.max_lin_accel, p.max_ang_accel)

        ## Obstacle avoidance — keep all nodes outside each obstacle
        for obs in obstacles:
            clearance_sq = (obs.radius + p.obstacle_buffer) ** 2
            for k in range(N + 1):
                dx = X[0, k] - obs.cx
                dy = X[1, k] - obs.cy
                opti.subject_to(dx * dx + dy * dy >= clearance_sq)

        alpha = np.linspace(0.0, 1.0, N + 1)
        X_init = np.outer(start_vec, (1.0 - alpha)) + np.outer(goal_vec, alpha)
        opti.set_initial(X, X_init)
        opti.set_initial(U, np.zeros((2, N)))

        opti.solver(
            "ipopt",
            {"expand": True},
            {"max_iter": 3000, "print_level": 5},
        )

        total_time = N * dt
        try:
            sol = opti.solve()
            return PlannerResult(
                success=True,
                x=np.array(sol.value(X[0, :])).flatten(),
                y=np.array(sol.value(X[1, :])).flatten(),
                theta=np.array(sol.value(X[2, :])).flatten(),
                v=np.array(sol.value(U[0, :])).flatten(),
                omega=np.array(sol.value(U[1, :])).flatten(),
                dt=dt,
                total_time=total_time,
                solver_stats=sol.stats(),
            )
        except RuntimeError as e:
            print(f"Solver failed: {e}")
            debug = opti.debug
            return PlannerResult(
                success=False,
                x=np.array(debug.value(X[0, :])).flatten(),
                y=np.array(debug.value(X[1, :])).flatten(),
                theta=np.array(debug.value(X[2, :])).flatten(),
                v=np.array(debug.value(U[0, :])).flatten(),
                omega=np.array(debug.value(U[1, :])).flatten(),
                dt=dt,
                total_time=total_time,
            )
