#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist
import time
import threading
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field

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

class TurtleBotController(Node):
    def __init__(self, trajectory_filename=None):
        super().__init__('turtlebot_controller')

        self.declare_parameter(
            "trajectory_file",
            trajectory_filename or "optimization_trajectory_cory105_tracking.npz",
        )
        self.declare_parameter("k_pos", 0.8)
        self.declare_parameter("k_heading", 2.0)
        self.declare_parameter("k_theta", 0.5)
        self.declare_parameter("v_max_cmd", 0.22)
        self.declare_parameter("omega_max_cmd", 1.8)
        self.declare_parameter("max_lin_accel", 0.18)
        self.declare_parameter("max_ang_accel", 1.2)
        self.declare_parameter("startup_hold_s", 0.5)
        self.declare_parameter("startup_ramp_s", 1.2)
        self.declare_parameter("use_feedback", False)
        self.declare_parameter("omega_sign", 1.0)
        self.declare_parameter("allow_in_place_turn", False)
        self.declare_parameter("min_turn_speed", 0.05)
        self.declare_parameter("max_curvature", 2.0)
        self.declare_parameter("straight_omega_deadband", 0.06)
        self.declare_parameter("omega_bias", 0.0)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("debug_control", True)
        self.declare_parameter("debug_every_n_cycles", 10)

        self.trajectory_filename = self._resolve_trajectory_path(self.get_parameter("trajectory_file").value)
        self.k_pos = float(self.get_parameter("k_pos").value)
        self.k_heading = float(self.get_parameter("k_heading").value)
        self.k_theta = float(self.get_parameter("k_theta").value)
        self.v_max_cmd = float(self.get_parameter("v_max_cmd").value)
        self.omega_max_cmd = float(self.get_parameter("omega_max_cmd").value)
        self.max_lin_accel = float(self.get_parameter("max_lin_accel").value)
        self.max_ang_accel = float(self.get_parameter("max_ang_accel").value)
        self.startup_hold_s = float(self.get_parameter("startup_hold_s").value)
        self.startup_ramp_s = float(self.get_parameter("startup_ramp_s").value)
        self.use_feedback = bool(self.get_parameter("use_feedback").value)
        self.omega_sign = float(self.get_parameter("omega_sign").value)
        self.allow_in_place_turn = bool(self.get_parameter("allow_in_place_turn").value)
        self.min_turn_speed = float(self.get_parameter("min_turn_speed").value)
        self.max_curvature = float(self.get_parameter("max_curvature").value)
        self.straight_omega_deadband = float(self.get_parameter("straight_omega_deadband").value)
        self.omega_bias = float(self.get_parameter("omega_bias").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.debug_control = bool(self.get_parameter("debug_control").value)
        self.debug_every_n_cycles = max(1, int(self.get_parameter("debug_every_n_cycles").value))
        self.mission_started = False
        self.control_thread = None

        # Publisher and TF setup
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer to trigger planning once pose is available
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info('TurtleBot controller node initialized.')

    def get_current_pose(self):
        try:
            # Get robot pose in odom frame.
            trans = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            yaw = self.quaternion_to_yaw(q.w, q.x, q.y, q.z)
            return (x, y, yaw)
        except Exception as e:
            self.get_logger().warn(
                f"Could not get transform {self.odom_frame}->{self.base_frame}: {e}"
            )
            return None

    def timer_callback(self):
        if self.mission_started:
            return
        # Timer stops so we don't plan again
        self.timer.cancel()
        self.mission_started = True
        self.control_thread = threading.Thread(target=self.plan_and_follow, daemon=True)
        self.control_thread.start()

    def plan_and_follow(self):
        if self.trajectory_filename is None:
            self.get_logger().info("No trajectory file provided.")
            return
        if not os.path.exists(self.trajectory_filename):
            self.get_logger().error(f"Trajectory file not found: {self.trajectory_filename}")
            return

        trajectory_data = np.load(self.trajectory_filename, allow_pickle=True)
        result = PlannerResult(
            x=trajectory_data['x'],
            y=trajectory_data['y'],
            theta=trajectory_data['theta'],
            v=trajectory_data['v'],
            omega=trajectory_data['omega'],
            total_time=trajectory_data['total_time'],
            dt=trajectory_data['dt'],
            solver_stats=None,
            success=True
        )
        self.get_logger().info(
            "Loaded trajectory "
            f"'{self.trajectory_filename}' with {len(result.x)} states, {len(result.v)} controls, "
            f"total_time={float(result.total_time):.3f}s, dt={float(result.dt):.4f}s."
        )
        self.follow_trajectory(result)

    def follow_trajectory(self, result):
        """
        Follow the generated trajectory by executing the computed controls (v, omega).
        """
        if result.dt <= 0.0 or len(result.v) == 0 or len(result.omega) == 0:
            self.get_logger().error("Invalid trajectory data. Nothing to execute.")
            self.pub.publish(Twist())
            return

        n_controls = min(len(result.v), len(result.omega), len(result.x) - 1, len(result.y) - 1, len(result.theta) - 1)
        if n_controls <= 0:
            self.get_logger().error("Trajectory arrays are inconsistent.")
            self.pub.publish(Twist())
            return

        dt = float(result.dt)
        rate_hz = max(5.0, self.control_rate_hz)
        nominal_cycle_dt = 1.0 / rate_hz
        if dt < nominal_cycle_dt:
            self.get_logger().warn(
                f"Trajectory dt ({dt:.4f}s) is smaller than control period ({nominal_cycle_dt:.4f}s). "
                "Using trajectory dt for per-step command timing."
            )
        v_prev = 0.0
        omega_prev = 0.0
        last_pose = None
        stale_pose_cycles = 0
        stale_pose_warned = False

        # Force a short zero-velocity startup period.
        hold_steps = max(1, int(round(max(0.0, self.startup_hold_s) * rate_hz)))
        for _ in range(hold_steps):
            if not rclpy.ok():
                break
            self.pub.publish(Twist())
            time.sleep(nominal_cycle_dt)

        # Align planned frame to measured start pose in odom to avoid startup spin.
        start_pose = None
        wait_start = time.monotonic()
        while rclpy.ok() and (time.monotonic() - wait_start) < 3.0 and start_pose is None:
            start_pose = self.get_current_pose()
            if start_pose is None:
                self.pub.publish(Twist())
                time.sleep(nominal_cycle_dt)

        if start_pose is None:
            rot_c, rot_s = 1.0, 0.0
            theta_offset = 0.0
            tx, ty = 0.0, 0.0
        else:
            x0_meas, y0_meas, yaw0_meas = start_pose
            x0_plan = float(result.x[0])
            y0_plan = float(result.y[0])
            theta0_plan = float(result.theta[0])
            theta_offset = self.wrap_to_pi(yaw0_meas - theta0_plan)
            rot_c = math.cos(theta_offset)
            rot_s = math.sin(theta_offset)
            tx = x0_meas - (rot_c * x0_plan - rot_s * y0_plan)
            ty = y0_meas - (rot_s * x0_plan + rot_c * y0_plan)

        self.get_logger().info(f"Executing {n_controls} control steps (dt={dt:.3f}s).")
        self.get_logger().info(
            f"Control mode: {'feedback' if self.use_feedback else 'open_loop'}, "
            f"rate={rate_hz:.1f}Hz, debug_control={self.debug_control}, "
            f"debug_every_n_cycles={self.debug_every_n_cycles}, "
            f"tf={self.odom_frame}->{self.base_frame}."
        )
        motion_start_time = time.monotonic()
        control_cycle = 0

        for k in range(n_controls):
            if not rclpy.ok():
                break

            x_plan = float(result.x[k + 1])
            y_plan = float(result.y[k + 1])
            theta_plan = float(result.theta[k + 1])
            x_ref = rot_c * x_plan - rot_s * y_plan + tx
            y_ref = rot_s * x_plan + rot_c * y_plan + ty
            theta_ref = self.wrap_to_pi(theta_plan + theta_offset)
            v_ff = float(result.v[k])
            omega_ff = float(result.omega[k])

            step_end = time.monotonic() + dt
            while rclpy.ok():
                now = time.monotonic()
                remaining = step_end - now
                if remaining <= 0.0:
                    break
                cycle_dt = min(nominal_cycle_dt, remaining)

                pose = self.get_current_pose()
                cmd = Twist()

                if pose is None:
                    self.pub.publish(cmd)
                    time.sleep(cycle_dt)
                    continue

                x, y, yaw = pose
                if last_pose is not None:
                    pose_delta = (
                        abs(x - last_pose[0]) + abs(y - last_pose[1]) + abs(self.wrap_to_pi(yaw - last_pose[2]))
                    )
                    if pose_delta < 1e-4:
                        stale_pose_cycles += 1
                    else:
                        stale_pose_cycles = 0
                        stale_pose_warned = False
                last_pose = (x, y, yaw)
                if stale_pose_cycles > 50 and not stale_pose_warned:
                    self.get_logger().warn(
                        "Robot pose appears frozen while commands are being sent. "
                        f"Check that {self.odom_frame}->{self.base_frame} TF is updating."
                    )
                    stale_pose_warned = True

                ex = x_ref - x
                ey = y_ref - y
                ex_body = math.cos(yaw) * ex + math.sin(yaw) * ey
                ey_body = -math.sin(yaw) * ex + math.cos(yaw) * ey
                theta_err = self.wrap_to_pi(theta_ref - yaw)
                if self.use_feedback:
                    # Pose error expressed in robot frame for stable unicycle tracking.
                    # Feedforward controls from planner + body-frame feedback terms.
                    v_raw = v_ff * math.cos(theta_err) + self.k_pos * ex_body
                    omega_raw = omega_ff + self.k_heading * ey_body + self.k_theta * theta_err

                    # Keep wheel velocities matched when driving straight.
                    if abs(theta_err) < 0.05 and abs(ey_body) < 0.03 and abs(omega_ff) < 0.05:
                        omega_raw = 0.0
                else:
                    # Robust default on hardware: follow planned controls open-loop.
                    v_raw = v_ff
                    omega_raw = omega_ff

                v_limited = float(np.clip(v_raw, -self.v_max_cmd, self.v_max_cmd))
                omega_limited = float(np.clip(omega_raw, -self.omega_max_cmd, self.omega_max_cmd))

                if self.startup_ramp_s > 0.0:
                    elapsed = time.monotonic() - motion_start_time
                    ramp = self._smoothstep(elapsed / self.startup_ramp_s)
                else:
                    ramp = 1.0
                v_target = ramp * v_limited
                omega_target = ramp * omega_limited

                # Slew-rate limit to avoid abrupt acceleration.
                v_step = self.max_lin_accel * cycle_dt
                omega_step = self.max_ang_accel * cycle_dt
                v_cmd = self._slew_limit(v_prev, v_target, v_step)
                omega_cmd = self._slew_limit(omega_prev, omega_target, omega_step)
                straight_zeroed = False
                inplace_zeroed = False

                # For near-straight segments, suppress yaw to keep wheel speeds matched.
                if abs(omega_ff) < self.straight_omega_deadband and abs(v_cmd) >= self.min_turn_speed:
                    omega_cmd = 0.0
                    straight_zeroed = True

                # Prevent spinning in place unless explicitly enabled.
                if (not self.allow_in_place_turn) and abs(v_cmd) < self.min_turn_speed:
                    omega_cmd = 0.0
                    inplace_zeroed = True

                # Limit curvature to avoid circular drift: |omega| <= kappa_max * |v|.
                curvature_bound = self.max_curvature * max(abs(v_cmd), self.min_turn_speed)
                omega_with_bias = omega_cmd + self.omega_bias
                omega_cmd = float(np.clip(omega_with_bias, -curvature_bound, curvature_bound))
                curvature_clipped = abs(omega_cmd - omega_with_bias) > 1e-9
                v_prev, omega_prev = v_cmd, omega_cmd

                cmd.linear.x = float(v_cmd)
                cmd.angular.z = float(self.omega_sign * omega_cmd)
                self.pub.publish(cmd)
                if self.debug_control and (control_cycle % self.debug_every_n_cycles == 0):
                    self.get_logger().info(
                        f"[k={k:03d} cycle={control_cycle:05d}] "
                        f"ref=({x_ref:.3f},{y_ref:.3f},{theta_ref:.3f}) "
                        f"pose=({x:.3f},{y:.3f},{yaw:.3f}) "
                        f"err_world=({ex:.3f},{ey:.3f}) err_body=({ex_body:.3f},{ey_body:.3f},{theta_err:.3f}) "
                        f"ff=(v={v_ff:.3f},w={omega_ff:.3f}) "
                        f"raw=(v={v_raw:.3f},w={omega_raw:.3f}) "
                        f"limited=(v={v_limited:.3f},w={omega_limited:.3f}) "
                        f"target=(v={v_target:.3f},w={omega_target:.3f}) "
                        f"cmd=(v={v_cmd:.3f},w={omega_cmd:.3f},pub_w={cmd.angular.z:.3f}) "
                        f"flags=(straight_zeroed={int(straight_zeroed)},"
                        f"inplace_zeroed={int(inplace_zeroed)},"
                        f"curvature_clipped={int(curvature_clipped)})"
                    )
                control_cycle += 1
                time.sleep(cycle_dt)

        # Stop the robot after trajectory is done
        self.pub.publish(Twist())
        if self.debug_control:
            self.get_logger().info(f"Executed {control_cycle} control cycles in trajectory loop.")
        self.get_logger().info("Trajectory finished.")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_from_yaw(yaw):
        """Return quaternion (x, y, z, w) from yaw angle."""
        return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
    @staticmethod
    def quaternion_to_yaw(w, x, y, z):
        """
        Convert a quaternion (w, x, y, z) into the yaw angle (rotation around z-axis).
        The result is in radians.
        """
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        
        return yaw

    @staticmethod
    def wrap_to_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _slew_limit(previous, target, max_step):
        return previous + float(np.clip(target - previous, -max_step, max_step))

    @staticmethod
    def _smoothstep(x):
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _resolve_trajectory_path(trajectory_filename):
        if trajectory_filename is None:
            return None

        p = Path(str(trajectory_filename)).expanduser()
        if p.is_absolute():
            return str(p)

        candidates = [Path.cwd() / p]
        # Search upward from this file location for common workspace roots.
        for parent in Path(__file__).resolve().parents:
            candidates.append(parent / p)
            candidates.append(parent / p.name)
            candidates.append(parent / "starter" / p.name)

        for c in candidates:
            if c.exists():
                return str(c)

        # Fallback to the original relative path.
        return str(p)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    
    default_file = os.environ.get("PLANNED_TRAJECTORY_FILE", "/home/cc/ee106b/sp26/class/ee106b-abg/106b-sp26-labs-starter/project3/starter/optimization_trajectory_cory105_min_time.npz")
    trajectory_filename = default_file
    if len(sys.argv) > 1:
        trajectory_filename = sys.argv[1]

    node = TurtleBotController(trajectory_filename=trajectory_filename)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
