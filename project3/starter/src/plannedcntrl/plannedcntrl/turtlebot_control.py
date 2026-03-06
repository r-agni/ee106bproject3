#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
from geometry_msgs.msg import Twist
import time
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
        self.declare_parameter("use_feedback", False)
        self.declare_parameter("omega_sign", 1.0)
        self.declare_parameter("allow_in_place_turn", False)
        self.declare_parameter("min_turn_speed", 0.05)
        self.declare_parameter("max_curvature", 2.0)
        self.declare_parameter("straight_omega_deadband", 0.06)
        self.declare_parameter("omega_bias", 0.0)
        self.declare_parameter("control_rate_hz", 20.0)

        self.trajectory_filename = self._resolve_trajectory_path(self.get_parameter("trajectory_file").value)
        self.k_pos = float(self.get_parameter("k_pos").value)
        self.k_heading = float(self.get_parameter("k_heading").value)
        self.k_theta = float(self.get_parameter("k_theta").value)
        self.v_max_cmd = float(self.get_parameter("v_max_cmd").value)
        self.omega_max_cmd = float(self.get_parameter("omega_max_cmd").value)
        self.max_lin_accel = float(self.get_parameter("max_lin_accel").value)
        self.max_ang_accel = float(self.get_parameter("max_ang_accel").value)
        self.startup_hold_s = float(self.get_parameter("startup_hold_s").value)
        self.use_feedback = bool(self.get_parameter("use_feedback").value)
        self.omega_sign = float(self.get_parameter("omega_sign").value)
        self.allow_in_place_turn = bool(self.get_parameter("allow_in_place_turn").value)
        self.min_turn_speed = float(self.get_parameter("min_turn_speed").value)
        self.max_curvature = float(self.get_parameter("max_curvature").value)
        self.straight_omega_deadband = float(self.get_parameter("straight_omega_deadband").value)
        self.omega_bias = float(self.get_parameter("omega_bias").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.mission_started = False

        # Publisher and TF setup
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer to trigger planning once pose is available
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info('TurtleBot controller node initialized.')

    def get_current_pose(self):
        try:
            # Get robot pose (base_link) in odom frame
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            yaw = self.quaternion_to_yaw(q.w, q.x, q.y, q.z)
            return (x, y, yaw)
        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")
            return None

    def timer_callback(self):
        if self.mission_started:
            return
        # Timer stops so we don't plan again
        self.timer.cancel()
        self.mission_started = True

        self.plan_and_follow()

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
        sleep_dt = 1.0 / rate_hz
        v_prev = 0.0
        omega_prev = 0.0

        # Force a short zero-velocity startup period.
        hold_steps = max(1, int(round(max(0.0, self.startup_hold_s) * rate_hz)))
        for _ in range(hold_steps):
            if not rclpy.ok():
                break
            self.pub.publish(Twist())
            time.sleep(sleep_dt)

        # Align planned frame to measured start pose in odom to avoid startup spin.
        start_pose = None
        wait_start = time.monotonic()
        while rclpy.ok() and (time.monotonic() - wait_start) < 3.0 and start_pose is None:
            start_pose = self.get_current_pose()
            if start_pose is None:
                self.pub.publish(Twist())
                time.sleep(sleep_dt)

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

            step_start = time.monotonic()
            while rclpy.ok() and (time.monotonic() - step_start) < dt:
                pose = self.get_current_pose()
                cmd = Twist()

                if pose is None:
                    self.pub.publish(cmd)
                    time.sleep(sleep_dt)
                    continue

                x, y, yaw = pose
                ex = x_ref - x
                ey = y_ref - y
                if self.use_feedback:
                    # Pose error expressed in robot frame for stable unicycle tracking.
                    ex_body = math.cos(yaw) * ex + math.sin(yaw) * ey
                    ey_body = -math.sin(yaw) * ex + math.cos(yaw) * ey
                    theta_err = self.wrap_to_pi(theta_ref - yaw)

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

                # Slew-rate limit to avoid abrupt acceleration.
                v_step = self.max_lin_accel * sleep_dt
                omega_step = self.max_ang_accel * sleep_dt
                v_cmd = self._slew_limit(v_prev, v_limited, v_step)
                omega_cmd = self._slew_limit(omega_prev, omega_limited, omega_step)

                # For near-straight segments, suppress yaw to keep wheel speeds matched.
                if abs(omega_ff) < self.straight_omega_deadband and abs(v_cmd) >= self.min_turn_speed:
                    omega_cmd = 0.0

                # Prevent spinning in place unless explicitly enabled.
                if (not self.allow_in_place_turn) and abs(v_cmd) < self.min_turn_speed:
                    omega_cmd = 0.0

                # Limit curvature to avoid circular drift: |omega| <= kappa_max * |v|.
                curvature_bound = self.max_curvature * max(abs(v_cmd), self.min_turn_speed)
                omega_cmd = float(np.clip(omega_cmd + self.omega_bias, -curvature_bound, curvature_bound))
                v_prev, omega_prev = v_cmd, omega_cmd

                cmd.linear.x = float(v_cmd)
                cmd.angular.z = float(self.omega_sign * omega_cmd)
                self.pub.publish(cmd)
                time.sleep(sleep_dt)

        # Stop the robot after trajectory is done
        self.pub.publish(Twist())
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
