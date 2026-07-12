"""Suscriptor de humo: cuenta mensajes de todos los topicos del pipeline
durante N segundos y valida el arbol TF map->odom->base_link."""
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage

sys.path.insert(0, "/workspace")
from vslam_msgs.msg import Keyframe, TrackingState  # noqa: E402


class Smoke(Node):
    def __init__(self):
        super().__init__("smoke_sub")
        self.counts = {"odom": 0, "state": 0, "kf": 0, "path": 0, "map": 0}
        self.tf_frames = set()
        self.last_state = None
        self.create_subscription(Odometry, "/vslam/odom",
                                 lambda m: self._c("odom"), 10)
        self.create_subscription(TrackingState, "/vslam/tracking_state",
                                 self._on_state, 10)
        self.create_subscription(Keyframe, "/vslam/keyframes",
                                 lambda m: self._c("kf"), 50)
        self.create_subscription(Path, "/vslam/optimized_path",
                                 self._on_path, 10)
        self.create_subscription(PointCloud2, "/vslam/map",
                                 self._on_map, 10)
        self.create_subscription(TFMessage, "/tf", self._on_tf, 100)
        self.path_len = 0
        self.map_pts = 0

    def _c(self, k):
        self.counts[k] += 1

    def _on_state(self, m):
        self.counts["state"] += 1
        self.last_state = (m.state, m.inliers, m.map_points, m.metric)

    def _on_path(self, m):
        self.counts["path"] += 1
        self.path_len = len(m.poses)

    def _on_map(self, m):
        self.counts["map"] += 1
        self.map_pts = m.width

    def _on_tf(self, m):
        for t in m.transforms:
            self.tf_frames.add((t.header.frame_id, t.child_frame_id))


def main():
    rclpy.init()
    n = Smoke()
    t0 = time.time()
    while time.time() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 25):
        rclpy.spin_once(n, timeout_sec=0.1)
    print("conteos:", n.counts)
    print("ultimo estado (state, inliers, map_points, metric):", n.last_state)
    print("path len:", n.path_len, "| puntos del mapa:", n.map_pts)
    print("aristas TF:", sorted(n.tf_frames))
    ok = (n.counts["odom"] > 50 and n.counts["state"] > 50
          and n.counts["kf"] >= 2 and n.counts["path"] >= 2
          and n.counts["map"] >= 1
          and ("odom", "base_link") in n.tf_frames
          and ("map", "odom") in n.tf_frames)
    print("SMOKE:", "OK" if ok else "FALLO")
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
