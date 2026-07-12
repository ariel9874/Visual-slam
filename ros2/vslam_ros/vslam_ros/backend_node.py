"""backend_node: la trayectoria de keyframes y la corrección map→odom.

En este repo el backend REAL (BA local, iSAM2, cierre de bucle) corre DENTRO
del tracker (frontend_node) — separarlo por tópicos exigiría re-arquitectura,
no cáscara (regla 4). Lo que sí es del backend de cara al robot (REP-105):

    suscribe  /vslam/keyframes        (poses ópticas de los KFs)
    publica   /vslam/optimized_path   nav_msgs/Path (REP-103, frame "map")
              TF map → odom           (la corrección a saltos)

─── La matemática de map→odom (REP-105) ──────────────────────────────────────
El frontend publica odom→base continuo (SUAVE: sin saltos, con deriva). La pose
del último keyframe en el mundo corregido es T_map_kf; la que el flujo de
odometría le asignó en su momento, T_odom_kf. La corrección que reconcilia
ambos árboles es

        T_map_odom = T_map_kf · T_odom_kf⁻¹

y salta SOLO cuando el backend corrige (bucle/BA) — los consumidores eligen:
control usa odom (suave), navegación usa map (consistente). Con el backend
embebido en el tracker, la odometría YA incorpora las correcciones al frame
siguiente, así que T_map_odom ≈ I salvo el instante del salto — el nodo
materializa el patrón para que el árbol TF sea el canónico de un SLAM real.
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from tf2_ros import TransformBroadcaster

sys.path.insert(0, "/workspace")

from vslam_msgs.msg import Keyframe                               # noqa: E402
from vslam_ros.conversions import (T_to_pose, T_to_transform,     # noqa: E402
                                   optical_to_rep103, pose_to_T)


class BackendNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("vslam_backend")
        self._active = False
        self._kf_poses: dict = {}                 # id → T óptica
        self._last_odom_T = np.eye(4)             # T_odom_base más reciente
        self._path = Path()
        self._path.header.frame_id = "map"

    def on_configure(self, state) -> TransitionCallbackReturn:
        self.create_subscription(Keyframe, "/vslam/keyframes",
                                 self._on_keyframe, 50)
        self.create_subscription(Odometry, "/vslam/odom", self._on_odom, 10)
        self.pub_path = self.create_lifecycle_publisher(
            Path, "/vslam/optimized_path", 1)
        self._tf = TransformBroadcaster(self)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        self._active = True
        return super().on_activate(state)

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        self._active = False
        return super().on_deactivate(state)

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        self._kf_poses.clear()
        self._path = Path()
        self._path.header.frame_id = "map"
        return TransitionCallbackReturn.SUCCESS

    def _on_odom(self, msg: Odometry) -> None:
        self._last_odom_T = pose_to_T(msg.pose.pose)          # ya en REP-103

    def _on_keyframe(self, kf: Keyframe) -> None:
        if not self._active:
            return
        self._kf_poses[kf.id] = pose_to_T(kf.pose)            # óptica
        T_map_kf = optical_to_rep103(self._kf_poses[kf.id])

        ps = PoseStamped()
        ps.header.stamp = kf.header.stamp
        ps.header.frame_id = "map"
        ps.pose = T_to_pose(T_map_kf)
        self._path.poses.append(ps)
        self._path.header.stamp = kf.header.stamp
        self.pub_path.publish(self._path)

        # T_map_odom = T_map_kf · T_odom_kf⁻¹ (teoría arriba). El KF acaba de
        # promoverse: su pose de odometría es la última publicada.
        T_map_odom = T_map_kf @ np.linalg.inv(self._last_odom_T)
        tf = TransformStamped()
        tf.header.stamp = kf.header.stamp
        tf.header.frame_id = "map"
        tf.child_frame_id = "odom"
        tf.transform = T_to_transform(T_map_odom)
        self._tf.sendTransform(tf)


def main() -> None:
    rclpy.init()
    node = BackendNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
