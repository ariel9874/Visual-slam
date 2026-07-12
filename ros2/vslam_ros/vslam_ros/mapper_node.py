"""mapper_node: el mapa como consumidor de keyframes (la arquitectura de v0.7).

El mismo patrón del hito 5 de v0.7 (dense_thread.py) con la cola de proceso
sustituida por un tópico: el frontend publica Keyframe (imagen + profundidad +
pose óptica + K) y este nodo construye el mapa con SU presupuesto — en otro
proceso por construcción (cada nodo ROS es un proceso: la lección 42 del GIL
viene de serie con la arquitectura).

    suscribe  /vslam/keyframes      vslam_msgs/Keyframe (QoS fiable)
    publica   /vslam/map            sensor_msgs/PointCloud2 (xyz+intensidad,
                                    REP-103, frame "map") a `map_period` s

Para la demo de RViz el mapa es la nube RETRO-PROYECTADA de los keyframes
(RViz no rasteriza gaussianas; y este contenedor ROS no trae CUDA/torch). El
mapa 3DGS foto-realista vive en los ejemplos 07/08 con el contenedor gsplat
(lección 40); unir ambos mundos (nodo con torch) queda anotado en docs/05 §7.
"""

from __future__ import annotations

import struct
import sys

import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from sensor_msgs.msg import PointCloud2, PointField

sys.path.insert(0, "/workspace")

from vslam_msgs.msg import Keyframe                               # noqa: E402
from vslam_ros.conversions import R_BO, pose_to_T                 # noqa: E402


def _cloud_msg(points: np.ndarray, intens: np.ndarray, stamp,
               frame_id: str) -> PointCloud2:
    """(N,3) float32 + (N,) float32 → PointCloud2 xyzi."""
    m = PointCloud2()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.height, m.width = 1, len(points)
    m.fields = [
        PointField(name=n, offset=o, datatype=PointField.FLOAT32, count=1)
        for n, o in (("x", 0), ("y", 4), ("z", 8), ("intensity", 12))]
    m.is_bigendian = False
    m.point_step = 16
    m.row_step = 16 * len(points)
    buf = np.empty((len(points), 4), dtype=np.float32)
    buf[:, :3], buf[:, 3] = points, intens
    m.data = buf.tobytes()
    m.is_dense = True
    return m


class MapperNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("vslam_mapper")
        self.declare_parameter("map_period", 2.0)
        self.declare_parameter("seed_step", 6)
        self._active = False
        self._points: list = []          # nube acumulada: [(xyz, gris)]

    def on_configure(self, state) -> TransitionCallbackReturn:
        self.create_subscription(Keyframe, "/vslam/keyframes",
                                 self._on_keyframe, 50)
        self.pub_map = self.create_lifecycle_publisher(PointCloud2,
                                                       "/vslam/map", 1)
        self.create_timer(float(self.get_parameter("map_period").value),
                          self._publish_map)
        self.get_logger().info("configurado: esperando keyframes")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        self._active = True
        return super().on_activate(state)

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        self._active = False
        return super().on_deactivate(state)

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        self._points.clear()
        return TransitionCallbackReturn.SUCCESS

    def _on_keyframe(self, kf: Keyframe) -> None:
        if not self._active:
            return
        h, w = kf.image.height, kf.image.width
        if h == 0 or kf.depth.height == 0:
            return
        image = np.frombuffer(kf.image.data, np.uint8).reshape(h, w)
        depth = np.frombuffer(kf.depth.data, np.float32).reshape(h, w)
        T = pose_to_T(kf.pose)                       # ÓPTICO (contrato del msg)
        step = int(self.get_parameter("seed_step").value)
        # Retro-proyección de la rejilla (lección 39/41) con la K del mensaje
        # (ya escalada a la imagen que viaja).
        fx, fy = kf.k[0], kf.k[4]
        cx, cy = kf.k[2], kf.k[5]
        gv, gu = np.mgrid[0:h:step, 0:w:step]
        gu, gv = gu.ravel(), gv.ravel()
        z = depth[gv, gu]
        ok = (z > 0.1) & (z < 10.0)
        if not ok.any():
            return
        u, v, z = gu[ok], gv[ok], z[ok]
        Xc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
        Xw = (T[:3, :3] @ Xc.T).T + T[:3, 3]
        gris = image[v, u].astype(np.float32) / 255.0
        self._points.append((Xw.astype(np.float32), gris))
        self.get_logger().info(
            f"KF {kf.id}: +{len(z)} puntos (total "
            f"{sum(len(p) for p, _ in self._points)})")

    def _publish_map(self) -> None:
        if not self._points:
            return
        pts = np.concatenate([p for p, _ in self._points])
        intens = np.concatenate([g for _, g in self._points])
        # Óptico → REP-103 para RViz: los puntos del MUNDO rotan con R_bo.
        pts_ros = (R_BO @ pts.T).T.astype(np.float32)
        self.pub_map.publish(_cloud_msg(pts_ros, intens,
                                        self.get_clock().now().to_msg(), "map"))
        self.get_logger().info(f"mapa publicado: {len(pts_ros)} puntos",
                               throttle_duration_sec=5.0)


def main() -> None:
    rclpy.init()
    node = MapperNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
