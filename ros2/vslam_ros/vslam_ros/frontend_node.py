"""frontend_node: el tracker del núcleo detrás de tópicos ROS (LifecycleNode).

Cáscara FINA (regla 4): este nodo no sabe de SLAM — construye el `PnPTracker`
del núcleo con la CameraInfo que llega, le pasa frames y traduce sus salidas:

    suscribe  /camera/image_raw + /camera/depth/image_raw (sincronizados)
              /camera/camera_info (una vez: intrínsecos + distorsión)
    publica   /vslam/odom            nav_msgs/Odometry   (REP-103, por frame)
              TF odom → base_link    (continuo y suave — REP-105)
              /vslam/tracking_state  vslam_msgs/TrackingState (diagnóstico)
              /vslam/keyframes       vslam_msgs/Keyframe (imagen+depth+pose
                                     ópticos + K, para mapper/backend)

LIFECYCLE (ros2/README.md): en un robot real el SLAM debe poder reiniciarse
sin tocar los drivers de cámara. `configure` arma las suscripciones (el tracker
nace perezoso de la CameraInfo), `activate/deactivate` PAUSAN el procesamiento
(los frames del driver siguen llegando y se ignoran), `cleanup` DESTRUYE el
tracker (la siguiente configure+activate arranca un SLAM nuevo).

    ros2 lifecycle set /vslam_frontend configure && ... activate

Parámetros: `kf_scale` (reducción de la imagen del Keyframe), `features`
(orb/superpoint), `stereo_bf` y `depth_max` (para EuRoC estéreo: el bf del rig
rectificado no viaja en CameraInfo — lo fija el launch, ver euroc_demo).

La conversión de ejes óptico→REP-103 pasa SOLO por conversions.py.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import rclpy
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

sys.path.insert(0, "/workspace")

from vslam_msgs.msg import Keyframe, TrackingState                # noqa: E402
from vslam_ros.conversions import (T_to_pose, T_to_transform,     # noqa: E402
                                   optical_to_rep103)


class FrontendNode(LifecycleNode):
    def __init__(self) -> None:
        super().__init__("vslam_frontend")
        self.declare_parameter("kf_scale", 2)     # reducción de la imagen del KF
        self.declare_parameter("features", "orb")
        self.declare_parameter("stereo_bf", 0.0)  # 0 = default del tracker (TUM)
        self.declare_parameter("depth_max", 0.0)  # 0 = default del tracker
        self._active = False
        self._tracker = None
        self._maps = None
        self._n_kfs = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_configure(self, state) -> TransitionCallbackReturn:
        self.pub_odom = self.create_lifecycle_publisher(Odometry,
                                                        "/vslam/odom", 10)
        self.pub_state = self.create_lifecycle_publisher(
            TrackingState, "/vslam/tracking_state", 10)
        # Keyframes con QoS FIABLE: perder uno corrompe el mapa (ros2/README).
        self.pub_kf = self.create_lifecycle_publisher(Keyframe,
                                                      "/vslam/keyframes", 50)
        self._tf = TransformBroadcaster(self)
        self.create_subscription(CameraInfo, "/camera/camera_info",
                                 self._on_info, qos_profile_sensor_data)
        sub_img = Subscriber(self, Image, "/camera/image_raw",
                             qos_profile=qos_profile_sensor_data)
        sub_depth = Subscriber(self, Image, "/camera/depth/image_raw",
                               qos_profile=qos_profile_sensor_data)
        self._sync = ApproximateTimeSynchronizer([sub_img, sub_depth],
                                                 queue_size=10, slop=0.05)
        self._sync.registerCallback(self._on_frame)
        self.get_logger().info("configurado: esperando camera_info y activate")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state) -> TransitionCallbackReturn:
        self._active = True
        self.get_logger().info("ACTIVO: procesando frames")
        return super().on_activate(state)

    def on_deactivate(self, state) -> TransitionCallbackReturn:
        self._active = False
        self.get_logger().info("PAUSADO: los frames del driver se ignoran")
        return super().on_deactivate(state)

    def on_cleanup(self, state) -> TransitionCallbackReturn:
        self._tracker = None                 # el siguiente ciclo = SLAM nuevo
        self._maps = None
        self._n_kfs = 0
        self.get_logger().info("limpio: tracker destruido")
        return TransitionCallbackReturn.SUCCESS

    # ── armado perezoso: el tracker nace de la CameraInfo (como de un driver) ─

    def _on_info(self, msg: CameraInfo) -> None:
        if self._tracker is not None:
            return
        from vslam.core.camera import PinholeCamera
        from vslam.frontend.features import create_extractor
        from vslam.frontend.matching import create_matcher
        from vslam.frontend.tracker import PnPTracker
        K = np.array(msg.k).reshape(3, 3)
        d = tuple(msg.d[:5]) + (0.0,) * max(0, 5 - len(msg.d))   # k1 k2 p1 p2 k3
        cam = PinholeCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                            width=msg.width, height=msg.height, distortion=d)
        self._maps = cv2.initUndistortRectifyMap(
            cam.K, cam.dist, None, cam.K, (cam.width, cam.height), cv2.CV_32FC1)
        feats = self.get_parameter("features").value
        self._tracker = PnPTracker(cam, extractor=create_extractor(feats),
                                   matcher=create_matcher("ratio"),
                                   local_window=8, local_ba=True,
                                   loop_closure=True)
        bf = float(self.get_parameter("stereo_bf").value)
        if bf > 0:
            self._tracker.STEREO_BF = bf     # estéreo real (EuRoC): bf del rig
        dmax = float(self.get_parameter("depth_max").value)
        if dmax > 0:
            self._tracker.DEPTH_MAX = dmax
        self.get_logger().info(
            f"tracker listo ({feats}, {cam.width}x{cam.height}"
            f"{f', bf={bf}' if bf > 0 else ''})")

    # ── por frame: procesar y traducir ────────────────────────────────────────

    def _on_frame(self, img_msg: Image, depth_msg: Image) -> None:
        if not self._active or self._tracker is None:
            return
        gray = np.frombuffer(img_msg.data, np.uint8).reshape(
            img_msg.height, img_msg.width)
        depth = np.frombuffer(depth_msg.data, np.float32).reshape(
            depth_msg.height, depth_msg.width)
        rect = cv2.remap(gray, self._maps[0], self._maps[1], cv2.INTER_LINEAR)
        drect = cv2.remap(depth, self._maps[0], self._maps[1], cv2.INTER_NEAREST)
        _, info = self._tracker.process_frame(rect, drect)

        stamp = img_msg.header.stamp
        T_ros = optical_to_rep103(self._tracker.T_w_c)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = T_to_pose(T_ros)
        self.pub_odom.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform = T_to_transform(T_ros)
        self._tf.sendTransform(tf)

        st = TrackingState()
        st.header.stamp = stamp
        st.frame_id = int(self._tracker._frame_idx)
        s = str(info.get("state", ""))
        st.state = (TrackingState.COASTING if "COAST" in s
                    else TrackingState.OK if info.get("tracked")
                    else TrackingState.LOST)
        st.inliers = int(info.get("n_inliers", 0))
        st.keypoints = int(info.get("n_kps", 0))
        st.map_points = len(self._tracker.mapper)
        st.metric = bool(self._tracker._metric)
        self.pub_state.publish(st)

        if len(self._tracker._kf_ids) > self._n_kfs:
            self._n_kfs = len(self._tracker._kf_ids)
            self._publish_keyframe(stamp, rect, drect)

    def _publish_keyframe(self, stamp, rect, drect) -> None:
        """El contrato del mapper denso (v0.7): imagen + profundidad + pose + K."""
        sc = int(self.get_parameter("kf_scale").value)
        h, w = rect.shape[0] // sc, rect.shape[1] // sc
        kf = Keyframe()
        kf.header.stamp = stamp
        kf.header.frame_id = "map_optical"
        kf.id = int(self._tracker._kf_ids[-1])
        kf.pose = T_to_pose(self._tracker.T_w_c)          # ÓPTICO (ver .msg)
        cam = self._tracker.camera
        kf.k = [cam.fx / sc, 0.0, cam.cx / sc,
                0.0, cam.fy / sc, cam.cy / sc, 0.0, 0.0, 1.0]
        img_s = cv2.resize(rect, (w, h), interpolation=cv2.INTER_AREA)
        d_s = cv2.resize(drect, (w, h), interpolation=cv2.INTER_NEAREST)
        kf.image.height, kf.image.width = h, w
        kf.image.encoding = "mono8"
        kf.image.step = w
        kf.image.data = img_s.tobytes()
        kf.depth.height, kf.depth.width = h, w
        kf.depth.encoding = "32FC1"
        kf.depth.step = w * 4
        kf.depth.data = d_s.astype(np.float32).tobytes()
        self.pub_kf.publish(kf)


def main() -> None:
    rclpy.init()
    node = FrontendNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
