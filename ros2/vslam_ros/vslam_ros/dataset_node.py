"""dataset_node: re-publica un dataset (TUM RGB-D o EuRoC estéreo) como cámara.

El sustituto del driver de cámara para las demos sin hardware: lee el dataset
del disco (los MISMOS loaders del núcleo, vslam/io/dataset.py) y publica

    /camera/image_raw          sensor_msgs/Image (mono8)
    /camera/depth/image_raw    sensor_msgs/Image (32FC1, metros)
    /camera/camera_info        sensor_msgs/CameraInfo (K + distorsión)

a la cadencia pedida. Dos personalidades (param `dataset`):
  - `tum`: imagen CRUDA + distorsión en la CameraInfo — quitarla es trabajo
    del consumidor (frontend_node), igual que con un driver real.
  - `euroc`: par estéreo RECTIFICADO por el rig del núcleo (EuRoCStereoRig) y
    profundidad de SGBM (EuRoCStereoLoader) — la MISMA firma (ts, gray, depth)
    que RGB-D (lección 37). La CameraInfo va sin distorsión (ya rectificada);
    el bf del rig NO viaja en CameraInfo: lo fija el launch en el frontend
    (param stereo_bf).

QoS sensor-data (best effort): un frame perdido no es un error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/workspace")             # el repo montado en el contenedor

from vslam.io.dataset import (EuRoCStereoLoader, TUMRGBDLoader,  # noqa: E402
                              tum_camera)


def _to_image_msg(arr: np.ndarray, stamp, frame_id: str, encoding: str) -> Image:
    m = Image()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.height, m.width = arr.shape[:2]
    m.encoding = encoding
    m.is_bigendian = False
    m.step = arr.strides[0]
    m.data = arr.tobytes()
    return m


class DatasetNode(Node):
    def __init__(self) -> None:
        super().__init__("vslam_dataset")
        self.declare_parameter("root", "")
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("dataset", "tum")     # tum | euroc
        root = Path(self.get_parameter("root").value)
        if not root.exists():
            raise SystemExit(f"dataset no encontrado: {root} (param 'root')")
        kind = str(self.get_parameter("dataset").value)
        if kind == "euroc":
            self._loader = EuRoCStereoLoader(root)
            self.camera = self._loader.camera        # izquierda RECTIFICADA
            self.get_logger().info(
                f"EuRoC estéreo: bf={self._loader.stereo_bf:.2f} "
                "(pasarlo al frontend como stereo_bf)")
        else:
            self.camera = tum_camera(root.name)
            self._loader = TUMRGBDLoader(root, with_depth=True)
        self._it = iter(self._loader)
        self._loop = bool(self.get_parameter("loop").value)

        self.pub_img = self.create_publisher(Image, "/camera/image_raw",
                                             qos_profile_sensor_data)
        self.pub_depth = self.create_publisher(Image, "/camera/depth/image_raw",
                                               qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/camera_info",
                                              qos_profile_sensor_data)
        cam = self.camera
        self._info = CameraInfo()
        self._info.header.frame_id = "camera_optical"
        self._info.width, self._info.height = cam.width, cam.height
        self._info.k = [float(v) for v in cam.K.ravel()]
        self._info.d = [float(v) for v in np.ravel(cam.dist)]
        self._info.distortion_model = "plumb_bob"
        self._info.p = [cam.K[0, 0], 0.0, cam.K[0, 2], 0.0,
                        0.0, cam.K[1, 1], cam.K[1, 2], 0.0,
                        0.0, 0.0, 1.0, 0.0]

        rate = float(self.get_parameter("rate").value)
        self.published = 0
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(f"publicando {root.name} a {rate:.0f} Hz")

    def _tick(self) -> None:
        try:
            ts, gray, depth = next(self._it)
        except StopIteration:
            if self._loop:
                self._it = iter(self._loader)
                return
            self.get_logger().info(
                f"dataset agotado ({self.published} frames) — fin.")
            raise SystemExit(0)
        stamp = self.get_clock().now().to_msg()
        self._info.header.stamp = stamp
        self.pub_info.publish(self._info)
        self.pub_img.publish(_to_image_msg(gray, stamp, "camera_optical", "mono8"))
        if depth is not None:
            self.pub_depth.publish(_to_image_msg(
                depth.astype(np.float32), stamp, "camera_optical", "32FC1"))
        self.published += 1


def main() -> None:
    rclpy.init()
    node = DatasetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
