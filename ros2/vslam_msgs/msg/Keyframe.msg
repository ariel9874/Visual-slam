# Un keyframe del frontend: el contrato central del repo (vslam/core/frame.py)
# en IDL. Lo consume el mapper_node (mapa denso 3DGS, v0.7: necesita imagen +
# profundidad + pose) y el backend_node (poses del grafo).
#
# CONVENCION: `pose` es T_w_c en ejes OPTICOS de OpenCV (+Z delante, +Y abajo),
# la convencion INTERNA del nucleo. La conversion a REP-103 (x delante, z
# arriba) se hace SOLO en las fronteras publicas (odom/TF) — ver
# vslam_ros/conversions.py. Entre nodos vslam_* viaja la convencion del nucleo.

std_msgs/Header header
uint64 id                       # frame_id del nucleo (creciente y unico)
geometry_msgs/Pose pose         # T_w_c OPTICO (ver arriba)
sensor_msgs/Image image         # mono8; puede ir reducida (param del frontend)
sensor_msgs/Image depth         # 32FC1 en metros; height=0 si no hay
float64[9] k                    # intrinsecos 3x3 fila-mayor DE LA IMAGEN QUE
#                                 VIAJA (ya escalados si va reducida)
float32[] keypoints             # [u1,v1,u2,v2,...] (vacio si no se publica)
uint8[] descriptors             # N*32 bytes ORB aplanados (vacio si no)
