# Diagnostico del frontend por frame: lo minimo para saber si el SLAM esta
# vivo sin abrir RViz (ros2 topic echo /vslam/tracking_state).

uint8 OK=0
uint8 COASTING=1                # perdio el frame pero extrapola (v0.4)
uint8 LOST=2                    # esperando relocalizacion (v0.4b)

std_msgs/Header header
uint64 frame_id
uint8 state
uint32 inliers                  # inliers de PnP del frame
uint32 keypoints                # detecciones totales del frame
uint32 map_points               # puntos activos del mapa disperso
bool metric                     # mapa en metros (init RGB-D/estereo, v0.6)
