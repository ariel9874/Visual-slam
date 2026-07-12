# Arista del grafo de poses (factores del backend, vslam/backend/pose_graph.py):
# restriccion relativa entre dos keyframes (odometria o cierre de bucle) con su
# matriz de informacion. Convencion: `transform` = T_from_to en ejes OPTICOS.

uint64 id_from
uint64 id_to
geometry_msgs/Pose transform    # T_from_to OPTICO
float64[36] information         # 6x6 fila-mayor, orden tangente [rho, omega]
bool is_loop                    # true si es un cierre de bucle
