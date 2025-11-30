import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from mocap4r2_msgs.msg import RigidBodies
import math
import time
import numpy as np 

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')
        
        self.output_file = "calibration_results_averaged.txt"
        self.samples_per_motion = 5
        self.motion_duration = 1.0
        self.stabilize_time = 1.0
        self.reset_time = 1.0
        
        # Diccionario de comandos
        self.commands_dict = {
            "Pivot_Left": 227, 
            "Pivot_Right": 228, 
             "FwdSteer_Left": 251, 
            # "Fwd": 252, 
             "FwdSteer_Right": 253,
             "BwdSteer_Left": 256, 
             "BwdSteer_Right": 258, 
            # "FastFwd": 262, 
            # "Bwd": 266, 
            # "FastBwd": 267,
            "FastFwdSteer_Left": 261,
            "FastFwdSteer_Right": 263
        }
        
        self.command_queue = list(self.commands_dict.items())
        
        self.current_cmd_idx = 0     
        self.current_sample_idx = 0  

        self.batch_fwd = []
        self.batch_lat = []
        self.batch_rot = []
        
        self.spider_body = None
        self.publisher_ = self.create_publisher(Int32, 'cm550_command', 10)
        self.rigid_body_suscriber = self.create_subscription(
            RigidBodies, 'rigid_bodies', self.positions_callback, 10
        )

        # Estados
        self.state = "WAITING_MOCAP" 
        self.timer = self.create_timer(0.1, self.control_loop)
        self.state_timer = 0.0
        self.last_time = time.time()
        self.start_pose = None 
        
        # Archivo
        with open(self.output_file, "w") as f:
            header = (f"Motion_Name, Motion_ID, Samples, "
                      f"Avg_Delta_Fwd(m), Avg_Delta_Lat(m), Avg_Delta_Theta(deg)\n")
            f.write(header)
        
        self.get_logger().info(f"Calibración iniciada. {self.samples_per_motion} repeticiones seguidas por motion.")

    def get_heading_from_quaternion(self, q):
        t3 = +2.0 * (q.w * q.y - q.z * q.x)
        t4 = +1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        heading_y = math.atan2(t3, t4)
        return heading_y

    def normalize_angle(self, angle):
        while angle > math.pi: angle -= 2.0 * math.pi
        while angle < -math.pi: angle += 2.0 * math.pi
        return angle

    def positions_callback(self, msg: RigidBodies):
        for body in msg.rigidbodies:
            if body.rigid_body_name == '10': 
                self.spider_body = body

    def send_command(self, value: int):
        msg = Int32()
        msg.data = value
        self.publisher_.publish(msg)

    def calculate_deltas(self, pose_start, pose_end):
        # 1. Globales
        x0, z0 = pose_start.position.x, pose_start.position.z
        x1, z1 = pose_end.position.x, pose_end.position.z
        
        theta0 = self.get_heading_from_quaternion(pose_start.orientation)
        theta1 = self.get_heading_from_quaternion(pose_end.orientation)

        # 2. Diferencias Globales
        dX_global = x1 - x0 
        dZ_global = z1 - z0 
        dTheta = self.normalize_angle(theta1 - theta0)

        # 3. Proyección Local (con Ejes Invertidos según tu pedido)
        cos_h = math.cos(theta0)
        sin_h = math.sin(theta0)

        # Ortogonal al Heading -> Forward
        delta_forward = -dZ_global * sin_h + dX_global * cos_h
        
        # Paralelo al Heading -> Lateral
        delta_lateral = dZ_global * cos_h + dX_global * sin_h

        return delta_forward, delta_lateral, dTheta

    def control_loop(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # --- MÁQUINA DE ESTADOS ---

        if self.state == "WAITING_MOCAP":
            if self.spider_body is not None:
                self.get_logger().info("Mocap detectado. Iniciando primera motion...")
                self.state = "RESET_POSE" # La primera vez siempre reseteamos

        # 1. RESET POSE: Solo se llama al cambiar de tipo de Motion
        elif self.state == "RESET_POSE":
            self.send_command(201) # Stand Up / Init Pose
            self.state_timer = 0
            self.state = "WAITING_RESET"
            
        elif self.state == "WAITING_RESET":
            self.state_timer += dt
            if self.state_timer > self.reset_time:
                self.state = "STABILIZING"
                self.state_timer = 0

        # 2. STABILIZING: Esperar quieto para medir t0
        elif self.state == "STABILIZING":
            self.state_timer += dt
            if self.state_timer > self.stabilize_time:
                if self.spider_body is None: return
                self.start_pose = self.spider_body.pose # FOTO T0
                
                # Ejecutar comando
                cmd_name, cmd_val = self.command_queue[self.current_cmd_idx]
                sample_num = self.current_sample_idx + 1
                self.get_logger().info(f"Test '{cmd_name}' | Muestra {sample_num}/{self.samples_per_motion}")
                
                self.send_command(cmd_val)
                self.state_timer = 0
                self.state = "EXECUTING_MOTION"

        # 3. EXECUTING: Esperar a que termine el movimiento y medir t1
        elif self.state == "EXECUTING_MOTION":
            self.state_timer += dt
            if self.state_timer > self.motion_duration:
                if self.spider_body is None: return
                end_pose = self.spider_body.pose # FOTO T1
                
                # Calcular y guardar
                d_fwd, d_lat, d_rot = self.calculate_deltas(self.start_pose, end_pose)
                
                self.batch_fwd.append(d_fwd)
                self.batch_lat.append(d_lat)
                self.batch_rot.append(d_rot)
                
                self.current_sample_idx += 1
                
                # --- LÓGICA DE TRANSICIÓN MODIFICADA ---
                if self.current_sample_idx < self.samples_per_motion:
                    # CASO A: Faltan muestras para el mismo comando.
                    # NO reseteamos pose. Vamos directo a estabilizar en la nueva posición.
                    self.state = "STABILIZING"
                    self.state_timer = 0
                else:
                    # CASO B: Terminamos el lote de este comando.
                    self.process_batch_results()
                    
                    # Preparar siguiente comando
                    self.current_cmd_idx += 1
                    self.current_sample_idx = 0
                    self.batch_fwd = [] 
                    self.batch_lat = []
                    self.batch_rot = []
                    
                    if self.current_cmd_idx >= len(self.command_queue):
                        self.get_logger().info("✅ CALIBRACIÓN FINALIZADA EXITOSAMENTE")
                        self.destroy_node()
                    else:
                        self.state = "RESET_POSE"

    def process_batch_results(self):
        avg_fwd = np.mean(self.batch_fwd)
        avg_lat = np.mean(self.batch_lat)
        avg_rot_rad = np.mean(self.batch_rot)
        avg_rot_deg = math.degrees(avg_rot_rad)
        
        cmd_name, cmd_val = self.command_queue[self.current_cmd_idx]
        
        with open(self.output_file, "a") as f:
            line = (f"{cmd_name}, {cmd_val}, {self.samples_per_motion}, "
                    f"{avg_fwd:.5f}, {avg_lat:.5f}, {avg_rot_deg:.4f}\n")
            f.write(line)
        
        self.get_logger().info(f"--> PROMEDIO {cmd_name}: Fwd={avg_fwd:.3f}, Lat={avg_lat:.3f}, Rot={avg_rot_deg:.1f}°")

def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()