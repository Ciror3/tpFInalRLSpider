import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

class MotionPublisherNode(Node):
    def __init__(self):
        super().__init__('cm550_motion_publisher')
        self.publisher_ = self.create_publisher(Int32, 'cm550_command', 10)
        self.get_logger().info("✅ Listo para enviar comandos al CM-550 vía micro-ROS")

    def send_command(self, value: int):
        msg = Int32()
        msg.data = value
        self.publisher_.publish(msg)
        self.get_logger().info(f'📤 Enviado comando: {value}')

def main(args=None):
    rclpy.init(args=args)
    node = MotionPublisherNode()

    try:
        while True:
            value = int(input("Ingrese un número para enviar al CM-550: "))
            node.send_command(value)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()