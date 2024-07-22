import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')

        # Initialize lists to store data
        self.cluster_ids_list = []
        self.avg_x_list = []
        self.avg_y_list = []
        self.avg_z_list = []

        # Subscribe to each topic
        self.cluster_ids_sub = self.create_subscription(
            Float32MultiArray,
            'cluster_ids',
            self.cluster_ids_callback,
            10
        )
        self.avg_x_sub = self.create_subscription(
            Float32MultiArray,
            'avg_x',
            self.avg_x_callback,
            10
        )
        self.avg_y_sub = self.create_subscription(
            Float32MultiArray,
            'avg_y',
            self.avg_y_callback,
            10
        )
        self.avg_z_sub = self.create_subscription(
            Float32MultiArray,
            'avg_z',
            self.avg_z_callback,
            10
        )

    def cluster_ids_callback(self, msg):
        self.cluster_ids_list = msg.data
        self.get_logger().info(f'Received cluster IDs: {self.cluster_ids_list}')

    def avg_x_callback(self, msg):
        self.avg_x_list = msg.data
        self.get_logger().info(f'Received average X coordinates: {self.avg_x_list}')

    def avg_y_callback(self, msg):
        self.avg_y_list = msg.data
        self.get_logger().info(f'Received average Y coordinates: {self.avg_y_list}')

    def avg_z_callback(self, msg):
        self.avg_z_list = msg.data
        self.get_logger().info(f'Received average Z coordinates: {self.avg_z_list}')


def main(args=None):
    rclpy.init(args=args)
    node = DataSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Shutdown the ROS client library
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
