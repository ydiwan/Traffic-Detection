import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import serial
import struct
import numpy as np
from ultralytics import YOLO

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')
        self.status_pub = self.create_publisher(String, '/traffic_light_status', 10)
        
        self.get_logger().info("Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt") 
        
        # port
        self.serial_port = '/dev/ttyACM0' 
        self.serial = serial.Serial(self.serial_port, baudrate=115200, timeout=1) 
        
        # 20 fps
        self.timer = self.create_timer(0.05, self.process_frame)

    def process_frame(self):
        # prompt openmv cam for a pic
        self.serial.write(b'snap')
        
        # check header for byte count
        size_bytes = self.serial.read(4)
        if len(size_bytes) != 4:
            return
            
        size = struct.unpack("<L", size_bytes)[0]
        
        # read bytes from jpeg
        img_bytes = self.serial.read(size)
        if len(img_bytes) != size:
            return
            
        # decode into an opencv image array
        np_arr = np.frombuffer(img_bytes, np.uint8) 
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
            
        # run intereference
        results = self.model.predict(source=frame, device=0, verbose=False)
        
        detected_light = "NONE"
        for result in results:
            class_ids = result.boxes.cls.tolist()
            labels = [result.names[int(i)] for i in class_ids]
            
            if 'red_light' in labels: 
                detected_light = "RED"
            elif 'green_light' in labels:
                detected_light = "GREEN"
                
        # publish interupt
        msg = String()
        msg.data = detected_light
        self.status_pub.publish(msg)
        self.get_logger().info(f"Published: {detected_light}")

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.serial.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()