import cv2
import torch
import time
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify, url_for
import threading
from picamera2 import Picamera2
import yaml
import serial
import time
import queue
import collections  # Import collections
from termcolor import colored

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Global variables
model = None
inference_lock = threading.Lock()
inference_enabled = False
label_names = []
current_frame = None  # Store the current frame
frame_lock = threading.Lock() # Lock for frame access
distanceInfo = 0
frame_stream = collections.deque(maxlen=4)


class ObjectDetector:
    def __init__(self, image_width, image_height, target_class_names):
        self.image_width = image_width
        self.image_height = image_height
        self.target_class_names = target_class_names
        self.image_center_x = image_width // 2
        self.image_center_y = image_height // 2

    def calculate_distance_x_direction(self, results):
        """
        Calculates the distance (in pixels or percentage) along the x-axis 
        from the center of the bounding box of the target class to the 
        center of the image, and determines the direction (left or right).

        Args:
            results: The detection results from the YOLO model.

        Returns:
            A dictionary containing the distance in pixels and percentage, 
            and the direction ("left" or "right"), or None if the target 
            class is not detected.
            Example: {"distance_pixels_x": 10, "distance_percentage_x": 0.01, "direction": "right"}
        """

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label_name = self.get_label_name(cls)

                if label_name in self.target_class_names:
                    xyxy = box.xyxy[0].tolist()
                    x1, _, x2, _ = map(int, xyxy)  # Only need x coordinates

                    bbox_center_x = (x1 + x2) // 2

                    distance_pixels_x = bbox_center_x - self.image_center_x
                    distance_percentage_x = distance_pixels_x / self.image_width
                    
                    bbox_width = abs(x2 - x1)  # Width of the bounding box
                    bbox_width_percentage = (bbox_width / self.image_width) * 100  # As a percentage of image width


                    direction = "left" if distance_pixels_x < 0 else "right"

                    return {
                        "distance_pixels_x": abs(distance_pixels_x),  # Absolute distance
                        "distance_percentage_x": abs(distance_percentage_x), # Absolute distance
                        "direction": direction,
                        "bbox_width_pixels": bbox_width,  # Add bbox width in pixels
                        "bbox_width_percentage": bbox_width_percentage,  # Add bbox width percentage
                    }

        return None


    def get_label_name(self, cls): # Dummy function, replace with your actual function
        global label_names
        """
        Gets the label name from the class ID.  You'll need to implement this
        based on how you manage your class names (e.g., from a YAML file).
        """
        # Replace this with your actual label name retrieval logic.
        # Example (if you have a list of label names):
        # label_names = ['class0', 'class1', 'class2', ...]
        # return label_names[cls]

        # Or, if you have a dictionary:
        #label_names = ['7Seg Display', 'Acumulator', 'Arduino Mega', 'Arduino Nano', 'Arduino Uno', 'Capacitor', 'DTH22Module', 'Diode', 'DistanceSensor', 'ESP-Cam', 'ESP32', 'FTDI', 'Geared motor', 'HeatSink', 'Integrated circuit', 'LCD16x2', 'MOSFET', 'MOSFETBoard', 'NFCModule', 'PiCamera', 'Potentiometer', 'Raspberry Pi 4', 'Raspberry Pi 5', 'RectifierBridge', 'RelayModule', 'Resistor', 'Servo', 'Step-up converter', 'Stepper motor driver', 'Transistor', 'USB-hub']

        return label_names[cls]
        return "PiCamera"
        return str(cls)  # Just return the class ID as a string for now

class SerialHandler:
    def __init__(self, serial_port, baud_rate):
        self.ser = None
        self.size = 0
        
                # Define RKZ commands and their properties
        self.rkz_commands = {
            'k': {'function': self.enable_inference, 'num_params': 0},
            'l': {'function': self.disable_inference, 'num_params': 0},
        }
        
        
        try:
            self.ser = serial.Serial(serial_port, baud_rate)
            print(colored(f"Serial port {self.ser.port} opened successfully.",'green'))
        except serial.SerialException as e:
            print(colored(f"Error opening serial port: {e}","red"))

    def convert_distance_to_range(self,distance_info, image_width):
        """Converts distance and direction to a value between 0 and 100."""
        if distance_info is None:
            return None

        self.size = distance_info["bbox_width_percentage"]
        distance_pixels = distance_info["distance_pixels_x"]
        direction = distance_info["direction"]

        # Calculate percentage distance (0.0 to 1.0)
        distance_percentage = distance_pixels / (image_width / 2)  # Normalize to half width

        # Convert percentage to range 0-100
        if direction == "left":
            range_value = int((1 - distance_percentage) * 50)  # Left: 50-0
        else:  # direction == "right"
            range_value = int((1 + distance_percentage) * 50)  # Right: 50-100

        # Clamp the value to the 0-100 range (important!)
        range_value = max(0, min(100, range_value))

        return range_value

    def send_distance_info(self, range_value):
        if self.ser is None:
            return
        if range_value is None:
            return

        try:
            stx = b'\x02'
            cst = int(self.size)  # Constant value
            dir_val = range_value
            etx = b'\x03'

            message = stx + bytes([cst]) + bytes([dir_val]) + etx # makes bytes of the message

            self.ser.write(message)
            self.ser.flush()
            print(colored(f"Sent: STX {cst} {dir_val} ETX",'yellow'))
        except serial.SerialException as e:
            print(colored(f"Serial error: {e}",'red'))

    def process_data(self):
        global inference_enabled

        if self.ser is None:
            return

        while True:
            try:
                if self.ser.in_waiting > 0:
                    stx = self.ser.read(1)
                    if stx == b'\x02':  # Check for STX (0x02)
                        rkz = self.ser.read(1).decode('utf-8')  # Read RKZ (1 char)
                        x_params = []
                        try:
                            x_len = int(self.ser.read(1).decode('utf-8')) # Read X length (1 char)
                        except ValueError:
                            print(colored("Invalid X length",'red'))
                            continue

                        for _ in range(x_len):
                            x_params.append(self.ser.read(1).decode('utf-8'))  # Read X params (x_len chars)

                        etx = self.ser.read(1)  # Read ETX (0x03)
                        if etx == b'\x03':  # Check for ETX
                            print(colored(f"Received data: RKZ={rkz}, X={x_params}",'yellow'))
                            self.handle_rkz(rkz, x_params)
                        else:
                            print(colored("ETX not found",'red'))
                else:
                    time.sleep(0.01)
            except serial.SerialException as e:
                print(colored(f"Serial communication error: {e}",'red'))
                break
            except Exception as e:
                print(colored(f"An unexpected error occurred: {e}",'red'))
                break

    def handle_rkz(self, rkz, x_params):
        if rkz in self.rkz_commands:
            command = self.rkz_commands[rkz]
            if len(x_params) == command['num_params']: # Check number of parameters
                command['function'](x_params) # Call the associated function
            else:
                print(colored(f"Incorrect number of parameters for RKZ '{rkz}'. Expected {command['num_params']}, got {len(x_params)}.",'red'))
        else:
            print(colored(f"Unknown RKZ command: {rkz}",'red'))
            
            

    def close_port(self):
        if self.ser:
            self.ser.close()
            print(colored("Serial port closed.",'yellow'))
            
        # Define the functions for each command
    def enable_inference(self, x_params):
        global inference_enabled
        with inference_lock:  # Protect shared variable
            inference_enabled = True
        print(colored("Inference enabled via serial command.",'green'))

    def disable_inference(self, x_params):
        global inference_enabled
        with inference_lock:  # Protect shared variable
            inference_enabled = False
        print(colored("Inference disabled via serial command.",'green'))


def capture_frames():
    global current_frame
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    while True:
        frame_rgb = picam2.capture_array()
        with frame_lock:  # Protect access to current_frame
            current_frame = frame_rgb.copy()  # Store a copy
    picam2.stop()
    picam2.close()

def initialize_model(model_path, yaml_path):
    global model, label_names
    model = YOLO(model_path, task='detect')
    print(colored(f"Model {model_path} loaded",'green'))

    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            label_names = data['names']
            print(colored(f"Loaded {len(label_names)} labels from {yaml_path}",'green'))
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(colored(f"Error loading YAML: {e}. Using default labels.",'yellow'))
        label_names = ['7Seg Display', 'Acumulator', 'Arduino Mega', 'Arduino Nano', 'Arduino Uno', 'Capacitor', 'DTH22Module', 'Diode', 'DistanceSensor', 'ESP-Cam', 'ESP32', 'FTDI', 'Geared motor', 'HeatSink', 'Integrated circuit', 'LCD16x2', 'MOSFET', 'MOSFETBoard', 'NFCModule', 'PiCamera', 'Potentiometer', 'Raspberry Pi 4', 'Raspberry Pi 5', 'RectifierBridge', 'RelayModule', 'Resistor', 'Servo', 'Step-up converter', 'Stepper motor driver', 'Transistor', 'USB-hub']


def generate_image_server():
    global frame_stream
    while True:
        if frame_stream:
            frame = frame_stream.popleft()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')    
        else:  # Deque is empty
            time.sleep(0.01)  # Small delay if deque is empty



def generate_frames():
    global model, label_names, current_frame, distanceInfo, frame_stream
    frame_count = 0
    previous_results = None  # Store results from the last inference frame

    while True:
        with frame_lock:
            frame_rgb = current_frame
            if frame_rgb is None:
                time.sleep(0.01)
                continue
            frame_to_display = frame_rgb.copy()

        frame_count += 1

        if inference_enabled:
            if frame_count % 4 == 0:  # Perform inference every 4th frame
                with inference_lock:
                    results = model(frame_rgb)
                    
                    
                    object_detector = ObjectDetector(640, 480, "PiCamera")
                    distanceInfo = object_detector.calculate_distance_x_direction(results)
                    info = serial_handler.convert_distance_to_range(distanceInfo,640)
                    serial_handler.send_distance_info(info)
                    
                    
                    previous_results = results  # Store the results

                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            xyxy = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            try:
                                label_name = label_names[cls]
                            except IndexError:
                                label_name = str(cls)
                                
                            x1, y1, x2, y2 = map(int, xyxy)
                            cv2.rectangle(frame_to_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{label_name} {conf:.2f}"
                            cv2.putText(frame_to_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            elif previous_results is not None:  # Use previous results if available
                with inference_lock:
                    for result in previous_results:  # Iterate through the previous results
                        boxes = result.boxes
                        for box in boxes:
                            xyxy = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            try:
                                label_name = label_names[cls]
                            except IndexError:
                                label_name = str(cls)

                            x1, y1, x2, y2 = map(int, xyxy)
                            cv2.rectangle(frame_to_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{label_name} {conf:.2f}"
                            cv2.putText(frame_to_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame_to_display)
        frame_bytes = buffer.tobytes()
        frame_stream.append(frame_bytes)




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_image_server(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_inference')
def toggle_inference():
    global inference_enabled
    inference_enabled = not inference_enabled
    return "Inference toggled"





if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True # Allow main thread to exit even if this is running
    capture_thread.start()
    
    serial_handler = SerialHandler('/dev/ttyS0', 9600)  # Replace with your port and baud rate
    serial_thread = threading.Thread(target=serial_handler.process_data)
    serial_thread.daemon = True
    serial_thread.start()
    
    display_thread = threading.Thread(target=generate_frames) # Start display_frames thread
    display_thread.daemon = True
    display_thread.start()



    try:
        initialize_model("best.onnx", "./data.yaml")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except Exception as e:
        print(colored(f"Error: {e}",'red'))

    finally:
         pass # No need to terminate a thread directly
