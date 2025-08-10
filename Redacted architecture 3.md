```python
import time
import ujson
from machine import Pin, SPI, ADC, I2C, UART
import neopixel
import pygame
from typing import Dict, List, Optional

# Simulated hardware dependencies
DEPENDENCIES = {'emg_sensor': False, 'eye_tracker': False, 'pressure_sensor': True}

# Logging utility
def log_event(event_type: str, value: str):
    """Logs events to a file and sends via communication interfaces."""
    log = f"{time.time()},{event_type},{value}\n"
    with open("events.txt", "a") as f:
        f.write(log)
    print(f"Logged: {event_type} - {value}")

class SensorInterface:
    """Base class for sensor reading and calibration."""
    
    def __init__(self, pin: int, sensor_type: str):
        self.pin = Pin(pin, Pin.IN)
        self.sensor_type = sensor_type
        self.active = True

    def read(self) -> Optional[float]:
        """Placeholder for sensor reading."""
        return None

class EMGReader(SensorInterface):
    """Handles EMG sensor data reading."""
    
    def __init__(self, spi: SPI, cs_pin: int, drdy_pin: int):
        super().__init__(cs_pin, "emg")
        self.spi = spi
        self.cs = Pin(cs_pin, Pin.OUT)
        self.drdy = Pin(drdy_pin, Pin.IN)
        self.sample_rate = 250

    def set_sample_rate(self, hz: int):
        """Sets the sampling rate for EMG data."""
        self.sample_rate = hz
        print(f"EMG sample rate set to {hz} Hz")

    def read(self) -> List[float]:
        """Reads 4-channel EMG data (simulated)."""
        if self.active:
            return [random.uniform(0, 1) for _ in range(4)]  # Simulated data
        return [0.0] * 4

class FeedbackController:
    """Manages visual, auditory, and communication feedback."""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.led = neopixel.NeoPixel(Pin(21), 1)
        self.uart = UART(1, baudrate=9600, tx=Pin(3), rx=Pin(2))
        self.silent_mode = False
        pygame.mixer.init()

    def flash_led(self, color: str, duration_ms: int = 200):
        """Flashes an LED with specified color."""
        if not self.silent_mode:
            colors = {"green": (0, 255, 0), "red": (255, 0, 0), "yellow": (255, 255, 0)}
            self.led[0] = colors.get(color, (0, 0, 0))
            self.led.write()
            time.sleep_ms(duration_ms)
            self.led[0] = (0, 0, 0)
            self.led.write()

    def play_sound(self, index: int):
        """Plays an audio alert via UART (simulated)."""
        if not self.silent_mode:
            cmd = b'\x7E\xFF\x06\x03\x00\x00' + bytes([index]) + b'\xEF'
            self.uart.write(cmd)
            print(f"Playing audio index: {index}")

    def send_command(self, event: str):
        """Sends command via communication interface (simulated)."""
        if not self.silent_mode:
            print(f"Sending command: {event}")

class GestureClassifier:
    """Classifies gestures using a lightweight ML model."""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.history = []
        self.vocabulary = config.get("vocabulary", ["YES", "NO", "HELP"])

    def classify(self, emg_data: List[float], emergency: bool, touch: bool) -> Optional[str]:
        """Classifies gestures based on sensor inputs."""
        thresholds = self.config.get("thresholds", {})
        if emergency:
            return "HELP"
        
        self.history.append(emg_data)
        if len(self.history) > 10:
            self.history.pop(0)
        
        if len(self.history) == 10:
            cheek_threshold = thresholds.get("emg_cheek", 0.4)
            jaw_threshold = thresholds.get("emg_jaw", 0.45)
            if emg_data[0] > cheek_threshold and emg_data[0] < 1.5 * cheek_threshold:
                return "YES"
            if emg_data[2] > jaw_threshold and emg_data[2] < 1.5 * jaw_threshold:
                return "NO"
            if touch and thresholds.get("touch_active", True):
                return "INTERACTIVE"
        return None

class AssistiveDevice:
    """Main system for gesture-based communication."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.config = {
            "thresholds": {
                "emg_cheek": 0.4,
                "emg_jaw": 0.45,
                "touch_active": True
            },
            "vocabulary": ["YES", "NO", "HELP", "INTERACTIVE"]
        }
        self.emg = EMGReader(SPI(1, baudrate=1000000, sck=Pin(18), mosi=Pin(19), miso=Pin(20)), 17, 16)
        self.feedback = FeedbackController(self.config)
        self.classifier = GestureClassifier(self.config)
        self.last_activity = time.time()

    def calibrate(self) -> Dict[str, any]:
        """Calibrates sensor thresholds."""
        print("Calibrating sensors...")
        baseline_emg = []
        for _ in range(50):
            baseline_emg.append(self.emg.read())
            time.sleep(0.1)
        
        baseline_emg = [sum(ch) / len(baseline_emg) for ch in zip(*baseline_emg)]
        thresholds = {
            "emg_cheek": baseline_emg[0] * 1.2,
            "emg_jaw": baseline_emg[2] * 1.2,
            "touch_active": True
        }
        with open("thresholds.json", "w") as f:
            ujson.dump(thresholds, f)
        log_event("calibration", "success")
        return thresholds

    def run(self):
        """Main loop for gesture detection and feedback."""
        self.config["thresholds"] = self.calibrate()
        while True:
            emergency = False  # Simulated emergency check
            touch = bool(Pin(27, Pin.IN).value())
            emg_data = self.emg.read()
            gesture = self.classifier.classify(emg_data, emergency, touch)
            
            if gesture:
                self.feedback.flash_led("green" if gesture != "HELP" else "yellow")
                self.feedback.play_sound(self.config["vocabulary"].index(gesture) + 1)
                self.feedback.send_command(gesture)
                log_event("gesture", gesture)
                self.last_activity = time.time()
            
            if time.time() - self.last_activity > 15:
                self.emg.set_sample_rate(250)
                log_event("power", "sleep_mode")
                time.sleep(0.5)
            else:
                self.emg.set_sample_rate(1000)
                time.sleep(0.1)

# Example usage
if __name__ == "__main__":
    device = AssistiveDevice(user_id="user123")
    device.run()
```
