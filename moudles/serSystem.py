from typing import Sequence

import serial


class SerSystem:
    def __init__(self, port, baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        # 打开串口
        self.open()

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"Serial port {self.port} opened successfully.")
        except serial.SerialException as e:
            print(f"Error opening serial port {self.port}: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"Serial port {self.port} closed.")

    def sendPoints(self, data: Sequence):
        """
        data: Sequence  2个整数坐标
        发送 #开头， ：结尾的数据
        """
        if self.ser.is_open and len(data) == 4:
            payload = ",".join(map(str, data))
            message = f":{payload}#"
            self.ser.write(message.encode())
            print(f"Sent: {data}")
        else:
            print(f"data is not fit ,len is {len(data)}")

    def receiveSigns(self):
        if not self.ser.is_open:
            print("Serial port is not open.")
            return None
