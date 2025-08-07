from typing import Sequence

import serial


class SerSystem:
    def __init__(self, port, baudrate=115200, timeout=0.1):
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
        with self.ser as ser:
            if ser.in_waiting > 0:
                line = ser.readline()
                data_str = line.decode("utf-8").strip()
                if data_str in ("y2", "y3", "y4", "y5"):
                    print(f"successfully receive {data_str}")
                    return data_str
                elif data_str == "n":
                    print(f"successfully receive {data_str}")
                    return data_str
                elif data_str:
                    # 接收到的不为空
                    print(f"receive unexpected data :{data_str}")
                    return None
