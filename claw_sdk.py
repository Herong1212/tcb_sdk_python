"""
LebaiClaw SDK — 乐白夹爪 Modbus RTU 控制

供大模型（pi0 / smolVLA 等）调用的高层 API::

    claw = LebaiClaw()
    claw.init(force=50, speed=80)    # 上电初始化（找行程 + 设置参数）
    claw.open()                      # 张开
    claw.close()                     # 闭合
    state = claw.get_state()         # 返回观测 dict

底层协议: Modbus RTU, RS485, 115200-8N1, 从机地址 1
写指令使用功能码 0x10，读指令使用功能码 0x03。

寄存器地址:
    0x9C40  幅度控制 (写)  0-100
    0x9C41  力度控制 (写)  0-100
    0x9C45  当前位置 (读)  0-100
    0x9C46  当前力矩 (读)  0-100
    0x9C47  动作完成 (读)  1=完成, 0=执行中
    0x9C48  找行程   (写)  1
    0x9C49  未校准   (读)  1=未校准, 0=已校准
    0x9C4A  速度     (读写) 0-100
    0x9C4B  速度(断电保留) (读写) 0-100
    0x9C9A  自动找行程 (写) 1=关闭, 2=关闭+保存, 3=恢复+保存
"""

import time
import serial


class LebaiClaw:

    # 寄存器地址
    _REG_POSITION_SET = 0x9C40
    _REG_FORCE_SET = 0x9C41
    _REG_POSITION_NOW = 0x9C45
    _REG_TORQUE_NOW = 0x9C46
    _REG_CMD_DONE = 0x9C47
    _REG_FIND_STROKE = 0x9C48
    _REG_NOT_CALIBRATED = 0x9C49
    _REG_SPEED = 0x9C4A
    _REG_SPEED_SAVE = 0x9C4B
    _REG_AUTO_FIND = 0x9C9A

    def __init__(
        self,
        port: str,
        slave_id: int,
        baudrate: int,
        timeout: float,
        debug: bool,
    ):
        self._slave_id = slave_id
        self._debug = debug
        self._ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=1,
            timeout=timeout,
        )

    # ---------------------------------------------------------------- #
    #  Context manager
    # ---------------------------------------------------------------- #

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.disconnect()

    def disconnect(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    # ---------------------------------------------------------------- #
    #  * Modbus RTU 底层
    # ---------------------------------------------------------------- #

    @staticmethod
    def _crc16(data: bytes) -> bytes:
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
        return bytes([crc & 0xFF, (crc >> 8) & 0xFF])

    def _send(self, payload: bytes, read_len: int) -> bytes:
        frame = payload + self._crc16(payload)
        self._ser.reset_input_buffer()
        if self._debug:
            print("TX:", frame.hex(" "))
        self._ser.write(frame)
        self._ser.flush()
        time.sleep(0.05)
        resp = self._ser.read(read_len)
        if self._debug:
            print("RX:", resp.hex(" "))
        if len(resp) < read_len:
            raise TimeoutError(f"夹爪无响应，收到 {len(resp)}/{read_len} 字节")
        data, recv_crc = resp[:-2], resp[-2:]
        if recv_crc != self._crc16(data):
            raise ValueError(
                f"CRC 校验失败: recv={recv_crc.hex()} calc={self._crc16(data).hex()}"
            )
        return resp

    def _write(self, reg: int, value: int):
        payload = bytes(
            [
                self._slave_id,
                0x10,
                (reg >> 8) & 0xFF,
                reg & 0xFF,
                0x00,
                0x01,
                0x02,
                (value >> 8) & 0xFF,
                value & 0xFF,
            ]
        )
        self._send(payload, read_len=8)

    def _read(self, reg: int) -> int:
        payload = bytes(
            [
                self._slave_id,
                0x03,
                (reg >> 8) & 0xFF,
                reg & 0xFF,
                0x00,
                0x01,
            ]
        )
        resp = self._send(payload, read_len=7)
        return (resp[3] << 8) | resp[4]

    @staticmethod
    def _clamp(value: int, lo: int = 0, hi: int = 100) -> int:
        if not lo <= value <= hi:
            raise ValueError(f"参数 {value} 超出范围 [{lo}, {hi}]")
        return value

    # ---------------------------------------------------------------- #
    # * 接口 API
    # ---------------------------------------------------------------- #

    def init(self, force: int, speed: int) -> None:
        """上电初始化：关闭自动找行程 → 执行找行程 → 设置力度和速度。"""
        self._write(self._REG_AUTO_FIND, 1)
        self._write(self._REG_FIND_STROKE, 1)
        if not self.wait_done(timeout=10.0, interval=0.05):
            raise TimeoutError("找行程超时，请检查夹爪供电和接线")

        self.set_force(force)
        self.set_speed(speed, save=False)

    def open(self, wait: bool) -> None:
        """张开夹爪到最大（position=100）。"""
        self.move(100, wait=wait)

    def close(self, wait: bool) -> None:
        """完全闭合夹爪（position=0）。"""
        self.move(0, wait=wait)

    def move(self, position: int, wait: bool) -> None:
        """移动到指定开合位置（0-100）。"""
        self._write(self._REG_POSITION_SET, self._clamp(position))
        if wait:
            self.wait_done(timeout=5.0, interval=0.05)

    def set_force(self, force: int) -> None:
        """设置夹持力度（0-100），不触发运动。"""
        self._write(self._REG_FORCE_SET, self._clamp(force))

    def set_speed(self, speed: int, save: bool) -> None:
        """设置开合速度（0-100）。save=True 则断电保留。"""
        reg = self._REG_SPEED_SAVE if save else self._REG_SPEED
        self._write(reg, self._clamp(speed))

    def get_position(self) -> int:
        """读取当前开合位置（0-100）。"""
        return self._read(self._REG_POSITION_NOW)

    def get_torque(self) -> int:
        """读取当前输出力矩（0-100）。"""
        return self._read(self._REG_TORQUE_NOW)

    def is_done(self) -> bool:
        """当前动作是否已完成。"""
        return self._read(self._REG_CMD_DONE) == 1

    def is_calibrated(self) -> bool:
        """夹爪是否已完成找行程校准。"""
        return self._read(self._REG_NOT_CALIBRATED) == 0

    def wait_done(self, timeout: float, interval: float) -> bool:
        """
        阻塞等待当前动作完成。
        返回 True=完成，False=超时。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_done():
                return True
            time.sleep(interval)
        return False

    def get_state(self) -> dict:
        """
        返回夹爪完整状态字典，供大模型观测或记录。

        示例返回：
            {
                "position":   75,    # 当前开合位置 0-100
                "torque":     12,    # 当前力矩 0-100
                "is_done":    True,  # 动作是否完成
                "calibrated": True,  # 是否已校准
            }
        """
        return {
            "position": self.get_position(),
            "torque": self.get_torque(),
            "is_done": self.is_done(),
            "calibrated": self.is_calibrated(),
        }


# ------------------------------------------------------------------ #
# * 示例: 开合循环
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    with LebaiClaw(
        port="/dev/ttyCH341USB0",
        slave_id=1,
        baudrate=115200,
        timeout=1.0,
        debug=False,
    ) as claw:
        print("初始化...")

        claw.init(force=20, speed=50)
        print("状态:", claw.get_state())

        while True:
            print("\n--- 闭合 ---")
            claw.close(wait=True)
            print("状态:", claw.get_state())
            time.sleep(0.3)

            print("--- 张开 ---")
            claw.open(wait=True)
            print("状态:", claw.get_state())
            time.sleep(0.3)
