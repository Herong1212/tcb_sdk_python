#!/usr/bin/env python3
"""
协作七自由度机械臂SDK(2403版本)
文件名: robotarm_sdk.py
作者: 杜宇坤
创建时间: 2025-11-03
最后修改时间: 2026-01-26
"""

import glob
import json
import logging
import os
import re
import select
import socket
import struct
import threading
import time
from datetime import datetime

import colorlog
import numpy as np
import yaml
import zlib

from typing import Any, Dict, List, Optional, Tuple


class RobotArmSDK:
    def __init__(self, ip: str, port: int, log_level: int = 0, color_log: bool = True):
        """
        初始化SDK

        Args:
            ip(str): 机械臂 IP 地址
            port(int): 机械臂端口号
            log_level(int): 日志等级
                - 0: info
                - 1: debug
            color_log(bool): 是否使用彩色日志输出
                - True: 使用彩色日志(默认)
                - False: 使用无颜色日志

        Returns:
            None
        """
        # 将传入的机械臂IP地址保存到实例变量中，用于后续建立TCP连接
        self.ip = ip

        # 将传入的端口号保存到实例变量中，端口号决定了连接的是通用控制(6001)还是运动控制(7000)功能
        self.port = port

        # 初始化socket对象为None，后续在connect方法中会创建实际的TCP连接套接字
        # Optional[socket.socket]表示该变量要么是socket.socket类型的对象，要么是None
        self.socket: Optional[socket.socket] = None

        # self._connect_flag实际上是一个连接状态控制标志，用于控制接收线程的运行状态：
        #   1、当self._connect_flag为False时，表示连接正常，接收线程可以继续运行；
        #   2、当self._connect_flag为True时，表示需要断开连接，接收线程会检测到这个标志并退出；
        self._connect_flag = False

        # 字典类型变量，用于存储从机械臂控制器返回的状态信息，包括各种传感器数据、关节角度、错误码等实时状态
        self.return_status: Dict[str, Any] = {}

        # 接收线程对象，用于异步接收来自机械臂控制器的数据
        # 使用独立线程避免阻塞主线程的其他操作
        self._receive_thread: Optional[threading.Thread] = None

        # 线程锁对象，用于保护共享资源 return_status 的安全访问，防止多个线程同时修改状态数据造成竞争条件
        self._status_lock = threading.Lock()

        # 设置 NumPy 数组打印格式，suppress=True 抑制科学计数法显示，precision=4 保留4位小数。便于查看和调试关节角度、位置坐标等浮点数值
        np.set_printoptions(suppress=True, precision=4)

        # 延迟时间变量，单位秒，用于控制命令发送的频率，防止发送过快导致控制器处理不过来
        # ps 根据连接的端口不同，该值会在 connect() 方法中被调整
        self.delay_time = 0.01

        # 是否启用彩色日志的标志位，影响日志输出的视觉效果
        self.color_log = color_log

        # 获取当前SDK文件的绝对路径目录，用于构建配置文件的相对路径
        SDK_path = os.path.dirname(os.path.abspath(__file__))

        # 构建 constants.yaml 配置文件的完整路径，该文件包含通讯协议的命令字定义
        self.constants_path = SDK_path + "/config/constants.yaml"

        # 加载通讯协议常量，解析 constants.yaml 文件内容到内存中，供后续命令发送使用
        self.CONSTANTS = self._load_constants(self.constants_path)

        # 构建日志文件存储目录的路径，用于保存运行过程中的日志信息
        self.log_path = SDK_path + "/log"

        # 构建 TCB 机械臂配置文件的完整路径，该文件包含机械臂的自由度等基本信息
        self.tcb_config_path = SDK_path + "/config/tcb_cfg.yaml"

        # 以 UTF-8 编码方式打开并读取 TCB 配置文件，解析 YAML 格式内容到字典对象
        with open(self.tcb_config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # 从配置文件中提取机械臂自由度信息，并保存到实例变量中
        self.dof = self.cfg["DOF"]

        # 根据日志级别参数设置相应的日志记录器
        if log_level == 0:
            # 设置 INFO 级别的日志记录，记录一般性信息和警告
            self._setup_logger(logging.INFO)
            self.logger.info(f"初始化TCB SDK(24.03版本), 日志等级为 INFO")
        elif log_level == 1:
            # 设置 DEBUG 级别的日志记录，记录详细的调试信息
            self._setup_logger(logging.DEBUG)
            self.logger.info(f"初始化TCB SDK(24.03版本), 日志等级为 DEBUG")
        else:
            # 如果日志级别参数不在有效范围内，则记录错误信息
            self.logger.error(f"日志等级错误, 检查是否为 0(INFO) 或 1(DEBUG)")

        # 验证端口号是否为预定义的有效端口（6001 或 7000）
        if self.port == 6001 or self.port == 7000:
            # 记录连接信息日志，确认IP地址和端口号
            self.logger.info(f"连接ip: {self.ip}, 连接port: {self.port}")
        else:
            # 如果端口号不是有效的 6001 或 7000，则记录错误信息
            self.logger.error(f"连接port错误, 检查是否为 6001 或者 7000")

    # note 设置日志记录器，创建日志目录、配置控制台和文件输出格式，并管理日志文件数量
    def _setup_logger(self, log_level: int) -> None:
        os.makedirs(self.log_path, exist_ok=True)

        logger_name = f"TCB_{self.dof}dof_2403_{self.port}"

        # 使用统一的时间戳格式（年月日_小时）
        today_str = datetime.now().strftime("%Y-%m-%d_%H")
        log_file_path = os.path.join(self.log_path, f"{today_str}.log")

        if self.color_log:
            log_colors = {
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }

            formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                log_colors=log_colors,
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
        else:
            self.logger.propagate = False

        # 删除多余日志文件（保留最近 50 个）
        log_files = sorted(
            glob.glob(os.path.join(self.log_path, "*.log")), key=os.path.getmtime
        )
        if len(log_files) > 50:
            for old_file in log_files[:-50]:
                try:
                    os.remove(old_file)
                except Exception as e:
                    self.logger.warning(f"删除旧日志文件失败: {old_file}, 错误: {e}")

    # NOTE python 特殊方法，实现了上下文管理器协议（context manager protocol），用于支持 with 语句
    def __enter__(self):
        """
        支持 with 上下文管理器
        1、在 with 语句开始时被自动调用；
        2、负责建立与机械臂的连接（调用 connect() 方法）；
        3、返回自身实例，使得可以在 with 块中使用这个实例；
        """

        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出时自动断开连接

        1、在 with 语句结束时被自动调用（无论是否发生异常）
        2、负责断开与机械臂的连接（调用 disconnect() 方法）
        """
        self.disconnect()

    # ? 为什么要使用上下文管理器而不是直接调用 connect()？
    #   1、资源管理的自动化：使用 with 语句可以确保即使在代码执行过程中发生异常，连接也会被自动断开。这避免了忘记手动调用 disconnect() 导致的资源泄露问题。
    #   2、异常安全性：如果在使用机械臂的过程中发生异常，exit 方法仍会被调用，从而确保连接被正确关闭。
    #   3、代码简洁性：使用 with 语句让代码更清晰地表达了"获取资源 → 使用资源 → 释放资源"的意图。

    # NOTE 连接机械臂
    def connect(self, retries: int = 3, delay: float = 2.0) -> bool:
        """
        连接到机械臂设备

        尝试通过 socket 连接到指定IP地址和端口的机械臂设备，支持多次重试机制。
        根据不同端口设置不同的延迟时间，并启动接收线程处理数据。

        Args:
            retries (int): 连接尝试次数，默认为 3 次
            delay (float): 每次重连之间的延迟时间（秒），默认为 2.0 秒

        Returns:
            bool: 连接成功返回 True，否则返回 False
        """
        # 尝试连接到机械臂设备，最多重试指定次数
        for attempt in range(1, retries + 1):
            try:
                # 创建 socket 连接并设置超时时间
                self.socket = socket.socket()
                self.socket.settimeout(5)  # 5秒
                self.socket.connect((self.ip, self.port))
                self._connect_flag = False
                # 启动接收数据线程
                self._start_receive_thread()
                self.logger.info(f"成功连接到 {self.ip}:{self.port}")

                # 根据端口号设置不同的延迟时间
                # case1 7000端口用于连续的运动控制，需要快速发送指令，因此使用固定的最短延迟
                if self.port == 7000:
                    self.delay_time = 0.001
                elif self.port == 6001:
                    # case2 6001端口用于配置和状态查询，其频率不需要太高，根据控制器的实际通讯周期来设置可以避免过度占用网络带宽和控制器资源
                    self.logger.info(f"当前SDK支持 {self.dof} 自由度的 TCB 机械臂")
                    # 查询控制周期并设置延迟时间
                    self.delay_time = self.control_cycle_inquire() / 1000

                self.return_status = {}
                return True

            except socket.error as e:
                self.logger.warning(f"第 {attempt} 次连接失败: {e}")
                # 如果不是最后一次尝试，则等待一段时间后重试
                if attempt < retries:
                    self.logger.info(f"{delay} 秒后重试连接...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"连接失败, 已重试 {retries} 次仍未成功")
                    return False

    # NOTE 断开连接
    def disconnect(self) -> bool:
        """
        断开连接并清理资源

        设置连接标志为 True，通知接收线程停止运行；
        在 _receive_loop() 中会检测这个标志并退出循环；
        """
        self._connect_flag = True

        # 获取当前线程对象，用于判断是否在接收线程内部调用此方法
        current_thread = threading.current_thread()

        try:
            # 检查是否存在socket连接
            if self.socket:
                # 关闭socket连接，释放网络资源
                self.socket.close()
                # 将socket引用设为None，避免后续意外使用
                self.socket = None
        except Exception as e:
            # 记录关闭socket时发生的错误，并返回False表示断开连接失败
            self.logger.error(f"关闭socket时出错: {e}")
            return False

        # 检查是否存在接收线程，并且当前线程不是接收线程，避免在接收线程内部尝试join自己而导致死锁
        if self._receive_thread and current_thread != self._receive_thread:
            # 检查接收线程是否仍在运行
            if self._receive_thread.is_alive():
                # 等待接收线程结束，最多等待2秒
                self._receive_thread.join(timeout=2)
                # 再次检查接收线程是否仍然存活
                if self._receive_thread.is_alive():
                    # 如果线程在超时时间内仍未结束，记录警告信息并返回 False
                    self.logger.warning("接收线程未能在超时时间内停止")
                    return False

        # 延迟1秒，确保所有资源完全释放
        time.sleep(1)
        # 记录连接断开成功的日志信息
        self.logger.info(f"{self.ip}:{self.port} 连接已断开")
        # 返回 True 表示断开连接成功
        return True

    # NOTE 重新连接
    def reconnect(self, retries=3, delay=2.0) -> bool:
        """自动断开后重连"""
        self.logger.warning("尝试重新连接控制器...")
        self.disconnect()
        return self.connect(retries=retries, delay=delay)

    def _start_receive_thread(self) -> None:
        """启动接收线程"""
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()

        if self._receive_thread and self._receive_thread.is_alive():
            self.logger.info("接收线程已启动")
            return

    def _decode_robot_data(self, raw_data: bytes) -> Tuple[List[str], str]:
        cmd_word = f"{raw_data[4]:02X}{raw_data[5]:02X}"
        match = re.search(b"\{.*?\}", raw_data[6:])
        if match:
            json_str = match.group()
            try:
                decoded_messages = [json.loads(json_str)]
            except json.JSONDecodeError:
                match = re.search(b"\{.*\}", raw_data)
                json_str = match.group()

            decoded_messages = [json_str]

        return decoded_messages, cmd_word

    def _receive_loop(self) -> None:
        while True:
            if self._connect_flag or not self.socket:
                break
            try:
                while not self._connect_flag and self.socket:
                    try:
                        readable, _, _ = select.select([self.socket], [], [], 1)
                        if readable:
                            data = self.socket.recv(2048)

                            self.logger.debug(f"收到的原始数据: {data}")
                            if data:
                                result, cmd_word = self._decode_robot_data(data)
                                for i in range(0, len(result)):
                                    self._parse_status(result[i], cmd_word)
                    except (OSError, ConnectionAbortedError):
                        pass
            except Exception as e:
                # self.logger.error(f"接收线程异常: {e}")
                pass

    def _crc(self, data_to: bytes, command: bytes, data_segment: str) -> bytes:
        length_bytes = struct.pack(">H", len(data_segment))
        crc32 = zlib.crc32(length_bytes + command + data_segment)
        crc_bytes = struct.pack(">I", crc32)
        return data_to + length_bytes + command + data_segment + crc_bytes

    def _send_command(self, cmd_word: bytes, cmd_data: str = None) -> None:
        if not self.socket:
            self.logger.error("未建立有效连接")
            raise ConnectionError("未建立有效连接")

        data_to_send = self._crc(
            self.CONSTANTS["FRAME_HEADER"], cmd_word, json.dumps(cmd_data).encode("GBK")
        )
        self.logger.debug(f"发送给控制器的数据为: {data_to_send}")

        try:
            self.socket.send(data_to_send)
            time.sleep(self.delay_time)
        except (BrokenPipeError, ConnectionResetError) as e:
            self.logger.error(f"命令发送失败: {e}, 尝试重连")
            if self.reconnect():
                try:
                    self.socket.send(data_to_send)
                    time.sleep(self.delay_time)
                    self.logger.info("命令在重连后成功发送")
                except Exception as e2:
                    self.logger.error(f"重连后发送失败: {e2}")
            else:
                self.logger.error("重连失败, 命令丢失")
                self.reconnect()
                return

    def _parse_status(self, data: bytes, cmd_word) -> None:
        with self._status_lock:
            json_str = data.decode("utf-8").strip()
            json_str_fixed = re.sub(r"\b(nan|NaN)\b", "null", json_str)
            json_str_fixed = re.sub(r"{21", "", json_str)

            try:
                parsed = json.loads(json_str_fixed)
                for key, value in parsed.items():
                    status_key = cmd_word + key
                    self.return_status[status_key] = value

                self.logger.debug(f"状态更新: {self.return_status}\n")

            except json.JSONDecodeError:
                self.logger.warning(
                    f"部分解析失败（但不影响执行）, 原始数据: {json_str}"
                )
                return

    def _load_constants(self, file_path: str) -> bytes:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        def convert_hex_to_bytes(obj):
            if isinstance(obj, dict):
                return {k: convert_hex_to_bytes(v) for k, v in obj.items()}
            elif isinstance(obj, str):
                return bytes.fromhex(obj)
            else:
                return obj

        return convert_hex_to_bytes(data)

    def _return_get(
        self, inquiry_key_name: str, key_value_clear: bool = True, timeout: float = 0.5
    ) -> Optional[Any]:
        start = time.time()

        while True:
            key_value = self.return_status.get(inquiry_key_name, None)

            if key_value is not None:
                self.logger.debug(f"获取到 {inquiry_key_name} 对应的键值: {key_value}")
                break

            if time.time() - start > timeout:
                self.logger.debug(f"获取 {inquiry_key_name} 对应的键值超时")
                break

            time.sleep(0.05)

        if key_value_clear:
            self.logger.debug(f"清除 {inquiry_key_name} 对应的键值")
            self.return_status[inquiry_key_name] = None

        return key_value

    ########################################
    # NOTE API 接口（6001 端口）
    ########################################
    def heartbeat(self) -> bool:
        """
        心跳包

        Args:
            None

        Returns:
            bool: 心跳包状态标志
                - True:  心跳包正常
                - False: 心跳包异常
        """
        tolerance = 1

        timestamp = round(time.time(), 6)
        self._send_command(self.CONSTANTS["HEARTBEAT"], {"time": timestamp})
        self.logger.debug(f"发送心跳包的时间戳: {timestamp}")

        heartbeat_return = self._return_get("1201time", key_value_clear=False)
        time.sleep(self.delay_time)
        self.logger.debug(f"心跳包返回的时间戳: {heartbeat_return}")

        if heartbeat_return is None:
            self.logger.error("心跳包异常, 返回为空")
            return False

        if abs(heartbeat_return - timestamp) <= tolerance:
            self.logger.debug("心跳包正常")
            return True
        else:
            self.logger.error("心跳包异常")
            return False

    def robot_stop(self, robot: int = 1) -> None:
        """
        紧急停止

        Args:
            robot: 机器人号码, 默认为 1

        Returns:
            None
        """
        self._send_command(self.CONSTANTS["ROBOT_STOP"], {"robot": robot})
        self.logger.info(f"机器人 紧急停止！")

    def control_cycle_set(self, controlCycle: int) -> None:
        """
        设置控制器通讯周期

        Args:
            controlCycle: 参数为 1、2、4、8 毫秒(ms), 控制器重启生效

        Returns:
            None
        """
        if controlCycle not in [1, 2, 4, 8]:
            self.logger.warning(
                f"控制器通讯周期 只能为 1、2、4、8 毫秒之一, 当前为: {controlCycle}"
            )
        else:
            cmd_data = {
                "controlCycle": controlCycle,
                "baudRate": "500K",  # 波特率
                "bustype": 1,
                "control_word": 7,  # 伺服控制字
                "pdo_lost_tolerance": 2,  # 丢帧容差
            }
            self._send_command(self.CONSTANTS["CONTROL_CYCLE"]["SET"], cmd_data)
            time.sleep(0.5)
            self.logger.info(f"机器人通讯周期设置为: {controlCycle}")
            self.logger.info(f"机器人通讯周期设置后, 控制器重启生效")

    def control_cycle_inquire(self) -> int:
        """
        查询控制器通讯周期

        Args:
            None

        Returns:
            int: 控制器通讯周期, 单位 毫秒(ms)
        """
        self._send_command(self.CONSTANTS["CONTROL_CYCLE"]["INQUIRE"])
        controlCycle = self._return_get("2022controlCycle")
        self.logger.info(f"机器人通讯周期为 {controlCycle} ms")  # 4ms
        return controlCycle

    def controller_init_finish_inquire(self) -> bool:
        """
        控制器初始化是否完成

        Args:
            None

        Returns:
            bool:控制器初始化状态
                - True: 控制器初始化完成
                - False: 控制器初始化未完成
        """
        self._send_command(self.CONSTANTS["CONTROLLER_INIT_FINISH"])
        finishinit = self._return_get("1001finishinit")
        if finishinit == True:
            self.logger.info(f"控制器初始化 完成")
        else:
            self.logger.info(f"控制器初始化 未完成")
        return finishinit

    def T5_system_inquire(self) -> bool:
        """
        查询当前控制器是否为 T5 系统

        Args:
            None

        Returns:
            bool:控制器是否为 T5 系统
                - True: 当前控制器为 T5 系统
                - False: 当前控制器为 非 T5 系统
        """
        self._send_command(self.CONSTANTS["T5_SYSTEM_INQUIRE"])
        T5_system = self._return_get("1003T5_system")
        if T5_system == True:
            self.logger.info(f"当前控制器为 T5 系统")
        else:
            self.logger.info(f"当前控制器为 非 T5 系统")
        return T5_system

    def hardware_system_version_inquire(self) -> str:
        """
        控制器硬件系统发布版本查询

        Args:
            None

        Returns:
            str: 硬件系统发布版本
        """
        self._send_command(self.CONSTANTS["HARDWARE_SYSTEM_VERSION"])
        hardware_system_version = self._return_get("1005version")
        if self._return_get("1005result"):
            self.logger.info(f"硬件系统发布版本: {hardware_system_version}")
            return hardware_system_version
        else:
            self.logger.info(f"硬件系统发布版本(获取失败)")
            return None

    def system_version_inquire(self, version: str) -> dict:
        """
        查询当前系统版本号

        Args:
            version (str): 版本号内容, 传入查询的版本号字符串

        Returns:
            dict: 系统版本信息字典, 包含以下字段:
                - version (str): 当前系统版本号
                - rtlVersion (str): 控制器版本"YY.MM.DD"
                - jobFileVersion (str): 作业文件版本
                - configFileVersionMismatch (bool): 配置文件是否匹配
                - sysClock (str): 系统时钟, 格式为"YYYY.MM.DD HH:MM:SS"
        """
        self._send_command(
            self.CONSTANTS["SYSTEM_VERSION_INQUIRE"], {"version": version}
        )
        self.logger.info(f"版本号内容: {version}")
        result = {
            "configFileVersionMismatch": self._return_get(
                "1011configFileVersionMismatch"
            ),
            "jobFileVersion": self._return_get("1011jobFileVersion"),
            "rtlVersion": self._return_get("1011rtlVersion"),
            "sysClock": self._return_get("1011sysClock"),
            "version": self._return_get("1011version"),
        }
        self.logger.info(f"版本查询结果: {result}")
        return result

    def dump_log_file_inquire(self) -> dict:
        """
        上位机查询崩溃日志文件

        Args:
            None
        Returns:
            dict: 崩溃日志文件查询结果, 包括以下字段:
                - absolutepath (str): 路径
                - dumpLogfilenum (int): 文件数量
                - dumpLogfilelist (str): 日志文件列表
        """
        self._send_command(self.CONSTANTS["DUMP_LOG_FILE_INQUIRE"])
        self.logger.info(f"查询崩溃日志文件")
        result = {
            "absolutepath": self._return_get("1021absolutepath"),
            "dumpLogfilenum": self._return_get("1021dumpLogfilenum"),
            "dumpLogfilelist": self._return_get("1021dumpLogfilelist"),
        }
        self.logger.info(f"崩溃日志文件查询结果: {result}")
        return result

    def log_file_inquire(self, num: int = 5) -> dict:
        """
        查询日志文件

        Args:
            num (int): 表示获取最近多少个文件, 可取5、30、100

        Returns:
            dict: 日志文件查询结果, 包括以下字段:
                - absolutepath (str): 日志所在目录
                - logfilenum (int): 日志数目, 这个数和上面的num不一定相等
                - logfilelist (str): 日志文件列表
        """
        self._send_command(self.CONSTANTS["LOG_FILE_INQUIRE"], {"num": num})
        self.logger.info(f"查询日志文件")
        result = {
            "absolutepath": self._return_get("1023absolutepath"),
            "logfilenum": self._return_get("1023logfilenum"),
            "logfilelist": self._return_get("1023logfilelist"),
        }
        self.logger.info(f"日志文件查询结果: {result}")
        return result

    def reboot_controller(self) -> None:
        """
        控制器重启

        Args:
            None

        Returns:
            None
        """
        self._send_command(self.CONSTANTS["REBOOT_CONTROLLER"])
        self.logger.info(f"控制器 重启")
        return None

    def shutdown_controller(self) -> None:
        """
        控制器关机

        Args:
            None

        Returns:
            None
        """
        self._send_command(self.CONSTANTS["SHUTDOWN_CONTROLLER"])
        self.logger.info(f"控制器 关机")
        return None

    def controller_ip_set(
        self, name: str, address: str, gateway: str = "", dns: str = ""
    ) -> None:
        """
        设置控制器ip

        Args:
            name(str): 网口名称
            address(str): ip地址
            gateway(str): 网关
            dns(str): DNS

        Returns:
            None
        """
        cmd_data = {"name": name, "address": address, "gateway": gateway, "dns": dns}
        self._send_command(self.CONSTANTS["CONTROLLER_IP"]["SET"], cmd_data)
        time.sleep(0.5)
        self.logger.info(
            f"修改控制器网口 {name} 的ip为 {address}, dns为 {dns}, gateway为 {gateway} "
        )
        self.logger.info(f"修改后控制器自动重启(ip修改前后相同不重启)")
        return None

    def controller_ip_inquire(self) -> dict:
        """
        查询控制器ip

        Args:
            None

        Returns:
            dict: 网口信息, 包含以下字段:
                - num(int): 网口数量
                - network(list): 每个网口的 ip、dns、gateway 以及网口名称
            返回 network(list) 数据示例:
                [
                    {
                        "name": "eth0",
                        "address": "192.168.1.13",
                        "gateway": "192.168.1.1",
                        "dns": "114.114.114.114"
                    },
                    {
                        "name": "eth1",
                        "address": "192.168.1.14",
                        "gateway": "192.168.1.1",
                        "dns": "114.114.114.114"
                    }
                ]
        """
        self._send_command(self.CONSTANTS["CONTROLLER_IP"]["INQUIRE"])
        num = self._return_get("1042num")
        network = self._return_get("1042network")
        self.logger.info(f"查询控制器IP")
        self.logger.info(f"> 网络ip数量: {num}")

        for number in range(0, num):
            address = network[number]["address"]
            eth = network[number]["name"]
            dns = network[number]["dns"]
            gateway = network[number]["gateway"]
            self.logger.info(f"> 网口 {eth}: {address}, dns: {dns}, gateway: {gateway}")

        return {"num": num, "network": network}

    def config_file_inquire(self, isExport: bool) -> dict:
        """
        查询控制器配置文件目录
        Args:
            isExport(bool): 是否为导出配置文件
                - True: 查询导出配置文件
                - False: 查询导入配置文件
        Returns:
            dict: 配置文件查询结果, 包括以下字段:
                - file_num (int): 配置文件数量
                - file_list (list): 配置文件列表
        """
        self._send_command(
            self.CONSTANTS["CONFIG_FILE_INQUIRE"], {"isExport": isExport}
        )
        file_list = self._return_get("1071filelist")
        file_num = self._return_get("1071filenum")
        if isExport == True:
            self.logger.info(f"查询导入配置文件")
            self.logger.info(f"> 配置文件数量: {file_num}")
            self.logger.info(f"> 配置文件查询结果: {file_list}")
        else:
            self.logger.info(f"查询导出配置文件")
            self.logger.info(f"> 配置文件数量: {file_num}")
            self.logger.info(f"> 配置文件查询结果: {file_list}")
        return {"file_num": file_num, "file_list": file_list}

    def fault_reset(self, robot: int = 1) -> bool:
        """
        清除伺服错误

        Args:
            robot: 机器人号码, 默认为 1

        Returns:
            bool: 清错状态标志
                - True:  清错成功
                - False: 清错失败 (通过示教器查看无法清除的错误类型, 告知专业人员处理)
        """
        self._send_command(self.CONSTANTS["FAULT_RESET"], {"robot": robot})
        clearErrorFlag = self._return_get("1101clearErrorFlag")
        if clearErrorFlag == True:
            self.logger.info(f"清除伺服错误 成功")
        else:
            self.logger.info(f"清除伺服错误 失败")
        return clearErrorFlag

    def servo_status_set(self, state: int, robot: int = 1) -> int:
        """
        设置伺服状态

        Args:
            robot(int): 机器人号码
            status(int): 需要设置的伺服状态
                - 0: 停⽌
                - 1: 就绪

        Returns:
            int: 设置后的伺服状态
                - 0: 停⽌
                - 1: 就绪
                - 2: 错误
                - 3: 运⾏
        """
        if state not in [0, 1]:
            self.logger.warning(
                f"伺服状态设置参数只能为0(停止)、1(就绪), 当前为: {state}"
            )
            return

        self._send_command(
            self.CONSTANTS["SERVO_COMMANDS"]["SET"], {"robot": robot, "state": state}
        )
        state_map = {0: "伺服 停止", 1: "伺服 就绪", 2: "伺服 错误", 3: "伺服 运行"}
        self.logger.info(
            f"设置伺服状态: {state_map.get(self._return_get('3002state'), '未知状态')}"
        )
        return state

    # NOTE 伺服状态查询
    def servo_status_inquire(self, robot: int = 1) -> int:
        """
        查询伺服状态

        Args:
            robot(int): 机器人号码

        Returns:
            int: 机器人当前伺服状态
                - 0: 停⽌
                - 1: 就绪
                - 2: 错误
                - 3: 运⾏
        """
        self._send_command(
            self.CONSTANTS["SERVO_COMMANDS"]["INQUIRE"], {"robot": robot}
        )
        state_return = self._return_get("3002state")
        status_map = {0: "伺服 停止", 1: "伺服 就绪", 2: "伺服 错误", 3: "伺服 运行"}
        self.logger.info(f"查询伺服状态: {status_map.get(state_return, '未知状态')}")
        return state_return

    def servo_connect_inquire(self) -> int:
        """
        查询伺服连接状态

        Args:
            None

        Returns:
            int: 伺服连接状态
                - 0: 真实伺服
                - 1: 虚拟伺服
                - 2: 无伺服
        """
        self._send_command(self.CONSTANTS["SERVO_CONNECT_INQUIRE"])
        servoType = self._return_get("3005servoType")
        if servoType == 0:
            self.logger.info(f"伺服连接状态: 真实伺服")
        elif servoType == 1:
            self.logger.info(f"伺服连接状态: 虚拟伺服")
        elif servoType == 2:
            self.logger.warning(f"伺服连接状态: 无伺服")
        return servoType

    # NOTE 设置机械臂的操作模式，包括示教模式、运行模式和远程模式
    def operation_mode_set(self, mode: int) -> int:
        """
        设置操作模式状态

        Args:
            mode(int):
                - 0: ⽰教模式(Teach)
                - 1: 远程模式(Circle)
                - 2: 运⾏模式(Repeat)

        Returns:
            int: 操作模式状态
                - 0: ⽰教模式(Teach)
                - 1: 远程模式(Circle)
                - 2: 运⾏模式(Repeat)
        """
        if mode not in [0, 1, 2]:
            self.logger.warning(
                f"操作模式状态设置参数应该为 0(示教)、1(远程)、2(运行)之一, 当前为: {mode}"
            )
            return

        self._send_command(self.CONSTANTS["OPERATION_MODE"]["SET"], {"mode": mode})
        mode_map = {
            0: "⽰教模式(Teach)",
            1: "远程模式(Circle)",
            2: "运⾏模式(Repeat)",
        }
        mode_return = self._return_get("3012mode")
        self.logger.info(f"设置操作模式: {mode_map.get(mode_return, '未知模式')}")
        return mode_return

    # note 查询机械臂的操作模式，包括示教模式、运行模式和远程模式
    def operation_mode_inquire(self) -> int:
        """
        查询操作模式状态

        Args:
            None

        Returns:
            int: 操作模式状态
                - 0: ⽰教模式(Teach)
                - 1: 远程模式(Circle)
                - 2: 运⾏模式(Repeat)
        """
        self._send_command(self.CONSTANTS["OPERATION_MODE"]["INQUIRE"])
        mode_map = {
            0: "⽰教模式(Teach)",
            1: "远程模式(Circle)",
            2: "运⾏模式(Repeat)",
        }
        mode_return = self._return_get("3012mode")
        self.logger.info(f"查询操作模式: {mode_map.get(mode_return, '未知模式')}")
        return mode_return

    def enable_status_set(self, deadman: int, deadmanmode: int = 0) -> int:
        """
        上位机设置使能状态

        Args:
            deadman(int):
                - 0:下使能
                - 1:上使能
            deadmanmode(int):
                - 0: 软件触发(默认)
                - 1: 硬件触发

        Returns:
            int: 使能状态
                - 0:下使能
                - 1:上使能
        """
        self._send_command(
            self.CONSTANTS["ENABLE_STATUS"]["SET"],
            {"deadman": deadman, "deadmanmode": deadmanmode},
        )
        deadman_map = {0: "下使能", 1: "上使能"}
        deadman_return = self._return_get("3032deadman")
        self.logger.info(f"设置使能状态: {deadman_map.get(deadman_return, '未知状态')}")
        return deadman_return

    def enable_status_inquire(self) -> int:
        """
        上位机查询使能状态

        Args:
            deadman(int):
                - 0:下使能
                - 1:上使能
            deadmanmode(int):
                - 0: 软件触发(默认)
                - 1: 硬件触发

        Returns:
            int: 使能状态
                - 0:下使能
                - 1:上使能
        """
        self._send_command(self.CONSTANTS["ENABLE_STATUS"]["INQUIRE"])
        deadman_map = {0: "下使能", 1: "上使能"}
        deadman_return = self._return_get("3032deadman")
        self.logger.info(f"查询使能状态: {deadman_map.get(deadman_return, '未知状态')}")
        return deadman_return

    def coord_mode_set(self, coord: int, robot: int = 1) -> int:
        """
        设置坐标模式状态

        Args:
            robot(int): 机器人号码
            coord(int): 坐标模式状态
                - 0: 关节坐标(Joint)
                - 1: 直角坐标(Cart)
                - 2: 工具坐标(Tool)
                - 3: 用户坐标(User)
                - 5: 大地坐标(Ground)

        Returns:
            int: 坐标模式状态
                - 0: 关节坐标(Joint)
                - 1: 直角坐标(Cart)
                - 2: 工具坐标(Tool)
                - 3: 用户坐标(User)
                - 5: 大地坐标(Ground)
        """
        if coord not in [0, 1, 2, 3, 5]:
            self.logger.warning(
                f"坐标模式状态设置参数只能为0(Joint)、1(Cart)、2(Tool)、3(User)、5(Ground)之一, 当前为: {coord}"
            )
            return
        else:
            self._send_command(
                self.CONSTANTS["COORD_MODE"]["SET"],
                {"coordinate": coord, "robot": robot},
            )
            coord_map = {
                0: "关节坐标(Joint)",
                1: "直角坐标(Cart)",
                2: "工具坐标(Tool)",
                3: "用户坐标(User)",
                5: "大地坐标(Ground)",
            }
            coord_return = self._return_get("3042coordinate")
            self.logger.info(f"设置坐标模式为: {coord_map.get(coord_return)}")
            return coord_return

    def coord_mode_inquire(self, robot: int = 1) -> int:
        """
        查询坐标模式状态

        Args:
            robot(int): 机器人号码

        Returns:
            int: 坐标模式状态
                - 0: 关节坐标(Joint)
                - 1: 直角坐标(Cart)
                - 2: 工具坐标(Tool)
                - 3: 用户坐标(User)
        """
        self._send_command(self.CONSTANTS["COORD_MODE"]["INQUIRE"], {"robot": robot})
        coord = self._return_get("3042coordinate")
        coord_map = {
            0: "关节坐标(Joint)",
            1: "直角坐标(Cart)",
            2: "工具坐标(Tool)",
            3: "用户坐标(User)",
            5: "大地坐标(Ground)",
        }
        self.logger.info(f"查询当前坐标模式为: {coord_map.get(coord)}")
        return coord

    # NOTE 机械臂全局速度设置
    def speed_set(
        self,
        speed: int,
        robot: int = 1,
        speed_type: int = 0,
        mocroDotSpeedACS: float = 0.001,
        mocroDotSpeedMCS: float = 0.01,
    ) -> int:
        """
        设置全局速度

        Args:
            robot(int): 机器人号码
            speed(int): 全局速度设置
                - 1-100: 速度值, 百分比
            speed_type(int): 当前速度模式
                - 0: 点动速度
                - 1: 关节坐标定距移动
                - 2: 直角坐标定距移动
            mocroDotSpeedACS(float): 关节坐标定距移动, 仅当前点动坐标为关节坐标时使用
                - (0,2000]°
            mocroDotSpeedMCS(float): 直角坐标定距移动, 当前点动坐标不为关节坐标时使用
                - (0,2000]mm

        Returns:
            int: 控制器返回设置后的速度值
                - 1-100: 速度值, 百分比
        """
        if not 0 < speed <= 100:
            self.logger.warning(f"速度值必须在1 ~ 100之间, 当前为: {speed}")
            return
        else:
            cmd_data = {
                "robot": robot,
                "speed": speed,
                "type": speed_type,
                "mocroDotSpeedACS": mocroDotSpeedACS,
                "mocroDotSpeedMCS": mocroDotSpeedMCS,
            }
            self._send_command(self.CONSTANTS["SPEED_CONTROL"]["SET"], cmd_data)
            speed_return = self._return_get("3052speed")
            self.logger.info(f"设置全局速度: {speed_return}%")
            return speed_return

    # NOTE 机械臂全局速度查询
    def speed_inquire(self, robot: int = 1) -> int:
        """
        查询全局速度

        Args:
            robot(int): 机器人号码

        Returns:
            int: 控制器返回当前的速度值
                - 1-100: 速度值, 百分比
        """
        self._send_command(self.CONSTANTS["SPEED_CONTROL"]["INQUIRE"], {"robot": robot})
        speed = self._return_get("3052speed")
        self.logger.info(f"查询全局速度: {speed}%")
        return speed

    # NOTE 机械臂回零点
    def go_home(self, robot: int = 1, isWithExternal: int = 0) -> None:
        """
        机器人回原点
        (需要示教模式下，将伺服改为使能状态, 否则发送命令机器人也不会动作)

        Args:
            robot(int): 机器人号码
            isWithExternal(int): 是否带外部轴
                - 0: 不带外部轴
                - 1: 带外部轴

        Returns:
            None
        """
        # 根据配置文件 tcb_cfg.yaml 中定义的原点位置参数
        self._send_command(
            self.CONSTANTS["GO_HOME"],
            {"robot": robot, "isWithExternal": isWithExternal},
        )

        time.sleep(1)

        self.logger.info(f"机器人回原点")

        return None

    # NOTE 关节运动
    def movj(
        self,
        vel: int,
        coord: int,
        pos: List[float],
        block: int = 1,
        robot: int = 1,
        timeout: int = 30,
        tolerance: float = 0.05,
    ) -> bool:
        """
        机器人关节运动 MOVJ

        Args:
            robot(int): 机器人号码
            vel(int): 速度百分⽐, 1-100 的整数
            coord(int): 坐标模式
                - 0:  关节坐标(Joint)
                - 1:  直角坐标(Cart)
                - 2:  工具坐标(Tool)
                - 3:  用户坐标(User)
            pos(List[float]): [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7]
                - 关节坐标分别代表关节角度值
                - 直角坐标系分别 x, y, z, a, b, c 其中欧拉角的旋转顺序为'zyx',

                  工具、用户同直角直角坐标系下第七位参数默认为 0 即可
            block(int):
                - 0: 非阻塞模式
                - 1: 阻塞模式
            timeout(int): 阻塞模式下的超时时间, 单位为秒
            tolerance(float): 误差范围, 单位为度

        Returns:
            bool: 返回机器人是否移动成功
                - True: 移动成功
                - False: 移动失败
        """
        if coord not in {0, 1, 2, 3}:
            self.logger.warning(
                f"坐标参数应在0(关节坐标)、1(直角坐标)、2(工具坐标)、3(用户坐标)之一, 当前为: {coord}"
            )
            return False

        if vel > 100 or vel < 1:
            self.logger.warning(f"速度百分比应在 1~100 之间, 当前为: {vel}")
            return False

        if block not in {0, 1}:
            self.logger.warning(
                f"block 参数应在 0(非阻塞模式)、1(阻塞模式)之一, 当前为: {block}"
            )
            return False

        if len(pos) not in {6, 7}:
            self.logger.warning(f"pos 参数长度应为 6 或 7, 当前为: {len(pos)}")
            return False

        # 获取机械臂当前位置
        current_pos = self.currentpos_inquiry(coord)
        # 如果是 0，则返回 6 个关节角度值；如果是 1、2、3，则返回 前3个值为位置(mm)，后3个值为姿态(弧度)
        self.logger.info(f"当前位置: {current_pos}")

        if coord in {1, 2, 3}:  # 直角 / 工具 / 用户坐标
            self.logger.info(f"已发送移动的直角坐标(movj)")
            self.logger.info(f"XYZ(mm): {pos[:3]}, ABC(rad): {pos[3:]}")
        else:  # coord == 0（关节坐标）
            self.logger.info(f"已发送移动的关节角度(movj)")
            self.logger.info(f"移动的目标关节角度(deg): {pos}")

        angle_diff = np.abs(np.array(current_pos) - np.array(pos))
        if np.all(angle_diff < tolerance):
            self.logger.info(
                f"当前点位与目标点位在允许误差范围({tolerance})内, 切换到非阻塞模式"
            )
            self.logger.info(f"机器人移动成功")

            return True

        cmd_data = {"robot": robot, "vel": vel, "coord": coord, "pos": pos}
        self._send_command(self.CONSTANTS["ROBOT_MOVEMENT"]["POINT"], cmd_data)

        if block == 1:  # 阻塞模式
            start_time = time.time()
            status = self._return_get("5411status")

            while status != 0:
                if time.time() - start_time > timeout:
                    self.logger.error(f"机器人移动超时, 超过 {timeout} 秒未完成！")
                    return False

                status = self._return_get("5411status")
            if status == 0:
                self.logger.info(f"机器人移动成功")
                return True
        elif block == 0:  # 非阻塞模式
            self.logger.info(f"机器人运动命令已发送, 当前为非阻塞模式")
            return True

    # NOTE 直线运动
    def movl(
        self,
        vel: int,
        coord: int,
        pos: List[float],
        block: int = 1,
        robot: int = 1,
        timeout: int = 30,
        tolerance: float = 0.05,
    ) -> bool:
        """
        机器人直线运动 MOVL

        Args:
            robot(int): 机器人号码
            vel(int): 速度百分⽐，1-100 的整数
            coord(int): 坐标模式
                - 0:  关节坐标(Joint)
                - 1:  直角坐标(Cart)
                - 2:  工具坐标(Tool)
                - 3:  用户坐标(User)
            pos(List[float]):[1.1,2.2,3.3,4.4,5.5,6.6,7.7]
                - 关节坐标分别代表关节角度值
                - 直角坐标系分别x,y,z,a,b,c其中欧拉角的旋转顺序为'zyx',
                  工具、用户同直角直角坐标系下第七位参数默认为0即可
            block(int):
                - 0: 非阻塞模式
                - 1: 阻塞模式
            timeout(int): 阻塞模式下的超时时间, 单位为秒
            tolerance(float): 误差范围, 单位为度

        Returns:
            bool: 返回机器人是否移动成功
                - True: 移动成功
                - False: 移动失败
        """
        if coord not in {0, 1, 2, 3}:
            self.logger.warning(
                f"坐标参数应在0(关节坐标)、1(直角坐标)、2(工具坐标)、3(用户坐标)之一, 当前为: {coord}"
            )
            return False

        if vel > 100 or vel < 1:
            self.logger.warning(f"速度百分比应在 2 ~ 1000 之间, 当前为: {vel}")
            return False

        if block not in {0, 1}:
            self.logger.warning(
                f"block 参数应在0(非阻塞模式)、1(阻塞模式)之一, 当前为: {block}"
            )
            return False

        if len(pos) not in {6, 7}:
            self.logger.warning(f"pos 参数长度应为6或7, 当前为: {len(pos)}")
            return False

        if coord in {1, 2, 3}:
            self.logger.info(f"已发送移动的直角坐标(movl)")
            self.logger.info(f"XYZ(mm): {pos[:3]}, ABC(rad): {pos[3:]}")
        else:
            self.logger.info(f"已发送移动的关节角度(movl)")
            self.logger.info(f"移动的关节角度(deg): {pos}")

        current_pos = self.currentpos_inquiry(coord)
        angle_diff = np.abs(np.array(current_pos) - np.array(pos))

        if np.all(angle_diff < tolerance):
            self.logger.info(
                f"当前关节角度与目标角度在允许误差范围({tolerance} 度)内, 切换到非阻塞模式"
            )
            self.logger.info(f"机器人移动成功")

            return True

        cmd_data = {"robot": robot, "vel": vel, "coord": coord, "pos": pos}
        self._send_command(self.CONSTANTS["ROBOT_MOVEMENT"]["LINEAR"], cmd_data)

        if block == 1:  # 阻塞模式
            start_time = time.time()
            status = self._return_get("5411status")

            while status != 0:
                if time.time() - start_time > timeout:
                    self.logger.error(f"机器人移动超时, 超过 {timeout} 秒未完成！")
                    return False

                status = self._return_get("5411status")
            if status == 0:
                self.logger.info(f"机器人移动成功")

                return True
        elif block == 0:  # 非阻塞模式
            self.logger.info(f"机器人运动命令已发送, 当前为非阻塞模式")

            return True

    # NOTE 圆弧运动
    def movc(
        self,
        vel: int,
        coord: int,
        isFull: bool,
        posOne: List[float],
        posTwo: List[float],
        posThree: List[float],
        robot: int = 1,
        block: int = 1,
        timeout: int = 30,
    ) -> bool:
        """
        机器人圆弧运动 MOVC

        Args:
            robot(int): 机器人号码
            vel(int): 速度百分⽐,1-100的整数
            coord(int): 坐标模式
                - 0:  关节坐标(Joint)
                - 1:  直角坐标(Cart)
                - 2:  工具坐标(Tool)
                - 3:  用户坐标(User)
            isFull(bool):
                - False: MOVC
                - True: MOVCA(整圆)
            posOne(List[float]): 圆弧起始点
            posTwo(List[float]): 圆弧经过的中间点
            posThree(List[float]): 圆弧的目标点
                - 关节坐标分别代表关节角度值
                - 直角坐标系分别x,y,z,a,b,c其中欧拉角的旋转顺序为'zyx',
                  工具、用户同直角直角坐标系下第七位参数默认为0即可
            block(int):
                - 0: 非阻塞模式
                - 1: 阻塞模式
            timeout(int): 阻塞模式下的超时时间, 单位为秒

        Returns:
            bool: 返回机器人是否移动成功
                - True: 移动成功
                - False: 移动失败
        """
        if coord not in {0, 1, 2, 3}:
            self.logger.warning(
                f"坐标参数应在0(关节坐标)、1(直角坐标)、2(工具坐标)、3(用户坐标)之一, 当前为: {coord}"
            )
            return

        if vel > 100 or vel < 1:
            self.logger.warning(f"速度百分比应在 2~1000 之间, 当前为: {vel}")
            return

        if block not in {0, 1}:
            self.logger.warning(
                f"block 参数应在0(非阻塞模式)、1(阻塞模式)之一, 当前为: {block}"
            )
            return False

        if len(posOne) not in {6, 7}:
            self.logger.warning(f"posOne 参数长度应为6或7, 当前为: {len(posOne)}")
            return False
        if len(posTwo) not in {6, 7}:
            self.logger.warning(f"posTwo 参数长度应为6或7, 当前为: {len(posTwo)}")
            return False
        if len(posThree) not in {6, 7}:
            self.logger.warning(f"posThree 参数长度应为6或7, 当前为: {len(posThree)}")
            return False

        if isFull not in {True, False}:
            self.logger.warning(
                f"isFull为布尔类型, True为整圆, False为圆弧, 当前为: {isFull}"
            )
            return False

        if coord in {1, 2, 3}:
            self.logger.info(f"已发送直角圆弧运动(movc)")
            if isFull == True:
                self.logger.info(f"> 当前为整圆")
            elif isFull == False:
                self.logger.info(f"> 当前为圆弧")
            else:
                self.logger.warning(
                    f"isFull为布尔类型, True为整圆, False为圆弧, 当前为: {isFull}"
                )
                return
            self.logger.info(f"> 起点: XYZ(mm): {posOne[:3]}, ABC(rad): {posOne[3:]}")
            self.logger.info(f"> 中间点: XYZ(mm): {posTwo[:3]}, ABC(rad): {posTwo[3:]}")
            self.logger.info(
                f"> 终点: XYZ(mm): {posThree[:3]}, ABC(rad): {posThree[3:]}"
            )
        else:
            self.logger.info(f"已发送关节圆弧运动(movc)")
            if isFull == True:
                self.logger.info(f"> 当前为整圆")
            elif isFull == False:
                self.logger.info(f"> 当前为圆弧")

            self.logger.info(f"> 起点(deg): {posOne}")
            self.logger.info(f"> 中间点(deg): {posTwo}")
            self.logger.info(f"> 终点(deg): {posThree}")

        cmd_data = {
            "robot": robot,
            "vel": vel,
            "coord": coord,
            "isFull": isFull,
            "posOne": posOne,
            "posTwo": posTwo,
            "posThree": posThree,
        }

        self._send_command(self.CONSTANTS["ROBOT_MOVEMENT"]["CIRULAR"], cmd_data)

        if block == 1:  # 阻塞模式
            start_time = time.time()
            status = self._return_get("5411status")

            while status != 0:
                if time.time() - start_time > timeout:
                    self.logger.error(f"机器人移动超时, 超过 {timeout} 秒未完成！")
                    return False

                status = self._return_get("5411status")
            if status == 0:
                self.logger.info(f"机器人移动成功")
                self.return_status["5411status"] = None
                return True
        elif block == 0:  # 非阻塞模式
            self.logger.info(f"机器人运动命令已发送, 当前为非阻塞模式")
            return True

    def robot_motion_parameters_set(
        self,
        absolutePosResolution: float,
        interpolationMethod: int,
        minAccTime: float,
        minDecTime: float,
        runDelayTime: int,
        stopTime: int,
    ) -> None:
        """
        设置机器人运动参数

        Args:
            absolutePosResolution(float): 绝对位置分辨率, 范围[0.0001, 0.1]°
            interpolationMethod(int): 机器人插补方式
                - 0: S型插补
                - 1: 梯形插补
                - 2: 加加速度插补
            minTrajectTime(dict): 最小轨迹时间
                {
                    "minAccTime": float,  # 最小加速度时间, [0.05, 1] s
                    "minDecTime": float,  # 最小减速度时间, [0.05, 1] s
                }
            runDelayTime(int): 运行延迟时间, 范围[500, 20000] ms
            stopTime(int): 暂停时间, 范围[240, 2000] ms
            robot(int): 机器人号码

        Returns:
            None
        """
        cmd_data = {
            "absolutePosResolution": absolutePosResolution,
            "interpolationMethod": interpolationMethod,
            "minTrajectTime": {"minAccTime": minAccTime, "minDecTime": minDecTime},
            "runDelayTime": runDelayTime,
            "stopTime": stopTime,
        }
        interpolationMethod_map = {0: "S型插补", 1: "梯形插补", 2: "加加速度插补"}
        self._send_command(self.CONSTANTS["ROBOT_MOTION_PARAMETERS"]["SET"], cmd_data)
        self.logger.info(f"设置机器人运动参数")
        self.logger.info(f"> 绝对位置分辨率: {absolutePosResolution}")
        self.logger.info(
            f"> 机器人插补方式: {interpolationMethod_map.get(interpolationMethod)}"
        )
        self.logger.info(f"> 最小加速度时间: {minAccTime}")
        self.logger.info(f"> 最小减速度时间: {minDecTime}")
        self.logger.info(f"> 运行延迟时间: {runDelayTime}")
        self.logger.info(f"> 暂停时间: {stopTime}")
        return None

    def robot_motion_parameters_inquire(self, robot: int = 1) -> dict:
        """
        查询机器人运动参数

        Args:
            robot(int): 机器人号码

        Returns:
            dict: 机器人运动参数字典, 包含以下键值:
                - absolutePosResolution (float): 绝对位置分辨率, 单位度, 取值范围: [0.0001, 0.1]
                - interpolationMethod (int): 插补方式标识:
                    - 0: S型插补
                    - 1: 梯形插补
                    - 2: 加加速度插补
                - minTrajectTime (dict): 最小轨迹时间配置字典:
                    - minAccTime (float): 最小加速度时间, 单位秒, 取值范围: [0.05, 1.0]
                    - minDecTime (float): 最小减速度时间, 单位秒, 取值范围: [0.05, 1.0]
                - runDelayTime (int): 运行延迟时间, 单位毫秒, 取值范围: [500, 20000]
                - stopTime (int): 暂停时间, 单位毫秒, 取值范围: [240, 2000]
        """
        self._send_command(
            self.CONSTANTS["ROBOT_MOTION_PARAMETERS"]["INQUIRE"], {"robot": robot}
        )
        absolutePosResolution = self._return_get("3092absolutePosResolution")
        interpolationMethod = self._return_get("3092interpolationMethod")
        minTrajectTime = self._return_get("3092minTrajectTime")
        minAccTime = minTrajectTime["minAccTime"]
        minDecTime = minTrajectTime["minDecTime"]
        runDelayTime = self._return_get("3092runDelayTime")
        stopTime = self._return_get("3092stopTime")
        result = {
            "absolutePosResolution": absolutePosResolution,
            "interpolationMethod": interpolationMethod,
            "minTrajectTime": {
                "minAccTime": minAccTime,
                "minDecTime": minDecTime,
            },
            "runDelayTime": runDelayTime,
            "stopTime": stopTime,
        }
        interpolationMethod_map = {0: "S型插补", 1: "梯形插补", 2: "加加速度插补"}
        self.logger.info(f"查询机器人运动参数")
        self.logger.info(f"> 绝对位置分辨率: {absolutePosResolution} °")
        self.logger.info(
            f"> 机器人插补方式: {interpolationMethod_map.get(interpolationMethod)}"
        )
        self.logger.info(f"> 最小加速度时间: {minAccTime} s")
        self.logger.info(f"> 最小减速度时间: {minDecTime} s")
        self.logger.info(f"> 运行延迟时间: {runDelayTime} ms")
        self.logger.info(f"> 暂停时间: {stopTime} ms")
        return result

    def jog_operation_move(self, axis: int, direction: int, robot: int = 1) -> None:
        """
        执行点动操作

        Args:
            axis(int): 代表所要操作的轴, 范围 [1, 12]
                - 关节坐标时表示对应的轴, 从 8 开始为外部轴
                - 直角坐标时按顺序分别表示 X轴 Y轴 Z轴 A轴 B轴 C轴
                - 工具坐标时按顺序分别表示 TX轴 TY轴 TZ轴 TA轴 TB轴 TC轴
                - 用户坐标时按顺序分别表示 UX轴 UY轴 UZ轴 UA轴 UB轴 UC轴
            direction(int): 关节移动方向
                - 1: 正向
                - -1: 反向
            robot(int): 机器人号码

        Returns:
            None
        """
        # ! 验证条件强制要求 axis 必须在 [1, self.dof] 范围内，这意味着实际上只能使用关节坐标系，但是上面提到了多种坐标系，可能是 Bug
        if not (1 <= axis <= self.dof):
            self.logger.warning(f"轴号应为 1 ~ {self.dof} 的关节号, 当前为: {axis}")
            return

        if direction not in [1, -1]:
            self.logger.warning(
                f"关节移动方向应为 1(正向)或者 -1(反向), 当前为: {direction}"
            )
            return

        cmd_data = {"axis": axis, "direction": direction, "robot": robot}
        direction_map = {1: "正向", -1: "反向"}
        self._send_command(self.CONSTANTS["JOG_OPERATION"]["MOVE"], cmd_data)
        self.logger.info(f"开始点动操作")
        self.logger.info(f"> 关节轴: {axis}")
        self.logger.info(f"> 方向: {direction_map.get(direction)}")

        return None

    def jog_operation_stop(self, axis: int, robot: int = 1) -> None:
        """
        停止执行点动操作

        Args:
            axis(int): 代表所要操作的轴, 范围[1, 12]
                - 关节坐标时表示对应的轴, 从8开始为外部轴
                - 直角坐标时按顺序分别表示X轴 Y轴 Z轴 A轴 B轴 C轴
                - 工具坐标时按顺序分别表示TX轴 TY轴 TZ轴 TA轴 TB轴 TC轴
                - 用户坐标时按顺序分别表示UX轴 UY轴 UZ轴 UA轴 UB轴 UC轴
            robot(int): 机器人号码

        Returns:
            None
        """
        if not (1 <= axis <= self.dof):
            self.logger.warning(f"轴号应为 1 ~ {self.dof} 的关节号, 当前为: {axis}")
            return

        self._send_command(
            self.CONSTANTS["JOG_OPERATION"]["STOP"], {"axis": axis, "robot": robot}
        )
        self.logger.info(f"停止 {axis} 轴点动操作")
        return None

    def jog_jointparameter_set(
        self, axis: int, maxSpeed: float, maxAcc: float, robot: int = 1
    ) -> None:
        """
        设置关节轴点动参数

        Args:
            axis(int): 表示设置的关节轴
            maxSpeed(float): 关节轴最大点动速度, 单位: °/s, 范围: [1, 100]
            maxAcc(float): 关节轴点动加速度, 单位: °/s², 范围: [1, 1000]
            robot(int): 机器人号码

        Returns:
            None
        """
        if not (1 <= axis <= self.dof):
            self.logger.warning(f"轴号应为 1 ~ {self.dof} 的关节号, 当前为: {axis}")
            return
        if not (1 <= maxSpeed <= 100):
            self.logger.warning(f"最大点动速度应在1~100之间, 当前为: {maxSpeed}")
            return
        if not (1 <= maxAcc <= 1000):
            self.logger.warning(f"点动加速度应在1~1000之间, 当前为: {maxAcc}")
            return

        cmd_data = {
            "axis": axis,
            "maxSpeed": maxSpeed,
            "maxAcc": maxAcc,
            "robot": robot,
        }
        self._send_command(self.CONSTANTS["JOG_JOINTPARAMETER"]["SET"], cmd_data)
        self.logger.info("设置关节轴点动参数")
        self.logger.info(f"> 关节轴编号为: {axis}")
        self.logger.info(f"> 关节轴最大点动速度为: {maxSpeed} °/s")
        self.logger.info(f"> 关节轴点动加速度为: {maxAcc} °/s²")
        return None

    def jog_jointparameter_inquire(self, axis: int, robot: int = 1) -> dict:
        """
        查询关节轴点动参数

        Args:
            axis(int): 需要查询关节点动参数的轴号, 范围最小为1, 最大为当前机器人轴数
            robot(int): 机器人号码

        Returns:
            dict: 关节轴点动参数, 包括以下字段:
                - axis: int, 当前查询关节点动参数的轴号
                - maxSpeed: float, 关节轴最大点动速度, 单位 °/s
                - maxAcc: float, 关节轴点动加速度, 单位 °/s²
                - robot: int, 机器人号码
        """
        if not (1 <= axis <= self.dof):
            self.logger.warning(f"轴号应为 1 ~ {self.dof} 的关节号, 当前为: {axis}")
            return

        self._send_command(
            self.CONSTANTS["JOG_JOINTPARAMETER"]["INQUIRE"],
            {"axis": axis, "robot": robot},
        )
        axis = self._return_get("30A4axis")
        MaxSpeed = self._return_get("30A4maxSpeed")
        MaxAcc = self._return_get("30A4maxAcc")
        result = {"axis": axis, "maxSpeed": MaxSpeed, "maxAcc": MaxAcc, "robot": robot}
        self.logger.info(f"查询关节轴点动参数")
        self.logger.info(f"> 关节轴编号为: {axis}")
        self.logger.info(f"> 关节轴最大点动速度为: {MaxSpeed} °/s")
        self.logger.info(f"> 关节轴点动加速度为: {MaxAcc} °/s²")
        return result

    def jog_rectparameter_set(self, maxSpeed: float, maxAcc: float) -> None:
        """
        设置直角坐标点动参数

        Args:
            MaxSpeed(float): 直角坐标下的点动最大速度, 单位 mm/s, 范围: [1,250]
            MaxAcc(float): 直角坐标下的点动最大加速度, 单位 mm/s², 范围: [1,4000]

        Returns:
            None
        """
        self._send_command(
            self.CONSTANTS["JOG_RECTPARAMETER"]["SET"],
            {"maxSpeed": maxSpeed, "maxAcc": maxAcc},
        )
        self.logger.info(f"设置直角坐标点动参数")
        self.logger.info(f"> 设置直角坐标下的点动最大速度为: {maxSpeed} mm/s")
        self.logger.info(f"> 设置直角坐标下的点动最大加速度为:{maxAcc} mm/s²")
        return None

    def jog_rectparameter_inqure(self, robot: int = 1) -> dict:
        """
        查询直角坐标点动参数

        Args:
            robot(int): 机器人号码

        Returns:
            dict: 直角坐标点动参数, 包括以下字段:
                - maxSpeed: float, 直角坐标下的点动最大速度, 单位 mm/s
                - maxAcc: float, 直角坐标下的点动最大加速度, 单位 mm/s²
                - robot: int, 机器人号码
        """
        self._send_command(
            self.CONSTANTS["JOG_RECTPARAMETER"]["INQUIRE"], {"robot": robot}
        )
        maxAcc = self._return_get("30A7maxAcc")
        maxSpeed = self._return_get("30A7maxSpeed")
        result = {"maxSpeed": maxSpeed, "maxAcc": maxAcc, "robot": robot}
        self.logger.info(f"查询直角坐标点动参数")
        self.logger.info(f"> 直角坐标下点动最大速度为: {maxSpeed} mm/s")
        self.logger.info(f"> 直角坐标下点动最大加速度为: {maxAcc} mm/s²")
        return result

    def jog_sensitivity_set(self, sensitivity: float) -> None:
        """
        设置点动灵敏度(默认 0.001)

        Args:
            sensitivity(float): 点动灵敏度, 单位 度, 围 [0.001, 1]

        Returns:
            None
        """
        self._send_command(
            self.CONSTANTS["JOG_SENSITIVITY"]["SET"], {"sensitivity": sensitivity}
        )
        self.logger.info(f"设置点动灵敏度为: {sensitivity} °")
        return None

    def jog_sensitivity_inquire(self, robot: int = 1) -> float:
        """
        查询点动灵敏度

        Args:
            None

        Returns:
            float: 当前点动灵敏度, 单位 度, 范围 [0.001, 1]
        """
        self._send_command(
            self.CONSTANTS["JOG_SENSITIVITY"]["INQUIRE"], {"robot": robot}
        )
        sensitivity = self._return_get("30AAsensitivity")
        self.logger.info(f"查询点动灵敏度为: {sensitivity} °")
        return sensitivity

    # NOTE 查询机械臂当前位置信息，依据坐标系而获取不同的位置表示
    def currentpos_inquiry(self, coord: int, robot: int = 1) -> List[float]:
        """
        获取当前位置

        Args:
            robot(int): 机器人号码
            coord(int): 坐标模式
                - -1: 控制器当前坐标
                - 0:  关节坐标(Joint)
                - 1:  直角坐标(世界坐标系)(Cart)
                - 2:  工具坐标(Tool)
                - 3:  用户坐标(User)

        Returns:
            List[float]:
                - 关节坐标分别代表关节角度值
                - 直角坐标系分别 x, y, z, a, b, c, 工具、用户同直角
        """
        if coord not in [-1, 0, 1, 2, 3, 4]:
            self.logger.warning(
                f"坐标参数应在 0(关节坐标)、1(直角坐标)、2(工具坐标)、3(用户坐标)之一, 当前为: {coord}"
            )
            return

        # 协议封装: 构造查询指令数据包
        cmd_data = {"robot": robot, "coord": coord}
        # 命令常量: 使用预定义的 CURRENTPOS_INQUIRE 常量作为命令类型
        # 通信方式: 通过 6001 端口发送TCP指令到控制器
        self._send_command(self.CONSTANTS["CURRENTPOS_INQUIRE"], cmd_data)
        # 从控制器接收位置数据
        # ! 控制器返回的原始位置数据，可能包含冗余信息。即：这里的 position 可能是 7 个或更多元素的列表，需要后续处理（如切片）
        position = self._return_get("3212pos")

        # case1 笛卡尔坐标系处理 (coord=1,2,3)
        if coord in {1, 2, 3}:
            # position = [round(angle, 4) for angle in np.delete(position, 6).tolist()]
            # 上面用删除方法，有一个隐患：如果机器人返回的直角坐标只有 6 个值（没有工具号），np.delete(position, 6) 会因索引越界报错。更稳健的方式是用切片：
            position = [round(angle, 4) for angle in position[:6]]
            self.logger.debug(f"当前位置: {position}")

        # case2 关节坐标系处理 (coord=-1,0,4)
        else:
            # position = [round(angle, 4) for angle in position]
            # 对关节坐标返回值进行 [:self.dof] 切片，将结果截断到配置的自由度数
            position = [round(angle, 4) for angle in position[: self.dof]]
            self.logger.debug(f"当前位置: {position}")

        return position

    def axis_actual_vel_inquire(self, robot: int = 1) -> dict:
        """
        查询轴速度

        Args:
            robot(int): 机器人号码

        Returns:
            dict: 轴速度, 包括以下字段:
                - actualLineVel: float, 当前末端线速度, 单位 mm/s
                - maxActualLineVel: float, 当前末端最大线速度, 单位 mm/s
                - axisActualVel: List[float], 当前轴速度, 单位 °/s
                - maxAxisActualVel: List[float], 当前最大轴速度, 单位 °/s
                - axisActualVelSync: List[float], 外部轴当轴速度, 单位 °/s
                - maxAxisActualVelSync: List[float], 外部轴最大轴速度, 单位 °/s
                - robot: int, 机器人号码
            返回数据示例:
                {
                    "actualLineVel": 0.0,
                    "maxActualLineVel": 0.0,
                    "axisActualVel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "maxAxisActualVel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "axisActualVelSync": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "maxAxisActualVelSync": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "robot": 1
                }
        """
        self._send_command(self.CONSTANTS["AXISACTUALVEL_INQUIRE"], {"robot": robot})
        actualLineVel = self._return_get("3231actualLineVel")
        maxActualLineVel = self._return_get("3231maxActualLineVel")
        axisActualVel = self._return_get("3231axisActualVel")
        maxAxisActualVel = self._return_get("3231maxAxisActualVel")
        axisActualVelSync = self._return_get("3231axisActualVelSync")
        maxAxisActualVelSync = self._return_get("3231maxAxisActualVelSync")
        result = {
            "actualLineVel": actualLineVel,
            "maxActualLineVel": maxActualLineVel,
            "axisActualVel": axisActualVel,
            "maxAxisActualVel": maxAxisActualVel,
            "axisActualVelSync": axisActualVelSync,
            "maxAxisActualVelSync": maxAxisActualVelSync,
            "robot": robot,
        }
        self.logger.info(f"查询轴速度")
        self.logger.info(f"> 当前末端线速度: {actualLineVel} mm/s")
        self.logger.info(f"> 当前末端最大线速度: {maxActualLineVel} mm/s")
        self.logger.info(
            f"> 1-{len(axisActualVel)} 轴当前轴速度: {axisActualVel} , 单位: °/s"
        )
        self.logger.info(
            f"> 1-{len(maxAxisActualVel)} 轴当前最大轴速度: {maxAxisActualVel} , 单位: °/s"
        )
        self.logger.info(f"> 外部轴当轴速度: {axisActualVelSync} , 单位: °/s")
        self.logger.info(f"> 外部轴最大轴速度: {maxAxisActualVelSync} , 单位: °/s")
        return result

    def servo_temperature_exist(self, robot: int = 1) -> bool:
        """
        查询伺服温度是否存在

        Args:
            robot(int): 机器人号码

        Returns:
            bool: 伺服温度是否存在
                - True: 存在
                - False: 不存在
        """
        self._send_command(self.CONSTANTS["SERVO_TEMPERATURE_EXIST"], {"robot": robot})
        IsExist = self._return_get("3251IsExist")
        self.logger.info(f"查询伺服温度是否存在")
        self.logger.info(f"> 伺服温度是否存在: {IsExist}")
        return IsExist

    def motor_speed_inquire(self, robot: int = 1) -> dict:
        """
        查询电机速度

        Args:
            robot(int): 机器人号码

        Returns:
            dict: 电机转速, 包括以下字段:
                - maxVel: List[int], 机器人最大电机转速列表, 单位: rpm
                - maxVelSync: List[int], 机器人外部轴最大电机转速列表, 单位: rpm
                - slave_max_vel: List[int], 机器人主轴及从动轴电机速度列表, 单位: rpm
                - slave_max_vel_sync: List[int], 机器人外部轴主轴及从动轴最大电机速度列表, 单位: rpm
                - slave_vel: List[int], 机器人主轴及从动轴电机速度列表, 单位: rpm
                - slave_vel_sync: List[int], 机器人外部轴主轴及从动轴电机速度列表, 单位: rpm
                - vel: List[int], 机器人当前电机转速列表, 单位: rpm
                - velSync: List[int], 机器人外部轴当前电机转速列表, 单位: rpm
                - robot: int, 机器人号码
        """
        self._send_command(
            self.CONSTANTS["MOTOR_STATUS_INQUIRE"]["SPEED"], {"robot": robot}
        )
        maxVel = self._return_get("3261maxVel")
        maxVelSync = self._return_get("3261maxVelSync")
        slave_max_vel = self._return_get("3261slave_max_vel")
        slave_max_vel_sync = self._return_get("3261slave_max_vel_sync")
        slave_vel = self._return_get("3261slave_vel")
        slave_vel_sync = self._return_get("3261slave_vel_sync")
        vel = self._return_get("3261vel")
        velSync = self._return_get("3261velSync")
        result = {
            "maxVel": maxVel,
            "maxVelSync": maxVelSync,
            "slave_max_vel": slave_max_vel,
            "slave_max_vel_sync": slave_max_vel_sync,
            "slave_vel": slave_vel,
            "slave_vel_sync": slave_vel_sync,
            "vel": vel,
            "velSync": velSync,
            "robot": robot,
        }
        self.logger.info(f"查询电机转速")
        self.logger.info(f"> 机器人最大电机转速列表: {maxVel} , 单位: rpm")
        self.logger.info(f"> 机器人外部轴最大电机转速列表: {maxVelSync} , 单位: rpm")
        self.logger.info(f"> 机器人当前电机转速列表: {vel} , 单位: rpm")
        self.logger.info(f"> 机器人外部轴当前电机转速列表: {velSync} , 单位: rpm")
        self.logger.info(f"> 机器人主轴及从动轴电机速度列表: {slave_vel} , 单位: rpm")
        self.logger.info(
            f"> 机器人主轴及从动轴最大电机速度列表: {slave_max_vel} , 单位: rpm"
        )
        self.logger.info(
            f"> 机器人外部轴主轴及从动轴电机速度列表: {slave_vel_sync} , 单位: rpm"
        )
        self.logger.info(
            f"> 机器人外部轴主轴及从动轴最大电机速度列表: {slave_max_vel_sync} , 单位: rpm"
        )
        return result

    def motor_torque_inquire(self, robot: int = 1) -> dict:
        """
        查询电机扭矩

        Args:
            robot(int): 机器人号码

        Returns:
            dict: 电机扭矩, 包括以下字段:
                - maxTorq: List[float], 机器人最大电机扭矩列表, 单位: ‰
                - maxTorqSync: List[float], 机器人外部轴最大电机扭矩列表, 单位: ‰
                - slave_max_torq: List[float], 机器人主轴及从动轴电机扭矩列表, 单位: ‰
                - slave_max_torq_sync: List[float], 机器人外部轴主轴及从动轴最大电机扭矩列表, 单位: ‰
                - slave_torq: List[float], 机器人主轴及从动轴电机扭矩列表, 单位: ‰
                - slave_torq_sync: List[float], 机器人外部轴主轴及从动轴电机扭矩列表, 单位: ‰
                - torq: List[float], 机器人当前电机扭矩列表, 单位: ‰
                - torqSync: List[float], 机器人外部轴当前电机扭矩列表, 单位: ‰
                - maxTheoTorq: List[float], 机器人最大电机理论扭矩列表, 单位: ‰
                - theoTorq: List[float], 机器人当前电机理论扭矩列表, 单位: ‰
                - robot: int, 机器人号码
        """
        self._send_command(
            self.CONSTANTS["MOTOR_STATUS_INQUIRE"]["TORQUE"], {"robot": robot}
        )
        maxTorq = self._return_get("3263maxTorq")
        maxTorqSync = self._return_get("3263maxTorqSync")
        slave_max_torq = self._return_get("3263slave_max_torq")
        slave_max_torq_sync = self._return_get("3263slave_max_torq_sync")
        slave_torq = self._return_get("3263slave_torq")
        slave_torq_sync = self._return_get("3263slave_torq_sync")
        torq = self._return_get("3263torq")
        torqSync = self._return_get("3263torqSync")
        maxTheoTorq = self._return_get("3263maxTheoTorq")
        theoTorq = self._return_get("3263theoTorq")
        result = {
            "maxTorq": maxTorq,
            "maxTorqSync": maxTorqSync,
            "slave_max_torq": slave_max_torq,
            "slave_max_torq_sync": slave_max_torq_sync,
            "slave_torq": slave_torq,
            "slave_torq_sync": slave_torq_sync,
            "torq": torq,
            "torqSync": torqSync,
            "maxTheoTorq": maxTheoTorq,
            "theoTorq": theoTorq,
            "robot": robot,
        }
        self.logger.info(f"查询电机扭矩")
        self.logger.info(f"> 机器人最大电机理论扭矩列表: {maxTheoTorq} , 单位: ‰")
        self.logger.info(f"> 机器人最大电机扭矩列表: {theoTorq} , 单位: ‰")
        self.logger.info(f"> 机器人外部轴最大电机扭矩列表: {maxTorq} , 单位: ‰")
        self.logger.info(f"> 机器人当前电机理论扭矩列表: {maxTorqSync} , 单位: ‰")
        self.logger.info(f"> 机器人当前电机扭矩列表: {torq} , 单位: ‰")
        self.logger.info(f"> 机器人外部轴当前电机扭矩列表: {torqSync} , 单位: ‰")
        self.logger.info(f"> 机器人主轴及从动轴电机扭矩列表: {slave_torq} , 单位: ‰")
        self.logger.info(
            f"> 机器人主轴及从动轴电机最大扭矩列表: {slave_max_torq} , 单位: ‰"
        )
        self.logger.info(
            f"> 机器人外部轴主轴及从动轴电机扭矩列表: {slave_torq_sync} , 单位: ‰"
        )
        self.logger.info(
            f"> 机器人外部轴主轴及从动轴最大电机扭矩列表: {slave_max_torq_sync} , 单位: ‰"
        )
        return result

    def motor_temperature_inquire(self, robot: int = 1) -> dict:
        """
        查询电机温度
        先通过检查servo_temperature_exist()函数查询伺服温度是否存在

        Args:
            robot(int): 机器人号码

        Returns:
            dict: 电机温度, 包括以下字段:
                - temperature: List[float], 机器人电机温度列表, 单位: ℃
                - maxTemperature: List[float], 机器人电机最大温度列表, 单位: ℃
                - temperatureSync: List[float], 机器人外部轴电机温度列表, 单位: ℃
                - maxtemperatureSync: List[float], 机器人外部轴电机最大温度列表, 单位: ℃
                - slave_temperature: List[float], 机器人主轴及从动轴电机温度列表, 单位: ℃
                - slave_temperature_max: List[float], 机器人主轴及从动轴电机最大温度列表, 单位: ℃
                - slave_temperature_sync_out: List[float], 机器人外部轴主轴及从动轴电机温度列表, 单位: ℃
                - slave_temperature_sync_out_max: List[float], 机器人外部轴主轴及从动轴电机最大温度列表, 单位: ℃
                - robot: int, 机器人号码
        """
        self._send_command(
            self.CONSTANTS["MOTOR_STATUS_INQUIRE"]["TEMPERATURE"], {"robot": robot}
        )
        temperature = self._return_get("326Btemperature")
        maxTemperature = self._return_get("326BmaxTemperature")
        temperatureSync = self._return_get("326BtemperatureSync")
        maxtemperatureSync = self._return_get("326BmaxtemperatureSync")
        slave_temperature = self._return_get("326Bslave_temperature")
        slave_temperature_max = self._return_get("326Bslave_temperature_max")
        slave_temperature_sync_out = self._return_get("326Bslave_temperature_sync_out")
        slave_temperature_sync_out_max = self._return_get(
            "326Bslave_temperature_sync_out_max"
        )
        result = {
            "temperature": temperature,
            "maxTemperature": maxTemperature,
            "temperatureSync": temperatureSync,
            "maxtemperatureSync": maxtemperatureSync,
            "slave_temperature": slave_temperature,
            "slave_temperature_max": slave_temperature_max,
            "slave_temperature_sync_out": slave_temperature_sync_out,
            "slave_temperature_sync_out_max": slave_temperature_sync_out_max,
            "robot": robot,
        }
        self.logger.info(f"查询电机温度")
        self.logger.info(f"> 机器人电机温度列表: {temperature} , 单位: ℃")
        self.logger.info(f"> 机器人电机最大温度列表: {maxTemperature} , 单位: ℃")
        self.logger.info(f"> 机器人外部轴电机温度列表: {temperatureSync} , 单位: ℃")
        self.logger.info(
            f"> 机器人外部轴电机最大温度列表: {maxtemperatureSync} , 单位: ℃"
        )
        self.logger.info(
            f"> 机器人主轴及从动轴电机温度列表: {slave_temperature} , 单位: ℃"
        )
        self.logger.info(
            f"> 机器人主轴及从动轴电机最大温度列表: {slave_temperature_max} , 单位: ℃"
        )
        self.logger.info(
            f"> 机器人外部轴主轴及从动轴电机温度列表: {slave_temperature_sync_out} , 单位: ℃"
        )
        self.logger.info(
            f"> 机器人外部轴主轴及从动轴电机最大温度列表: {slave_temperature_sync_out_max} , 单位: ℃"
        )
        return result

    def following_error_inquire(self, robot: int = 1) -> dict:
        """
        查询跟随误差

        Args:
            robot(int): 机器人号码

        Returns:
            dict: 跟随误差, 包括以下字段:
                - curDeviations: List[float], 当前机器人误差列表, 单位: ‱
                - curExternalDeviations: List[float], 当前外部轴误差列表, 单位: ‱
                - maxDeviations: List[float], 最大机器人误差列表, 单位: ‱
                - maxExternalDeviations: List[float], 最大外部轴误差列表, 单位: ‱
                - robot: int, 机器人号码
        """
        self._send_command(self.CONSTANTS["FOLLOWING_ERROR_INQUIRE"], {"robot": robot})
        curDeviations = self._return_get("3871curDeviations")
        curExternalDeviations = self._return_get("3871curExternalDeviations")
        maxDeviations = self._return_get("3871maxDeviations")
        maxExternalDeviations = self._return_get("3871maxExternalDeviations")
        result = {
            "curDeviations": curDeviations,
            "curExternalDeviations": curExternalDeviations,
            "maxDeviations": maxDeviations,
            "maxExternalDeviations": maxExternalDeviations,
            "robot": robot,
        }
        self.logger.info(f"查询跟随误差")
        self.logger.info(f"> 当前机器人误差列表: {curDeviations} , 单位: ‱")
        self.logger.info(f"> 当前外部轴误差列表: {curExternalDeviations} , 单位: ‱")
        self.logger.info(f"> 最大机器人误差列表: {maxDeviations} , 单位: ‱")
        self.logger.info(f"> 最大外部轴误差列表: {maxExternalDeviations} , 单位: ‱")
        return result

    def jobsend_done(
        self,
        jobname: str,
        suffixname: str = ".JBR",
        line: int = 1,
        globalStart: bool = False,
        startOver: bool = 1,
        continueRun: int = 0,
        robot: int = 1,
    ) -> None:
        """
        开始运行作业文件
        (程序的运行需要在运行模式下, 并且伺服就绪, 上电没有报错)

        Args:
            robot(int): 机器人号码
            jobname(str): 作业文件名字
            suffixname(str): 作业文件后缀名
                - .JBR: 主程序
                - .JBP: 后台局部程序
                - .JBPG: 后台全局程序
            globalStart(bool):
                - True: 多机模式
                - False: 单机模式
            startOver(bool):
                - 1: 重头运行
                - 0: 继续运行
            line(int): 程序开始运行的行号,不能为零, 不能超过总⾏数
            continueRun(int):
                - 1: 继续运⾏
                - 0: 不继续运⾏
            "continueRun"和"startOver"的组合使用:
                - "continueRun":1 "startOver":1 断点执行
                - "continueRun":0 "startOver":1 从头 运行
                - "continueRun":0 "startOver":0 当前行 运行

        Returns:
            None
        """
        cmd_data = {
            "robot": robot,
            "jobname": jobname,
            "suffixname": suffixname,
            "line": line,
            "globalStart": globalStart,
            "continueRun": continueRun,
            "startOver": startOver,
        }
        self.logger.info(f"运行作业文件: {jobname}")
        self._send_command(self.CONSTANTS["JOB_CONTROL"]["JOBSEND_DONE"], cmd_data)
        return None

    def stop_job_run(self, robot: int = 1) -> None:
        """
        停止正在运行的作业文件

        Args:
            robot(int): 机器人号码

        Returns:
            None
        """
        cmd_data = {"robot": robot}
        self._send_command(self.CONSTANTS["JOB_CONTROL"]["STOP_JOB_RUN"], cmd_data)
        self.logger.info(f"停止正在运行的作业文件")
        return None

    def jobfile_list_inquire(self) -> dict:
        """
        获取作业文件列表

        Args:
            None

        Returns:
            dict: 作业文件列表, 包括以下字段:
                - absolutepath: List[str], 作业文件路径
                - jobfilenum: List[int], 各作业文件路径下的作业文件数量
                - jobfilelist: List[str], 作业文件列表
                - listnum: int, 作业文件数量, 取值范围 [0, 5]
        """
        self._send_command(self.CONSTANTS["JOB_CONTROL"]["JOBFILE_LIST_INQUIRE"])
        time.sleep(0.8)
        absolutepath = self.return_status.get("5011absolutepath")
        jobfilenum = self.return_status.get("5011jobfilenum")
        jobfilelist = self.return_status.get("5012jobfilelist")
        listnum = self.return_status.get("5012listnum")
        result = {
            "absolutepath": absolutepath,
            "jobfilenum": jobfilenum,
            "jobfilelist": jobfilelist,
            "listnum": listnum,
        }
        self.logger.info(f"获取作业文件列表")
        self.logger.info(f"> 作业文件路径: {absolutepath}")
        self.logger.info(f"> 各作业文件路径下的作业文件数量: {jobfilenum}")
        self.logger.info(f"> {listnum} 个文件为: {jobfilelist}")
        return result

    def jobfile_open(
        self, jobName: str, suffixname: str = ".JBR", robot: int = 1
    ) -> None:
        """
        打开作业文件

        Args:
            robot(int): 机器人号码
            jobName(str): 作业文件名字
            suffixname(str): 作业文件后缀名
                - .JBR: 主程序
                - .JBP: 后台局部程序
                - .JBPG: 后台全局程序

        Returns:
            None
        """
        self._send_command(
            self.CONSTANTS["JOB_CONTROL"]["JOBFILE_OPEN"],
            {"robot": robot, "jobName": jobName, "suffixname": suffixname},
        )
        self.logger.info(f"打开作业文件: {jobName}")
        return None

    def jobfile_open_inquire(self, robot: int = 1) -> str:
        """
        查询已打开的作业文件

        Args:
            robot(int): 机器人号码

        Returns:
            str: 已打开的作业文件
        """
        self._send_command(
            self.CONSTANTS["JOB_CONTROL"]["JOBFILE_OPEN_INQUIRE"], {"robot": robot}
        )
        openedJobName = self._return_get("5104openedJobName")
        self.logger.info(f"已打开的作业文件: {openedJobName}")
        return openedJobName

    def directmotion_mode_set(self, open: bool, robot: int = 1) -> bool:
        """
        开启/关闭 socket 直接控制运动模式
        (开启后会进去特殊的运行模式, 关闭后需要手动切回示教模式)

        Args:
            robot(int): 机器人号码
            open(bool):
                - True:  开启socket直接控制运动模式
                - False: 关闭socket直接控制运动模式

        Returns:
            bool: 开启/关闭状态
                - True:  socket直接控制运动模式开启
                - False: socket直接控制运动模式关闭
        """
        if not isinstance(open, bool):
            self.logger.warning(
                f"实际收到: {open}, 类型: {type(open)}, open 必须是布尔类型"
            )
            return

        cmd_data = {"robot": robot, "open": open}
        self._send_command(self.CONSTANTS["DIRECTMOTION"]["MODE_SET"], cmd_data)
        time.sleep(1)
        open_status = self._return_get("5642open")
        open_map = {True: "开启", False: "关闭"}
        self.logger.info(
            f"设置 直接控制运动模式: {open_map.get(open_status, '未知状态')}"
        )
        return open_status

    def directmotion_mode_inquire(self, robot: int = 1) -> bool:
        """
        查询socket直接控制运动模式

        Args:
            None

        Returns:
            bool: 开启/关闭状态
                - True:  socket直接控制运动模式开启
                - False: socket直接控制运动模式关闭
        """
        self._send_command(
            self.CONSTANTS["DIRECTMOTION"]["MODE_INQUIRE"], {"robot": robot}
        )
        open_status = self._return_get("5642open")
        open_map = {True: "开启", False: "关闭"}
        self.logger.info(
            f"查询 直接控制运动模式: {open_map.get(open_status, '未知状态')}"
        )
        return open_status

    def directmotion_insert_instrvec(
        self,
        trajectory: List[List[float]],
        acc: int = 100,
        dec: int = 100,
        pl: int = 5,
        velocity: int = 100,
        imovecoord: str = "RF",
        move_type: int = 1,
    ) -> None:
        """
        发送指令队列控制机械臂移动, 采用自动点位

        Args:
            ParaACC(int): 加速度  1-100
            ParaDEC(int): 减速度  1-100
            ParaPL(int): 平滑系数  0-5      不需要平滑填写0
            ParaSPIN(int): 圆弧和整圆指令, 0=姿态不变 1=六轴不转 2=六轴旋转  不需要填写0
            ParaTIME(int): 提前跳出该点往下执行的时间设置  不需要填写0
            ParaV(int): 速度1-100 这个是全局速度的百分比
            m_vUnit(int): 速度单位: 0 表示 cm/s, 1 表示 mm/s, 2 表示 百分比, 注意: 关节坐标填写2, 直角等坐标填写 0 或者 1 建议填写1

            data(List[float]): [0.0,0.0,0.0,0.0,0.0,0.0,0.0,%s,%s,%s,%s,%s,%s,%s,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ]
                - 第 1、2 位表示坐标 0 0 : 表示关节坐标(其中第二位角度-0 、弧度-1) 1 1: 表示直角坐标 2 1: 工具坐标 3 1: 用户坐标
                - 第 3 位 左右手 1-左 2-右 0-无左右手 默认为 0
                - 第 4, 5, 6, 7 位备用, 默认为 0
                - 第 8 至 14 位保存机器人本体 坐标值(7 位)
                    - 关节坐标下, 分别表示 1 到 7 轴的角度值
                    - 其他坐标下, 分别表示 x,y,z,a,b,c 六个轴的坐标
                    (按顺序填写 后面无值默认为 0.0 示例中,1.0,2.0,3.0,4.0,5.0,6.0 为关节坐标值)
                - 第 15 至 19 位 保存外部轴坐标值（最大支持五个外部轴, 外部轴只有关节值, 不足五个外部轴后关节坐标值补零）

            imovecoord(str):
                - "RF": 关节坐标
                - "BF": 直角坐标
                - "TF": 工具坐标
                - "UF": 用户坐标
            move_type(int):
                - 1: 点到
                - 2: 直线
                - 3: 圆弧
                - 4: 整圆

        Returns:
            None
        """
        motion_queue = []

        DATA_LEN = 21
        POS_START_IDX = 7
        dof = self.dof

        for pos in trajectory:
            if len(pos) != dof:
                raise ValueError(f"pos 长度应为 {dof}, 当前为 {len(pos)}")

            data = [0.0] * DATA_LEN
            data[POS_START_IDX : POS_START_IDX + dof] = pos

            para_var_data = [
                {"data": 0.0, "secondvalue": 0, "value": 0, "varname": ""}
                for _ in range(DATA_LEN - 1)
            ]
            for i in range(dof):
                para_var_data[POS_START_IDX + i]["data"] = pos[i]

            q = {
                "ParaACC": {"data": acc, "secondvalue": 0, "value": 0, "varname": ""},
                "ParaDEC": {"data": dec, "secondvalue": 0, "value": 0, "varname": ""},
                "ParaPL": {"data": pl, "secondvalue": 0, "value": 0, "varname": ""},
                "ParaSPIN": {"data": 0.0, "secondvalue": 0, "value": 0, "varname": ""},
                "ParaSYNC": {"data": 0.0, "secondvalue": 0, "value": 0, "varname": ""},
                "ParaTIME": {"data": 0.0, "secondvalue": 0, "value": 0, "varname": ""},
                "ParaV": {
                    "data": velocity,
                    "m_vUnit": 2,
                    "secondvalue": 0,
                    "value": 0,
                    "varname": "",
                },
                "RobotPos": {
                    "ctype": 1,
                    "data": data,
                    "key": "",
                    "paraVarData": para_var_data,
                },
                "ctype": 0,
                "imovecoord": imovecoord,
                "length": 0.0,
                "logout": False,
                "margin": 0.0,
                "offsetAxis": 0,
                "para": 0,
                "polish": 0,
                "polishAngle": 0.0,
                "polishID": 1,
                "posidname": "",
                "posidtype": 0,
                "positionId": "",
                "radius": 0.0,
                "side": 0.0,
                "type": move_type,
                "userParamInt": 0,
                "userParamString": "",
                "width": 0.0,
            }

            motion_queue.append(q)

        self._send_command(
            self.CONSTANTS["DIRECTMOTION"]["INSERT_INSTRVEC"],
            {"data": motion_queue, "robot": 1},
        )

        self.logger.info(f"已发送队列 {len(motion_queue)} 组")
        self.logger.info(f"队列内容: {trajectory}")
        return None

    def directmotion_mode_suspend(self, robot: int = 1) -> None:
        """
        暂停追加队列运行

        Args:
            robot(int): 机器人号码

        Returns:
            None
        """
        self._send_command(
            self.CONSTANTS["DIRECTMOTION"]["MODE_SUSPEND"], {"robot": robot}
        )
        self.logger.info(f"暂停 追加队列运行")
        return None

    def directmotion_mode_start(self, robot: int = 1) -> None:
        """
        开始追加队列运行（暂停后, 发送使用）

        Args:
            robot(int): 机器人号码

        Returns:
            None
        """
        self._send_command(
            self.CONSTANTS["DIRECTMOTION"]["MODE_START"], {"robot": robot}
        )
        self.logger.info(f"开始 追加队列运行")
        return None

    def directmotion_mode_stop(self, robot: int = 1) -> None:
        """
        停止追加队列运行
        （尽量不用, 使用后可能有断开连接的风险）

        Args:
            robot(int): 机器人号码

        Returns:
            None
        """
        self._send_command(
            self.CONSTANTS["DIRECTMOTION"]["MODE_STOP"], {"robot": robot}
        )
        self.logger.info(f"停止 追加队列运行")
        return None

    def directmotion_mode_stop_keep_power_on(self, robot: int = 1) -> None:
        """
        设置队列模式停止但不下电

        Args:
            robot(int): 机器人号码

        Returns:
            None
        """
        self._send_command(
            self.CONSTANTS["DIRECTMOTION"]["MODE_STOP_KEEP_POWER_ON"], {"robot": robot}
        )
        self.logger.info(f"设置队列模式停止不下电")
        return None

    def slavetype_list_respond(self) -> dict:
        """
        查询从站列表

        Args:
            None

        Returns:
            dict: 从站列表
                - IO编号 (List[int]): 从站的 I/O 编号
                - 伺服编号 (List[int]): 从站的伺服编号
                - 伺服型号 中文 (List[str]): 从站的伺服型号（中文）
                - 伺服型号 英文 (List[str]): 从站的伺服型号（英文）
        """
        self._send_command(self.CONSTANTS["SLAVETYPE_LIST_INQUIRE"])
        IONum = self._return_get("7021IONum")
        servoNum = self._return_get("7021servoNum")
        slaveType = self._return_get("7021slaveType")
        slaveTypeEnglish = self._return_get("7021slaveTypeEnglish")
        result = {
            "IONum": IONum,
            "servoNum": servoNum,
            "slaveType": slaveType,
            "slaveTypeEnglish": slaveTypeEnglish,
        }
        self.logger.info(f"查询从站列表")
        self.logger.info(f"> IO编号: {IONum}")
        self.logger.info(f"> 伺服编号: {servoNum}")
        self.logger.info(f"> 伺服型号 中文: {slaveType}")
        self.logger.info(f"> 伺服型号 英文: {slaveTypeEnglish}")
        return result

    def jointparameter_set(
        self, axis: int, positiveLimit: float, reverseLimit: float, direction: int = 1
    ) -> bool:
        """
        设置关节参数

        Args:
            axis(int): 关节轴数
            positiveLimit(float):关节正限位
            reverseLimit(float):关节反限位
            direction(int):模型方向, 七轴的时候，四号关节(六轴的时候为三号关节)设置前查询一下是否与其他关节相反
                - 1: 正向
                - -1: 反向

        Returns:
            None
        """
        if not (1 <= axis <= self.dof):
            self.logger.warning(f"关节轴数应为 1 ~ {self.dof} 的关节号, 当前为: {axis}")
            return

        cmd_data = {
            "axis": axis,
            "axisDirection": 1,
            "backLash": 0.0,
            "direction": direction,
            "encoderResolution": 19,
            "isReduceRatioEnable": True,
            "maxAcc": 1.5,
            "maxDec": -1.5,
            "maxJerkAcc": 1.0,
            "maxJerkDec": -1.0,
            "maxRPM": 1.0,
            "maxReverseRPM": -1.0,
            "negativeLimitIO": -1,
            "negativeLimitIOMode": 0,
            "positiveLimit": positiveLimit,
            "positiveLimitIO": -1,
            "positiveLimitIOMode": 0,
            "ratedRPM": 3000.0,
            "ratedReverseRPM": -3000.0,
            "ratedReverseSpeed": -180.0,
            "ratedSpeed": 180.0,
            "reducRatio": 100.0,
            "reverseLimit": reverseLimit,
            "robot": None,
        }

        self._send_command(self.CONSTANTS["JOINTPARAMETER"]["SET"], cmd_data)
        self.logger.info(
            f"关节 {axis} 参数已经设置, 关节正限位: {positiveLimit} °, 关节反限位: {reverseLimit} °"
        )

    def jointparameter_inquery(self, axis: int) -> dict:
        """
        查询关节参数

        Args:
            axis(int): 代表关节轴号

        Returns:
            dict: 关节参数
                - axis(int): 关节轴数
                - axisDirection(int): 关节实际方向
                - backLash(float): backLash
                - direction(int): 模型方向
                - encoderResolution(int): 编码器位数
                - isReduceRatioEnable(bool): 编码器是否经过减速机
                - maxAcc(float): 最大加速度
                - maxDec(float): 最大减速度
                - maxJerkAcc(float): 最大加加速度
                - maxJerkDec(float): 最大减减速度
                - maxRPM(float): 最大正转速
                - maxReverseRPM(float): 最大反转速
                - positiveLimit(float): 关节正限位
                - ratedRPM(float): 关节额定正转速
                - ratedReverseRPM(float): 关节额定反转速
                - ratedReverseSpeed(float): 关节额定反速度
                - ratedSpeed(float): 关节额定正速度
                - reducRatio(int): 减速比
                - reverseLimit(float): 关节反限位
        """
        if not (1 <= axis <= self.dof):
            self.logger.warning(f"关节轴数应为 1 ~ {self.dof} 的关节号, 当前为: {axis}")
            return

        self._send_command(self.CONSTANTS["JOINTPARAMETER"]["INQUIRE"], {"axis": axis})
        axis_num = self._return_get("20C7axis")
        axisDirection = self._return_get("20C7axisDirection")
        backLash = self._return_get("20C7backLash")
        direction = self._return_get("20C7direction")
        encoderResolution = self._return_get("20C7encoderResolution")
        isReduceRatioEnable = self._return_get("20C7isReduceRatioEnable")
        maxAcc = self._return_get("20C7maxAcc")
        maxDec = self._return_get("20C7maxDec")
        maxJerkAcc = self._return_get("20C7maxJerkAcc")
        maxJerkDec = self._return_get("20C7maxJerkDec")
        maxRPM = self._return_get("20C7maxRPM")
        maxReverseRPM = self._return_get("20C7maxReverseRPM")
        negativeLimitIO = self._return_get("20C7negativeLimitIO")
        negativeLimitIOMode = self._return_get("20C7negativeLimitIOMode")
        positiveLimit = self._return_get("20C7positiveLimit")
        positiveLimitIO = self._return_get("20C7positiveLimitIO")
        positiveLimitIOMode = self._return_get("20C7positiveLimitIOMode")
        ratedRPM = self._return_get("20C7ratedRPM")
        ratedReverseRPM = self._return_get("20C7ratedReverseRPM")
        ratedReverseSpeed = self._return_get("20C7ratedReverseSpeed")
        ratedSpeed = self._return_get("20C7ratedSpeed")
        reducRatio = self._return_get("20C7reducRatio")
        reverseLimit = self._return_get("20C7reverseLimit")
        result = {
            "axis": axis_num,
            "axisDirection": axisDirection,
            "backLash": backLash,
            "direction": direction,
            "encoderResolution": encoderResolution,
            "isReduceRatioEnable": isReduceRatioEnable,
            "maxAcc": maxAcc,
            "maxDec": maxDec,
            "maxJerkAcc": maxJerkAcc,
            "maxJerkDec": maxJerkDec,
            "maxRPM": maxRPM,
            "maxReverseRPM": maxReverseRPM,
            "negativeLimitIO": negativeLimitIO,
            "negativeLimitIOMode": negativeLimitIOMode,
            "positiveLimit": positiveLimit,
            "positiveLimitIO": positiveLimitIO,
            "positiveLimitIOMode": positiveLimitIOMode,
            "ratedRPM": ratedRPM,
            "ratedReverseRPM": ratedReverseRPM,
            "ratedReverseSpeed": ratedReverseSpeed,
            "ratedSpeed": ratedSpeed,
            "reducRatio": reducRatio,
            "reverseLimit": reverseLimit,
        }
        self.logger.info(f"查询关节参数")
        self.logger.info(f"> 查询参数关节: {axis_num} 关节")
        self.logger.info(f"> 关节正限位: {positiveLimit} 度")
        self.logger.info(f"> 关节反限位: {reverseLimit} 度")
        self.logger.info(f"> 减速比: {reducRatio}")
        self.logger.info(f"> 编码器位数: {encoderResolution}")
        self.logger.info(f"> 关节额定正转速: {ratedRPM} rpm")
        self.logger.info(f"> 关节额定反转速: {ratedReverseRPM} rpm")
        self.logger.info(f"> 最大正转速: {maxRPM} 倍数")
        self.logger.info(f"> 最大反转速: {maxReverseRPM} 倍数")
        self.logger.info(f"> 关节额定正速度: {ratedSpeed} 度/秒")
        self.logger.info(f"> 关节额定反速度: {ratedReverseSpeed} 度/秒")
        self.logger.info(f"> 最大加速度: {maxAcc} 倍数")
        self.logger.info(f"> 最大减速度: {maxDec} 倍数")
        self.logger.info(f"> 最大加加速度: {maxJerkAcc}")
        self.logger.info(f"> 最大减减速度: {maxJerkDec}")
        self.logger.info(f"> 关节实际方向: {axisDirection}")
        self.logger.info(f"> 模型方向: {direction}")
        self.logger.info(f"> 齿轮反向间隙: {backLash}")
        self.logger.info(f"> 编码器是否经过减速机: {isReduceRatioEnable}")
        return result

    def decareparameter_set(
        self,
        maxSpeed: float,
        maxAcc: float,
        maxDec: float,
        maxAttitudeVel: float,
        speedLimitMode: int,
    ) -> None:
        """
        设置机器人笛卡尔参数

        Args:
            maxSpeed(int): 最大速度, 范围[1,5000], 单位: mm/s
            maxAcc(int): 最大加速度, 范围[1,15], 单位: 倍数
            maxDec(int): 最大减速度, 范围[-15,-1], 单位: 倍数
            maxAttitudeVel(int): 姿态运动最大速度, 范围[1-1000], 单位: °/s
            speedLimitMode(int): 速度限制方式
                    - 0: 位姿
                    - 1: 位置

        Returns:
            None
        """
        cmd_data = {
            "maxSpeed": maxSpeed,
            "maxAcc": maxAcc,
            "maxDec": maxDec,
            "maxAttitudeVel": maxAttitudeVel,
            "speedLimitMode": speedLimitMode,
            "robot": None,
        }
        self._send_command(self.CONSTANTS["DECAREPARAMETER"]["SET"], cmd_data)
        self.logger.info(f"设置机器人笛卡尔参数")
        self.logger.info(f"> 最大速度: {maxSpeed} mm/s")
        self.logger.info(f"> 最大加速度: {maxAcc} 倍数")
        self.logger.info(f"> 最大减速度: {maxDec} 倍数")
        self.logger.info(f"> 姿态运动最大速度: {maxAttitudeVel} °/s")
        SpeedLimitMode_map = {0: "位姿", 1: "位置"}
        self.logger.info(f"> 速度限制方式: {SpeedLimitMode_map.get(speedLimitMode)}")
        state = self._return_get("3002state")
        if state == 1:
            self.logger.info(f"设置机器人笛卡尔参数 成功")
        else:
            self.logger.warning(f"设置机器人笛卡尔参数 失败")

    def decareparameter_inquire(self) -> dict:
        """
        查询机器人笛卡尔参数

        Args:
            None

        Returns:
            dict: 笛卡尔参数
                - maxSpeed: int, 最大速度, 单位: mm/s
                - maxAcc: int, 最大加速度, 单位: 倍数
                - maxDec: int, 最大减速度, 单位: 倍数
                - maxJerk: int, 最大加加速度, 单位: mm/s³
                - maxAttitudeVel: int, 姿态运动最大速度, 单位: °/s
                - speedLimitMode: int, 速度限制方式
                    - 0: 位姿
                    - 1: 位置
        """
        self._send_command(self.CONSTANTS["DECAREPARAMETER"]["INQUIRE"])
        maxAcc = self._return_get("20CAmaxAcc")
        maxAttitudeVel = self._return_get("20CAmaxAttitudeVel")
        maxDec = self._return_get("20CAmaxDec")
        maxSpeed = self._return_get("20CAmaxSpeed")
        speedLimitMode = self._return_get("20CAspeedLimitMode")
        maxJerk = self._return_get("20CAmaxJerk")
        result = {
            "maxSpeed": maxSpeed,
            "maxAcc": maxAcc,
            "maxDec": maxDec,
            "maxJerk": maxJerk,
            "maxAttitudeVel": maxAttitudeVel,
            "speedLimitMode": speedLimitMode,
        }
        self.logger.info(f"查询机器人笛卡尔参数")
        self.logger.info(f"> 最大速度: {maxSpeed} mm/s")
        self.logger.info(f"> 最大加速度: {maxAcc} 倍数")
        self.logger.info(f"> 最大减速度: {maxDec} 倍数")
        self.logger.info(f"> 最大加加速度: {maxJerk} mm/s³")
        self.logger.info(f"> 姿态运动最大速度: {maxAttitudeVel} °/s")
        SpeedLimitMode_map = {0: "位姿", 1: "位置"}
        self.logger.info(f"> 速度限制方式: {SpeedLimitMode_map.get(speedLimitMode)}")
        return result

    def robot_type_and_mapping_inquire(self) -> dict:
        """
        查询当前机器人类型及映射

        Args:
            None

        Returns:
            dict: 机器人类型及映射
                - sum: int, 机器人总数
                - servoSum: int, 伺服总数
                - robot: list,
                    - robotType: int, 机器人类型
                    - servoMap: list, 伺服映射
                        - 0, 表示虚拟伺服
                        - [1, 7], 表示真实伺服号
        """
        self._send_command(self.CONSTANTS["ROBOT_TYPE_AND_MAPPING"]["INQUIRE"])
        sum = self._return_get("2004sum")
        servoSum = self._return_get("2004servoSum")
        robot_info = self._return_get("2004robot")
        self.logger.info(f"查询当前机器人类型及映射")
        self.logger.info(f"> 机器人总数: {sum}")
        self.logger.info(f"> 伺服总数: {servoSum}")
        for i, robot in enumerate(robot_info):
            self.logger.info(f"> 机器人{i + 1}类型: {robot['robotType']}")
            self.logger.info(f"> 机器人{i + 1}映射: {robot['servoMap']}")

        return {"sum": sum, "servoSum": servoSum, "robot": robot_info}

    def robot_type_and_mapping_set(
        self, servoMap: List, robotType: str = "R_GENERAL_7S"
    ) -> bool:
        """
        设置当前机器人类型及映射

        Args:
            robotType(str): 机器人类型
            servoMap(list): 机器人映射
                - 0, 表示虚拟伺服
                - [1, 7], 表示真实伺服号
        Returns:
            bool: 是否设置成功
                - True: 设置成功
                - False: 设置失败
        """
        cmd_data = {
            "robot": [{"note": "", "robotType": robotType, "servoMap": servoMap}],
            "sum": 1,
        }
        self._send_command(self.CONSTANTS["ROBOT_TYPE_AND_MAPPING"]["SET"], cmd_data)
        self.logger.info(f"设置当前机器人类型及映射")
        self.logger.info(f"> 机器人类型: {robotType}")
        self.logger.info(f"> 机器人映射: {servoMap}")
        state = self._return_get("3002state")
        if state != None:
            self.logger.info(f"设置当前机器人类型及映射 成功")
            return True
        else:
            self.logger.warning(f"设置当前机器人类型及映射 失败")
            return False

    ########################################
    # NOTE API 接口（7000 端口）
    ########################################
    def motion_control(
        self,
        trajectory: List[List[float]],
        coord: str = "ACS",
        speed: int = 10,
        acc: int = 10,
        pl: int = 5,
        moveMode: str = "TeachSet",
        robot: int = 1,
    ) -> bool:
        """
        运动控制接口

        通信端口和用途的区别：
            1、6001 端口：通用控制端口，用于状态查询、参数设置、错误清除以及高级运动指令
                movj、movl、movc：这些是单点运动指令，每次只发送一个目标位置，适合精确控制单个动作；
                延迟较高，因为需要等待每条指令的响应；
                适合需要精确状态反馈的场景；
            2、7000 端口：运动控制端口，专门用于高性能运动控制
                motion_control：批量发送多个轨迹点，支持连续路径规划；
                延迟固定为 1ms，适合实时性要求高的连续运动；
                一次可以发送多个目标点，减少通信开销；

        注意：这个控制端口一定要记得在控制器上建立作业文件, 通过 6001 端口运行该文件之后才可使用！

        但不是必须使用示教器才能使用 7000 端口功能。"在控制器上建立作业文件"可以通过多种方式实现：
            1、通过示教器手动创建；
            2、通过 6001 端口的 API（如 jobfile_open、jobsend_done 等）创建和运行；

        为什么要这样做？
            安全机制：控制器需要知道上位机将要执行运动控制任务，以便分配必要的资源和进入正确的控制模式
            权限验证：确保上位机有权限使用高性能运动控制功能
            状态同步：确保控制器处于合适的运行状态以接受7000端口的指令

        建立作业文件步骤:
            1、在示教器界面上点击左侧“工程”, 点击“新建”
            2、在“程序名称”输入框中输入作业文件名称, 例如“tlibot”, 点击“确定”
            3、在“工程预览/程序指令”界面点击左下角的“插入”
            4、在指令类型中选择“条件控制类”, 然后在右边“指令”中选择“循环”, 点击“确定”
            5、在“工程预览/程序指令/指令插入/参数设定”界面, 点击“条件未设定”后点击“确定”
            6、界面回到了“工程预览/程序指令/指令插入/参数设定”界面后再点击“确定”
            7、在“工程预览/程序指令”界面继续点击左下角的“插入”
            8、在指令类型中选择“运动控制类”, 然后在右边“指令”中选择“外部点”, 点击“确定”
            9、在“工程预览/程序指令/指令插入/参数设定”界面, 修改合适参数(可以直接默认, 不做修改), 点击“确定”

        Args:
            target_vecs(List[List[float]]):
                - 轨迹运动: [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                            ......]
                - 关节运动: [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]]
            speed(int): 运动速度, 范围: 1-100
            acc(int): 运动加速度, 范围: 1-100
            pl(int): 平滑系数, 若不写, 则使用默认值 5, 范围: 1-5
            coord(str):
                - ACS: 关节坐标
                - MCS: 直角坐标
                  需要注意的是, 如果是直角坐标, 发送的数据为 6 个元素
                  比如：需要在直角坐标下移动到的点位坐标为 X、Y、Z 为 279.852 mm、0 mm、409.237 mm, A、B、C (单位弧度) 为 -3.141、0、0
                  trajectory = [[279.852, 0, 409.237, -3.141, 0, 0]]
                  轨迹为 trajectory = [[279.852, 0, 409.237, -3.141, 0, 0],
                                    [379.852, 0, 409.237, -3.141, 0, 0],
                                    [179.852, 0, 409.237, -3.141, 0, 0],
                                    [279.852, 0, 409.237, -3.141, 0, 0]]

            moveMode(str): 运动模式
                - 可选"TeachSet", "MOVJ", "MOVL", "MOVS", 默认值 "TeachSet"
            robot(int): 机器人号码

        Returns:
            bool: 运动控制是否成功
                - True: 成功
                - False: 失败
        """
        # 构建运动控制命令的数据结构，包含机器人编号、缓冲区清空标志、目标模式和配置参数
        cmd_data = {
            # 机器人编号，用于指定哪个机器人执行运动
            "robot": robot,
            # 清空缓冲区标志，1 表示清空之前的运动指令
            "clearBuffer": 1,
            # 目标模式，0 表示多点运动模式
            "targetMode": 0,
            # 配置参数，包含坐标系、速度、加速度、平滑系数和运动模式
            "cfg": {
                # 坐标系类型：ACS 表示关节坐标系，MCS 表示直角坐标系
                "coord": coord,
                # 运动速度百分比（1-100）
                "speed": speed,
                # 运动加速度百分比（1-100）
                "acc": acc,
                # 平滑系数，用于轨迹平滑过渡（1-5）
                "pl": pl,
                # 运动模式：TeachSet、MOVJ、MOVL、MOVS
                "moveMode": moveMode,
            },
            # 目标位置列表，将轨迹中的每个位置包装成字典格式
            "targetVec": [{"pos": pos} for pos in trajectory],
        }

        # 通过 7000 端口发送运动控制指令
        self._send_command(self.CONSTANTS["MULTI_POINT"], cmd_data)
        # 等待并获取控制器返回的执行结果
        success = self._return_get("1E03success")

        # 检查运动控制是否成功执行
        if success:
            # 记录成功的日志信息，显示发送的运动控制指令详情
            self.logger.info(f"已发送 7000 端口运动控制指令 (motion_control):")
            # 根据坐标系类型输出相应的日志信息
            if coord == "ACS":
                # case1 ACS 表示关节坐标系，记录关节运动路径
                self.logger.info(f"关节坐标 移动的路径如下:")
            elif coord == "MCS":
                # case2 MCS 表示直角坐标系，记录直角坐标运动路径
                self.logger.info(f"直角坐标 移动的路径如下:")
            # 输出具体的轨迹数据
            self.logger.info(f"{trajectory}")

            # 返回 True 表示运动控制成功
            return True
        else:
            # 如果运动控制失败，获取失败原因
            cause = self._return_get("1E03cause")
            # 根据不同的失败原因输出相应的错误日志
            if cause == "busy":
                # 控制器繁忙，有未完成的传输
                self.logger.error(f"当前有未传输完成的数据")
            elif cause == "timeout":
                # 数据传输超时
                self.logger.error(f"接收超时")
            elif cause == "dataErr":
                # 发送的数据格式错误
                self.logger.error(f"数据错误")
            elif cause == "termination":
                # 发送端提前终止了数据传输
                self.logger.error(f"发送端终止了正在传输的数据")

            # 返回 False 表示运动控制失败
            return False

    def set_servo_point_motion_control(self, switch: bool, robot: int = 1) -> bool:
        """
        开关伺服点位运动控制
        (该功能和"运动控制"功能 motion_control() 不可同时使用)

        Args:
            robot(int): 机器人号码
            switch(bool): 开关控制命令
                - True: 开启伺服点位运动控制
                - False: 关闭伺服点位运动控制

        Returns:
            bool: 开关伺服点位运动控制是否成功
                - True: 开启成功
                - False: 开启失败
        """
        # 验证输入参数 switch 是否为布尔类型，如果不是则记录警告信息并返回
        if not isinstance(switch, bool):
            # 记录类型错误的警告日志，显示实际收到的值和类型
            self.logger.warning(
                f"实际收到: {switch}, 类型: {type(switch)}, switch 必须是布尔类型"
            )
            # 参数类型错误时直接返回，不执行后续操作
            return

        # 向控制器发送开启/关闭伺服点位运动控制的命令
        self._send_command(
            self.CONSTANTS["SERVOCONTROL"][
                "OPEN"
            ],  # 使用常量字典中的 SERVOCONTROL.OPEN 命令字
            {
                "robot": robot,
                "switch": switch,
            },  # 发送的命令数据，包含机器人编号和开关状态
        )

        # 等待并获取控制器返回的执行结果
        success = self._return_get("1E11success")

        # 定义开关状态映射表，将布尔值转换为中文描述
        switch_map = {True: "开启", False: "关闭"}

        # 检查命令执行是否成功
        if success:
            # 延迟 0.5 秒，确保控制器状态稳定
            time.sleep(0.5)
            # 记录成功的日志信息，显示接收成功和当前状态
            self.logger.info(
                f"开关伺服点位运动控制状态设置: 接收成功; 状态: {switch_map.get(success, '未知')}"
            )
        else:
            # 如果命令执行失败，获取失败的具体原因
            cause = self._return_get("1E11cause")
            # 定义失败原因映射表，用于将错误代码转换为可读的中文描述
            cause_map = {  # 接收成功时为空
                "dataErr": "接收到的数据错误",
                "startupErr": "启动失败",
                "busy": "当前通道被占用",
            }
            # 记录失败的警告信息，显示失败原因和状态
            self.logger.warning(
                f"开关伺服点位运动控制状态设置: {cause_map.get(cause)}; 状态: {switch_map.get(success, '未知')}"
            )

        # 返回执行结果，成功返回 True，失败返回 False
        return success

    def servo_point_motion_control(
        self, end: int, sum: int, count: int, PosVec: List[List[float]], robot: int = 1
    ) -> bool:
        """
        伺服运动控制

        Args:
            robot(int): 机器人号码
            end(int):
                - 1: 停止之前的持续传输, 下面的数据可不发
                - 0: 可不发 end 参数值，0 表示不结束传输
            sum(int): 总共要发的机械臂运动帧数，用于分批传输大量数据
            count(int): 当前为第几帧，用于跟踪传输进度
            PosVec(List[List[float]]): 当前要发送的位置向量列表，包含机器人的关节角度数据
                - [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                    [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                    ......]

        Returns:
            - bool: 判断是否接收成功
                - True: 成功
                - False: 失败
        """

        # 构建发送给控制器的命令数据结构
        cmd_data = {
            # 机器人编号，指定目标机器人
            "robot": robot,
            # 结束标志，用于控制数据流
            "end": end,
            # 总帧数，用于分批传输控制
            "sum": sum,
            # 当前帧序号，用于跟踪传输进度
            "count": count,
            # 位置向量列表，包含要发送的关节角度数据
        }

        # 通过 7000 端口发送伺服运动控制命令
        self._send_command(self.CONSTANTS["SERVOCONTROL"]["MOVE"], cmd_data)

        # 等待并获取控制器返回的执行结果
        success = self._return_get("1E12success")

        # 检查命令是否成功执行
        if success:
            # 记录成功信息（注意：这里的信息过于简单，实际应用中应提供更多细节）
            self.logger.info("okok")
            # 返回 True 表示操作成功
            return True

        else:
            # 如果操作失败，获取失败原因
            cause = self._return_get("1E13cause")

            # 根据不同的失败原因记录相应的错误信息
            if cause == "notStart":
                # 如果未开启伺服点位运动控制模式，记录错误
                self.logger.error(f"未开启服点位运动控制模式")
            elif cause == "dataErr":
                # 如果数据格式错误，记录错误
                self.logger.error(f"数据错误")
            elif cause == "termination":
                # 如果发送端终止了数据传输，记录错误
                self.logger.error(f"发送端终止了正在传输的数据")
            elif cause == "cacheFull":
                # 如果控制器缓存区已满，记录错误
                self.logger.error(f"缓存区已满(最大缓存 6 条轨迹)")

            # 返回 False 表示操作失败
            return False

    # 以下的伺服控制还未测试（2403 版本的控制器还未支持）
    def open_servo_j(
        self, vmax: list[float], amax: list[float], jmax: list[float], robot: int = 1
    ):
        """
        打开关节跟踪模式

        Args:
            robot(int): 机器人号码
            vmax(list[float]): 速度约束,  单位: 度/秒
            amax(list[float]): 加速度约束,  单位: 度/秒^2
            jmax(list[float]): 加加速度约束,  单位: 度/秒^3

        Returns:
            None
        """
        self._send_command(
            self.CONSTANTS["SERVO_J_CONTROL"]["OPEN"],
            {"robot": robot, "vmax": vmax, "amax": amax, "jmax": jmax},
        )
        self.logger.info(f"打开关节跟踪模式")
        self.logger.info(f"> 速度约束: {vmax} °/s")
        self.logger.info(f"> 加速度约束: {amax} °/s²")
        self.logger.info(f"> 加加速度约束: {jmax} °/s³")
        return None

    def set_servo_j(self, q: List[float], robot: int = 1) -> None:
        """
        发送跟踪关节位置

        Args:
            robot(int): 机器人号码
            PosVec(List[float]): 机器人的关节角度

        Returns:
            None
        """
        self._send_command(
            self.CONSTANTS["SERVO_J_CONTROL"]["MOVE"], {"robot": robot, "q": q}
        )
        self.logger.debug(f"关节跟踪模式")
        self.logger.debug(f"> 机器人的关节角度: {q}")
        return None

    def stop_servo_j(self, robot: int = 1) -> None:
        """
        停止跟踪关节位置

        Args:
            robot(int): 机器人号码

        Returns:
            None
        """
        self._send_command(self.CONSTANTS["SERVO_J_CONTROL"]["STOP"], {"robot": robot})
        self.logger.info(f"停止关节跟踪模式")
        return None
