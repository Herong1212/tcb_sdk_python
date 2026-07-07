#!/usr/bin/env python3
"""
协作七自由度机械臂功能库
作者: 杜宇坤
文件名: robotarm_function.py
创建时间: 2025-04-07
修改时间: 2026-01-14
"""

from typing import List, Union
import logging, colorlog
from datetime import datetime
import time
import yaml
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "."))
sys.path.append(parent_dir)
from tl_kinematics import RobotKinematics


class RobotArmFunction:
    def __init__(self, color_log: bool = True):
        self.color_log = color_log
        self.stop_flag = False  # 用于打断检查线程
        self.move_complete = False  # 标记是否完成移动
        self.move_reached = False  # 用于http接口非阻塞情况下的判断是否运动到位
        self.ERROR_THRESHOLD = 0.05  # 误差阈值
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, ".."))

        self.log_path = self.parent_dir + "/log/"
        self.tcb_cfg_path = self.parent_dir + "/config/tcb_cfg.yaml"
        with open(self.tcb_cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.dof = self.cfg["DOF"]
        self.arm_type = self.cfg["ARM_TYPE"]
        self.arm_origin_type = self.cfg["ORIGIN_TYPE"]
        self.base_frame = "tl_robot_link0"
        if self.dof == 7:
            if self.arm_type == "TCB705_05":
                self.urdf_path = (
                    self.parent_dir
                    + "/models/TCB705_05N_urdf/urdf/TCB705_05N_urdf.urdf"
                )
            elif self.arm_type == "TCB710_06":
                self.urdf_path = (
                    self.parent_dir
                    + "/models/TCB710_06N_urdf/urdf/TCB710_06N_urdf.urdf"
                )

            if self.arm_origin_type == "default":
                self.end_frame = "tl_robot_link_end"
            elif self.arm_origin_type == "straight":
                self.end_frame = "tl_robot_link7"

        elif self.dof == 6:
            if self.arm_type == "TCB605_05":
                self.urdf_path = (
                    self.parent_dir
                    + "/models/TCB605_05N_urdf/urdf/TCB605_05N_urdf.urdf"
                )
            elif self.arm_type == "TCB610_06":
                self.urdf_path = (
                    self.parent_dir
                    + "/models/TCB610_06N_urdf/urdf/TCB610_06N_urdf.urdf"
                )
                print(f"urdf_path: ", {self.urdf_path})

            if self.arm_origin_type == "default":
                self.end_frame = "tl_robot_link_end"
            elif self.arm_origin_type == "straight":
                self.end_frame = "tl_robot_link6"

        self.kinematics = RobotKinematics(
            self.urdf_path, self.base_frame, self.end_frame
        )

        self._setup_logger()
        self.logger.info(f"RobotArmFunction 初始化成功")

    def _setup_logger(self):
        os.makedirs(self.log_path, exist_ok=True)

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

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        self.logger = logging.getLogger("RobotArmFunction")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            self.logger.addHandler(handler)
            self.logger.addHandler(file_handler)
        else:
            self.logger.propagate = False

    def _monitor_movement(self, robotarm6001, target_pos, coord, timeout=60):
        """
        监控机械臂是否到达目标位置
        """
        start_time = time.time()
        time.sleep(0.5)
        while not self.stop_flag:
            current_pos = robotarm6001.currentpos_inquiry(coord)

            reached = self._check_target_reached(current_pos, target_pos)
            if reached:
                self.move_complete = True
                return True
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.05)
        return False

    def _convert_to_float(
        self, data: Union[List[float], List[List[float]]]
    ) -> Union[List[float], List[List[float]]]:
        """
        将数据中的 int 类型转换为 float。
        """
        if isinstance(data, list):
            if all(isinstance(item, list) for item in data):
                # 对 List[List[float]] 类型数据进行递归转换
                return [self._convert_to_float(sublist) for sublist in data]
            else:
                # 对 List[float] 类型数据进行转换
                return [float(item) if isinstance(item, int) else item for item in data]
        return data

    def _validate_data(
        self, data: Union[List[float], List[List[float]]], coord=0
    ) -> Union[bool, tuple]:
        """
        验证输入数据是否为 List[float] 或 List[List[float]] 类型，
        同时将其中的 int 类型转换为 float。
        """
        # 先将数据中的 int 转换为 float
        data = self._convert_to_float(data)

        if not data or not isinstance(data, list):
            self.logger.error("输入数据为空或不是列表类型")
            return False, None, "Empty or invalid"

        if coord == 0:
            data_len = self.dof
        elif coord == 1:
            data_len = 6

        # 情况1: 双列表 (List[List[float]])
        if all(isinstance(item, list) for item in data):
            valid_sublists = True
            for sublist in data:
                if len(sublist) != data_len or not all(
                    isinstance(x, float) for x in sublist
                ):
                    valid_sublists = False
                    break

            if valid_sublists:
                self.logger.debug(
                    f"验证通过: List[List[float]] with {len(data)} points"
                )
                return True, len(data), "List[List[float]]"
            else:
                self.logger.error(
                    f"双列表数据无效: 子列表长度不为{data_len}或包含非浮点数"
                )
                return False, None, "Invalid List[List[float]]"

        # 情况2: 单列表 (List[float])
        elif all(isinstance(item, float) for item in data):
            if len(data) == data_len:
                self.logger.debug(f"验证通过: List[float] with {data_len} elements")
                return True, 1, "List[float]"
            else:
                self.logger.error(f"单列表数据无效: 长度={len(data)} (应为{data_len})")
                return False, None, "Invalid List[float]"

        # 未知类型
        self.logger.error(f"未知数据类型: {type(data[0]) if data else 'Empty'}")
        return False, None, "Unknown"

    def _check_target_reached(self, current_pos, target_pos):
        """判断是否到达目标点"""
        error = [
            abs(current - target) for current, target in zip(current_pos, target_pos)
        ]
        max_error = max(error)
        return max_error <= self.ERROR_THRESHOLD

    def _matrix_to_pose(self, matrix: np.ndarray):
        """
        将 4x4 齐次变换矩阵转换为 [x, y, z, rx, ry, rz], 其中旋转部分为 XYZ 顺序欧拉角(单位: 弧度)

        参数:
            matrix (np.ndarray): 4x4 齐次变换矩阵

        返回:
            list: [x, y, z, rx, ry, rz]，弧度制欧拉角
        """
        np.set_printoptions(suppress=True, precision=4)
        if matrix.shape != (4, 4):
            # raise ValueError("输入必须是4x4矩阵")
            self.logger.warning(f"输入必须是4x4矩阵")
            return

        # 平移部分
        x, y, z = matrix[0:3, 3]

        # 旋转部分
        rot_matrix = matrix[0:3, 0:3]
        r = R.from_matrix(rot_matrix)
        rz, ry, rx = r.as_euler(
            "zyx", degrees=False
        )  # 纳博特控制器欧拉角旋转方向为 zyx
        return [x, y, z, rx, ry, rz]

    # NOTE 示教模式初始化，只需要传入一个 RobotArmSDK 实例（包含端口信息）和运行速度（百分比）即可
    def robot_init_teach(self, robotarm6001, speed=35):
        """
        这个是通过示教模式下的直接上电, 要保持一直的上电状态，需要按照步骤: 设置示教模式 -> 确保下电 -> 最后上电
        (在示教模式上电之后可以进行点动操作)

        Args:
            robotarm6001(socket.socket): 6001端口连接的socket对象
            speed(int): 初始化的全局速度

        Returns:
            bool: 初始化状态
                - True: 初始化成功
                - False: 初始化失败
        """
        # 记录日志：开始机械臂初始化过程
        self.logger.info(f"机械臂初始化(robot_init_teach)")

        # 设置超时时间：10秒，防止初始化过程无限等待
        timeout = 10

        # 查询当前的操作模式状态
        # operation_mode: 0=示教模式, 1=运行模式
        operation_mode = robotarm6001.operation_mode_inquire()
        # 如果当前不是示教模式(0)，则切换到示教模式
        if operation_mode != 0:
            robotarm6001.operation_mode_set(0)
            time.sleep(0.5)

        # 查询伺服状态
        # servo_status: 0=停止, 1=就绪, 2=错误, 3=运行
        # robotarm.enable_status_set(0)
        servo_status = robotarm6001.servo_status_inquire()

        # case1: 如果伺服状态已经是运行状态(3)
        if servo_status == 3:
            self.logger.info("正在初始化...")
            # 查询当前机械臂速度
            current_speed = robotarm6001.speed_inquire()

            # 如果当前速度与目标速度不同
            if current_speed != speed:
                # 重新设置操作模式为示教模式（确保状态正确）
                robotarm6001.operation_mode_set(0)
                # 设置新的速度值
                robotarm6001.speed_set(speed)

                # 重新查询伺服状态，准备进入状态同步循环
                servo_status = robotarm6001.servo_status_inquire()
                # 记录开始时间用于超时判断
                start_time = time.time()

                # 循环等待直到伺服状态变为运行(3)或超时
                while servo_status != 3:
                    # 切换模式，先确保是示教模式
                    robotarm6001.operation_mode_set(0)
                    # 清除可能存在的错误状态
                    robotarm6001.fault_reset()
                    # 伺服就绪
                    robotarm6001.servo_status_set(1)
                    # 设置全局速度，这里设置的是示教模式的速度
                    robotarm6001.speed_set(speed)
                    # 全局速度查询
                    robotarm6001.speed_inquire()

                    # 先下电再上电（确保电源状态正确）
                    robotarm6001.enable_status_set(0)
                    robotarm6001.enable_status_set(1)
                    # 等待0.5秒
                    time.sleep(0.5)

                    # 重新查询伺服状态
                    servo_status = robotarm6001.servo_status_inquire()

                    # 检查是否超时
                    if time.time() - start_time > timeout:
                        self.logger.error("等待机械臂初始化超时")
                        return False
                self.logger.info("初始化结束!!")

            # 如果当前速度已经匹配，但仍需确保状态同步
            else:
                servo_status = robotarm6001.servo_status_inquire()
                start_time = time.time()
                while servo_status != 3:
                    robotarm6001.operation_mode_set(0)
                    robotarm6001.fault_reset()
                    robotarm6001.servo_status_set(1)
                    robotarm6001.speed_set(speed)
                    robotarm6001.speed_inquire()
                    robotarm6001.enable_status_set(0)
                    robotarm6001.enable_status_set(1)
                    time.sleep(0.5)
                    servo_status = robotarm6001.servo_status_inquire()
                    if time.time() - start_time > timeout:
                        self.logger.error("等待机械臂初始化超时")
                        return False
                self.logger.info("初始化结束!!")

        # case2: 伺服状态不是就绪状态（需要执行完整初始化流程）
        else:
            self.logger.info("正在初始化...")

            # 基础状态设置
            robotarm6001.operation_mode_set(0)  # 设置示教模式
            robotarm6001.servo_status_set(1)  # 使能伺服
            robotarm6001.enable_status_set(0)  # 先下电
            robotarm6001.enable_status_set(1)  # 再上电

            # 查询当前速度
            current_speed = robotarm6001.speed_inquire()
            # 如果速度不匹配
            if current_speed != speed:
                # 设置新速度
                robotarm6001.speed_set(speed)

                # 进入状态同步循环
                servo_status = robotarm6001.servo_status_inquire()
                start_time = time.time()
                while servo_status != 3:
                    robotarm6001.operation_mode_set(0)
                    robotarm6001.fault_reset()
                    robotarm6001.servo_status_set(1)
                    robotarm6001.speed_set(speed)
                    robotarm6001.speed_inquire()
                    robotarm6001.enable_status_set(0)
                    robotarm6001.enable_status_set(1)
                    time.sleep(0.5)
                    servo_status = robotarm6001.servo_status_inquire()
                    if time.time() - start_time > timeout:
                        self.logger.error("等待机械臂初始化超时")
                        return False
                self.logger.info("初始化结束!!")

            # 速度匹配但状态需要同步
            else:
                servo_status = robotarm6001.servo_status_inquire()
                start_time = time.time()
                while servo_status != 3:
                    robotarm6001.operation_mode_set(0)
                    robotarm6001.fault_reset()
                    robotarm6001.servo_status_set(1)
                    robotarm6001.speed_set(speed)
                    robotarm6001.speed_inquire()
                    robotarm6001.enable_status_set(0)
                    robotarm6001.enable_status_set(1)
                    time.sleep(0.5)
                    servo_status = robotarm6001.servo_status_inquire()
                    if time.time() - start_time > timeout:
                        self.logger.error("等待机械臂初始化超时")
                        return False
                self.logger.info("初始化结束!!")

        # 返回初始化成功状态
        return True

    # NOTE 运行模式初始化
    def robot_init_repeat(self, robotarm6001, speed=35):
        """
        这个是通过运行模式下的直接上电，需要按照步骤: 查询伺服状态 -> 设置运行模式

        Args:
            robotarm6001(socket.socket): 6001端口连接的socket对象
            speed(int): 初始化的全局速度

        Returns:
            bool: 初始化状态
                - True: 初始化成功
                - False: 初始化失败
        """
        # 记录日志：开始运行模式初始化
        self.logger.info(f"机械臂初始化(robot_init_repeat)")

        # 设置超时时间：10秒，防止初始化过程无限等待
        timeout = 10

        # 查询当前伺服状态（0=停止, 1=就绪, 2=错误, 3=运行）
        servo_status = robotarm6001.servo_status_inquire()

        # case1: 如果伺服状态已经是运行状态(3)
        if servo_status == 3:
            self.logger.info("正在初始化...")

            # 确保任何正在运行的作业文件停止。这是为了安全，避免新任务与旧任务冲突
            robotarm6001.stop_job_run()

            # 确保队列模式（directmotion）关闭
            # directmotion_mode 是另一种运动控制模式，需要先关闭
            if robotarm6001.directmotion_mode_inquire() == True:
                robotarm6001.directmotion_mode_set(False)

            # 设置操作模式为运行模式
            # operation_mode: 0=示教模式, 1=远程模式, 2=运行模式
            if robotarm6001.operation_mode_inquire() != 2:
                robotarm6001.operation_mode_set(2)

            # 在运行模式下设置全局速度
            if robotarm6001.speed_inquire() != speed:
                robotarm6001.speed_set(speed)

                # 重新查询伺服状态，准备进入同步循环
                servo_status = robotarm6001.servo_status_inquire()
                start_time = time.time()

                while servo_status != 3:
                    # 先切换到示教模式再切换回运行模式，为了确保模式设置生效
                    robotarm6001.operation_mode_set(0)
                    robotarm6001.operation_mode_set(2)

                    # 再次确认速度设置
                    if robotarm6001.speed_inquire() != speed:
                        robotarm6001.speed_set(speed)

                    time.sleep(0.5)

                    # 超时检查
                    servo_status = robotarm6001.servo_status_inquire()
                    if time.time() - start_time > timeout:
                        self.logger.error("等待机械臂初始化超时")
                        return False

                self.logger.info("初始化结束!!")

            # 速度已经匹配，但仍需状态同步
            else:
                servo_status = robotarm6001.servo_status_inquire()
                start_time = time.time()
                while servo_status != 3:
                    robotarm6001.operation_mode_set(0)
                    robotarm6001.operation_mode_set(2)
                    if robotarm6001.speed_inquire() != speed:
                        robotarm6001.speed_set(speed)
                    time.sleep(0.5)
                    servo_status = robotarm6001.servo_status_inquire()
                    if time.time() - start_time > timeout:
                        self.logger.error("等待机械臂初始化超时")
                        return False
                self.logger.info("初始化结束!!")

        # case2: 伺服状态不是运行状态（0,1,2）
        else:
            self.logger.info("正在初始化...")

            # 清除可能存在的错误状态
            robotarm6001.fault_reset()

            # 使能伺服系统（设置为 1，即就绪状态）
            robotarm6001.servo_status_set(1)

            # 确保打开的作业文件停止运行
            robotarm6001.stop_job_run()

            # 确保队列模式关闭
            if robotarm6001.directmotion_mode_inquire() == True:
                robotarm6001.directmotion_mode_set(False)

            # 设置运行模式
            if robotarm6001.operation_mode_inquire() != 2:
                robotarm6001.operation_mode_set(2)

            # 在运行模式下设置全局速度
            if robotarm6001.speed_inquire() != speed:
                robotarm6001.speed_set(speed)

                # 状态同步循环
                servo_status = robotarm6001.servo_status_inquire()
                start_time = time.time()
                while servo_status != 3:
                    robotarm6001.operation_mode_set(0)
                    robotarm6001.operation_mode_set(2)
                    if robotarm6001.speed_inquire() != speed:
                        robotarm6001.speed_set(speed)
                    time.sleep(0.5)
                    servo_status = robotarm6001.servo_status_inquire()
                    if time.time() - start_time > timeout:
                        self.logger.error("等待机械臂初始化超时")
                        return False
                self.logger.info("初始化结束!!")

            # 速度匹配的状态同步
            else:
                servo_status = robotarm6001.servo_status_inquire()
                start_time = time.time()
                while servo_status != 3:
                    robotarm6001.operation_mode_set(0)
                    robotarm6001.operation_mode_set(2)
                    if robotarm6001.speed_inquire() != speed:
                        robotarm6001.speed_set(speed)
                    time.sleep(0.5)
                    servo_status = robotarm6001.servo_status_inquire()
                    if time.time() - start_time > timeout:
                        self.logger.error("等待机械臂初始化超时")
                        return False
                self.logger.info("初始化结束!!")

        # 额外等待 1.5 秒，确保所有设置完全生效
        time.sleep(1.5)

        # 返回初始化成功
        return True

    def robot_stop(self, robotarm6001):
        """
        停止机械臂当前操作, 恢复机械臂到示教模式, 并且伺服下电状态

        Args:
            robotarm6001(socket.socket): 6001端口连接的socket对象

        Returns:
            None
        """
        self.logger.info(f"机械臂停止(robot_stop)")
        self.stop_flag = True  # 设置标志以停止监控线程
        robotarm6001.stop_job_run()
        robotarm6001.operation_mode_set(0)
        robotarm6001.enable_status_set(0)
        robotarm6001.servo_status_set(0)
        self.stop_flag = False
        self.logger.info("机械臂已停止操作")

    def robot_direction_move(self, robotarm6001, axis, distance, frame="base"):
        """
        机械臂末端直线移动，可选择基坐标系或工具坐标系作为参考系

        Args:
            robotarm6001 (socket.socket): 6001端口连接的socket对象
            axis (str): 移动方向
                - "x": x轴方向
                - "y": y轴方向
                - "z": z轴方向
            distance (float): 需要移动的距离, 单位 米(m)，正为正向，负为反向
            frame (str): 参考坐标系
                - "base": 基坐标系
                - "tool": 工具坐标系(末端)

        Returns:
            bool: 运动状态
                - True: 运动结束
                - False: 运动故障，超时
        """
        self.logger.info(f"机械臂门型运动(robot_direction_move)")
        current_pos = [round(angle, 4) for angle in robotarm6001.currentpos_inquiry(0)]
        T_EndToBase = np.array(self.kinematics.fkine(current_pos)).astype(float)

        axis = axis.lower()
        if axis not in ["x", "y", "z"]:
            raise ValueError("Invalid axis input. Please use 'x', 'y', or 'z'.")

        if frame == "base":
            # 基坐标系方向
            if axis == "x":
                T_EndToBase[0, 3] += distance
            elif axis == "y":
                T_EndToBase[1, 3] += distance
            elif axis == "z":
                T_EndToBase[2, 3] += distance

        elif frame == "tool":
            # 工具坐标系(末端)方向
            R_EndToBase = T_EndToBase[:3, :3]
            delta_local = {
                "x": np.array([distance, 0, 0]),
                "y": np.array([0, distance, 0]),
                "z": np.array([0, 0, distance]),
            }[axis]
            delta_base = R_EndToBase @ delta_local
            T_EndToBase[:3, 3] += delta_base
        try:
            traj = self.kinematics.straightline(T_EndToBase, current_pos)
            # status = self.robot_move(robotarm6001, traj)
            return traj
        except Exception as e:
            self.logger.error(f"机械臂门型运动规划失败: {e}")
            return []

    def robot_queue_move(
        self, robotarm6001, trajectory, speed=35, acc=70, dec=70, pl=5
    ):
        """
        队列运动控制
        要注意的是队列运动控制下的速度不是示教模式下的速度，而是开启直接控制运动模式后进去特殊的运行模式下的速度。

        Args:
            robotarm6001(socket.socket): 6001端口连接的socket对象
            trajectory(List[float]、List[List[float]]): 一组关节角度或者轨迹
            speed(int): 队列运动下的速度值
            acc(int): 加速度, 范围1-100
            dec(int): 减速度, 范围1-100
            pl(int): 平滑系数, 范围0-5

        Returns:
            bool: 运动状态
                - True: 运动结束
                - False: 运动故障，超时
        """
        self.move_complete = False
        self.logger.info(f"机械臂队列运动(robot_queue_move)")

        queue_status = robotarm6001.directmotion_mode_inquire()
        if queue_status:
            self.logger.info("机械臂已经在队列运动模式")
        else:
            robotarm6001.directmotion_mode_set(True)

        timeout = 45
        current_speed = robotarm6001.speed_inquire()
        if current_speed != speed:
            robotarm6001.speed_set(speed)
        status, data_len, joint_data_type = self._validate_data(trajectory)

        if not status:
            self.logger.error(
                "检查一下输入的数据是否为列表或者双列表,如果是列表(双列表),看看(子列表)元素个数是否正确"
            )
            return False

        if joint_data_type == "List[float]":
            current_pos = robotarm6001.currentpos_inquiry(0)
            reached = self._check_target_reached(current_pos, trajectory)
            if reached:
                self.logger.info("机械臂已到达目标点")
                self.move_reached = True
            else:
                robotarm6001.directmotion_insert_instrvec(
                    [trajectory], acc=acc, dec=dec, pl=pl
                )
                self.logger.info(f"已发送一组队列")
                monitor_thread = threading.Thread(
                    target=self._monitor_movement,
                    args=(robotarm6001, trajectory, 0, timeout),
                )
                monitor_thread.start()
                monitor_thread.join()

                if self.move_complete:
                    self.logger.info("机械臂已到达目标点")
                    self.move_reached = True
                    self.move_complete = False
                    return True
                else:
                    self.logger.error("等待机械臂到达目标超时或者中断")
                    self.move_reached = False
                    return False

        elif joint_data_type == "List[List[float]]":
            robotarm6001.directmotion_insert_instrvec(
                trajectory, acc=acc, dec=dec, pl=pl
            )

            monitor_thread = threading.Thread(
                target=self._monitor_movement,
                args=(robotarm6001, trajectory[-1], 0, timeout),
            )
            monitor_thread.start()
            monitor_thread.join()

            if self.move_complete:
                self.logger.info("机械臂已到达目标点")
                self.move_reached = True
                self.move_complete = False
                return True
            else:
                self.logger.error("等待机械臂到达目标超时或者中断")
                self.move_reached = False
                return False
        self.logger.info(">>>>>> 运动完成!")
        return True

    def motion_control_move(
        self, robotarm6001, robotarm7000, traj, jobname="tlibot", coord=0, timeout=25
    ):
        """
        7000端口的运动控制

        Args:
            robotarm6001(socket.socket): 6001端口连接的socket对象
            robotarm7000(socket.socket): 7000端口连接的socket对象
            trajectory(List[float]、List[List[float]]): 一组关节角度或者轨迹
            jobname(str): 作业文件名称
            timeout(int): 超时时间

        Returns:
            bool: 运动状态
                - True: 运动结束
                - False: 运动故障，超时
        """
        self.move_complete = False
        self.logger.info(f"机械臂7000端口运动控制(motion_control_move)")

        if coord == 0:
            self.logger.info(f"关节坐标下的运动控制")
            coord_ = "ACS"
        elif coord == 1:
            self.logger.info(f"直角坐标下的运动控制")
            coord_ = "MCS"

        robotarm6001.stop_job_run()
        robotarm6001.jobsend_done(jobname)
        status, data_len, joint_data_type = self._validate_data(traj, coord=coord)
        if not status:
            self.logger.error(
                "检查一下输入的数据是否为列表或者双列表,如果是列表(双列表),看看(子列表)元素个数是否正确"
            )
            return False

        if joint_data_type == "List[float]":
            target_pos = traj
            current_pos = robotarm6001.currentpos_inquiry(coord)
            reached = self._check_target_reached(current_pos, target_pos)

            if reached:
                self.logger.info("机械臂已到达目标点")
                self.move_reached = True
                return True
            else:
                robotarm7000.motion_control([traj], coord_)
                monitor_thread = threading.Thread(
                    target=self._monitor_movement,
                    args=(robotarm6001, target_pos, coord, timeout),
                )
                monitor_thread.start()
                monitor_thread.join()
                if self.move_complete:
                    self.logger.info("机械臂已到达目标点")
                    self.move_reached = True
                    self.move_complete = False
                    return True
                else:
                    self.logger.error("等待机械臂到达目标超时或者中断")
                    self.move_reached = False
                    return False

        elif joint_data_type == "List[List[float]]":
            target_pos = traj[-1]
            current_pos = robotarm6001.currentpos_inquiry(coord)

            robotarm7000.motion_control(traj, coord_)
            monitor_thread = threading.Thread(
                target=self._monitor_movement,
                args=(robotarm6001, target_pos, coord, timeout),
            )
            monitor_thread.start()
            monitor_thread.join()
            if self.move_complete:
                self.logger.info("机械臂已到达目标点")
                self.move_reached = True
                self.move_complete = False
                return True
            else:
                self.logger.error("等待机械臂到达目标超时或者中断")
                self.move_reached = False
                return False


if __name__ == "__main__":
    RobotArmFunction()
