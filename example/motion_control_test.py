#!/usr/bin/env python3
import sys, os
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from robotarm_sdk import RobotArmSDK
from utils.robotarm_function import RobotArmFunction


class MotionControlTest:
    def __init__(self):
        self.ip = "192.168.1.13"  # 修改为实际机械臂的IP地址
        self.speed = 30

        self.robot_func = RobotArmFunction(color_log=True)
        self.arm_6001 = RobotArmSDK(self.ip, 6001, log_level=0, color_log=True)
        self.arm_7000 = RobotArmSDK(self.ip, 7000, log_level=0, color_log=True)

        self.arm_6001.connect()
        self.arm_7000.connect()

        # self.q_zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.q_zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 DOF

    def arm_init(self, mode):
        print(f"机械臂初始化")

        # 如果是示教模式，调用 robot_init_teach()
        if mode == "teach":
            self.robot_func.robot_init_teach(
                self.arm_6001, self.speed
            )  # 示教模式下上电

        # 如果是运行模式（repeat），调用 robot_init_repeat()
        elif mode == "repeat":
            self.robot_func.robot_init_repeat(
                self.arm_6001, self.speed
            )  # 运行模式下上电

    # NOTE 回原点测试
    def go_home_test(self):
        print(f"机械臂回原点")

        self.arm_init("repeat")
        self.arm_6001.go_home()  # 通过 6001 端口（通用控制端口）向机械臂控制器发送回原点指令

        time.sleep(2)  # 等待2秒，给机械臂执行回原点动作的时间

    # NOTE 关节空间运动测试
    def movj_test(self):
        print(f"机械臂关节空间运动")

        self.arm_init("repeat")
        # 定义第一个目标关节位置
        # self.q1 = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.q1 = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
        # 定义第二个目标关节位置
        # self.q2 = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.q2 = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 移动到 q1，block=1: 设置为阻塞模式(同步)，程序会等待运动完成后再继续执行
        self.arm_6001.movj(vel=10, coord=0, pos=self.q1, block=1)
        # 移动到 q2
        self.arm_6001.movj(vel=10, coord=0, pos=self.q2, block=1)
        # 移动到 q_zero
        self.arm_6001.movj(vel=10, coord=0, pos=self.q_zero, block=1)

        # 定义一个笛卡尔坐标系下的目标位置
        q_test = [379.852, 0, 209.237, 2.40, 0, 0]
        # 1、关节插补到笛卡尔位置
        # ps 目标解析：控制器首先将提供的笛卡尔坐标 [X, Y, Z, Rx, Ry, Rz] 通过 IK 解算为对应的关节角度值；
        # ps 路径规划：获得目标关节角度后，控制器使用关节空间插补方式规划路径 - 即每个关节独立地从当前位置移动到目标位置；
        # ps 执行特点: 末端执行器的路径不是直线，而是由各关节独立运动合成的曲线路径；
        self.arm_6001.movj(vel=10, coord=1, pos=q_test, block=1)

        # 2、笛卡尔插补到笛卡尔位置 (直线运动)
        # self.arm_6001.movel(vel=100, coord=1, pos=q_test, block=1)

        # 最后回到零点位置
        self.arm_6001.movj(vel=10, coord=0, pos=self.q_zero, block=1)

        print(f"机械臂关节空间运动完成！！！")

    def movl_test(self):
        """
        执行机械臂笛卡尔空间直线运动测试。

        该函数演示了机械臂在笛卡尔空间中的直线运动控制，包括：
        1. 初始化机械臂到运行模式
        2. 先通过关节空间运动回到零点位置
        3. 执行第一个直线运动到指定的关节角度位置
        4. 执行第二个直线运动到指定的笛卡尔坐标位置

        坐标系说明：
        - coord=0: 关节空间坐标系（使用关节角度）
        - coord=1: 笛卡尔空间坐标系（使用[x, y, z, rx, ry, rz]格式）

        Args:
            self: 类实例引用，包含机械臂对象和相关配置

        Returns:
            None: 该函数不返回任何值，仅执行机械臂运动控制
        """
        # 打印提示信息，表明开始执行笛卡尔空间直线运动测试
        print(f"机械臂笛卡尔空间直线运动")

        # 初始化机械臂到运行模式（repeat mode），准备执行自动化任务
        self.arm_init("repeat")

        # 定义目标关节角度位置 q3，设置第1个关节为40度，其余关节保持0度
        self.q3 = [40.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 通过关节空间运动（movj）以100%速度回到零点位置，coord=0表示关节坐标系，block=1表示阻塞等待运动完成
        self.arm_6001.movj(vel=100, coord=0, pos=self.q_zero, block=1)

        # 执行直线运动（movl）到q3位置，虽然使用 movl 但 coord=0 表示仍在关节空间，实际路径为关节插补
        self.arm_6001.movl(vel=100, coord=0, pos=self.q3, block=1)

        # 定义笛卡尔空间目标位置，格式为[x, y, z, rx, ry, rz]，其中x=379.852mm, y=0mm, z=409.237mm, rx=-3.141rad, ry=0rad, rz=0rad
        # q_test = [379.852, 0, 409.237, -3.141, 0, 0]

        # 在笛卡尔空间执行真正的直线运动，coord=1表示笛卡尔坐标系，末端执行器将沿直线路径移动到目标位置
        # self.arm_6001.movl(vel=100, coord=1, pos=q_test, block=1)

        # 最后回到零点位置
        self.arm_6001.movj(vel=100, coord=0, pos=self.q_zero, block=1)

    def movc_test(self):
        """
        执行机械臂笛卡尔空间圆弧运动测试。

        该函数演示了机械臂的圆弧运动控制功能，通过三个点定义圆弧路径：
        - posOne: 起始点
        - posTwo: 中间点（用于确定圆弧方向和曲率）
        - posThree: 终点

        测试包含两种模式：
        1. 关节空间圆弧运动（coord=0）：在关节角度空间进行圆弧插补
        2. 笛卡尔空间圆弧运动（coord=1）：在笛卡尔坐标空间进行真正的圆弧路径规划

        isFull 参数说明：
        - isFull=False: 执行从起始点到中间点的半圆弧
        - isFull=True: 执行完整的圆弧路径（起始点 ——> 中间点 ——> 终点）

        Args:
            self: 类实例引用，包含机械臂对象和相关配置

        Returns:
            None: 该函数不返回任何值，仅执行机械臂运动控制
        """
        # 打印提示信息，表明开始执行笛卡尔空间圆弧运动测试
        print(f"机械臂笛卡尔空间圆弧运动")

        # 初始化机械臂到运行模式（repeat mode），准备执行自动化任务
        self.arm_init("repeat")

        # 定义第一个目标关节角度位置 q4，设置第1个关节为10度，其余关节保持0度
        self.q4 = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # 定义第二个目标关节角度位置 q5，设置第2个关节为10度，其余关节保持0度
        self.q5 = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 在关节空间执行半圆弧运动（isFull=False），从零点 ——> q4 ——> q5，coord=0 表示关节坐标系
        self.arm_6001.movc(
            vel=100,
            coord=0,
            isFull=False,
            posOne=self.q_zero,
            posTwo=self.q4,
            posThree=self.q5,
            block=1,
        )
        # 在关节空间执行完整圆弧运动（isFull=True），从零点 ——> q4 ——> q5，coord=0 表示关节坐标系
        self.arm_6001.movc(
            vel=100,
            coord=0,
            isFull=True,
            posOne=self.q_zero,
            posTwo=self.q4,
            posThree=self.q5,
            block=1,
        )

        # 定义笛卡尔空间第一个目标位置 [x, y, z, rx, ry, rz]，x=279.852mm, y=0mm, z=409.237mm
        pos1 = [279.852, 0, 409.237, -3.141, 0, 0]
        # 定义笛卡尔空间第二个目标位置，y坐标增加到200mm，形成圆弧的中间点
        pos2 = [279.852, 200, 409.237, -3.141, 0, 0]
        # 定义笛卡尔空间第三个目标位置，x坐标减少到179.852mm，作为圆弧的终点
        pos3 = [179.852, 0, 409.237, -3.141, 0, 0]

        # 在笛卡尔空间执行半圆弧运动（isFull=False），从 pos1 ——> pos2 ——> pos3，coord=1表示笛卡尔坐标系
        # self.arm_6001.movc(
        #     vel=100,
        #     coord=1,
        #     isFull=False,
        #     posOne=pos1,
        #     posTwo=pos2,
        #     posThree=pos3,
        #     block=1,
        # )
        # 在笛卡尔空间执行完整圆弧运动（isFull=True），从pos1 ——> pos2 ——> pos3，实现完整的圆弧轨迹
        # self.arm_6001.movc(
        #     vel=100,
        #     coord=1,
        #     isFull=True,
        #     posOne=pos1,
        #     posTwo=pos2,
        #     posThree=pos3,
        #     block=1,
        # )

    def jog_operation_test(self):
        """
        执行机械臂点动操作测试。

        该函数演示了机械臂在示教模式下的点动（Jog）操作功能，通过逐个关节进行正向和反向点动测试。
        点动操作允许操作员手动控制每个关节的运动方向和速度，常用于位置调试和路径规划。

        测试流程：
        1. 初始化机械臂到示教模式（teach mode）
        2. 回到原点位置作为起始点
        3. 对每个关节（1-7）依次执行：
           - 正向点动0.4秒
           - 停止并等待1秒
           - 反向点动0.4秒
           - 停止并等待1秒
        4. 最后回到原点位置结束测试

        注意：点动操作必须在示教模式下进行，这是安全要求。

        Args:
            self: 类实例引用，包含机械臂对象和相关配置

        Returns:
            None: 该函数不返回任何值，仅执行机械臂点动操作
        """
        # 打印提示信息，表明开始执行机械臂点动操作测试
        print(f"机械臂点动")

        # 初始化机械臂到示教模式（teach mode），点动操作必须在示教模式下进行
        self.arm_init("teach")

        # 发送回原点指令，将机械臂移动到预设的零点位置作为测试起始点
        self.arm_6001.go_home()

        # time.sleep(3)  # 注释掉的延时代码，实际使用状态查询替代固定延时

        # * 点动需要在示教模式下进行（此注释保留原有说明）
        # 循环遍历所有 dof 个关节（joint 索引 0-(dof-1)，对应关节编号1-dof）
        for joint in range(self.arm_6001.dof):
            # 对当前关节执行正向点动操作，joint+1 转换为 1-based 关节编号，方向参数 1 表示正向
            self.arm_6001.jog_operation_move(joint + 1, 1)
            # 延时0.4秒，让关节有足够时间进行点动运动
            time.sleep(0.4)

            # 停止当前关节的点动操作
            self.arm_6001.jog_operation_stop(joint + 1)

            # 延时1秒，确保关节完全停止后再进行下一步操作
            time.sleep(1)

            # 对当前关节执行反向点动操作，方向参数 -1 表示反向
            self.arm_6001.jog_operation_move(joint + 1, -1)

            # 延时0.4秒，让关节有足够时间进行反向点动运动
            time.sleep(0.4)

            # 停止当前关节的点动操作
            self.arm_6001.jog_operation_stop(joint + 1)

            # 延时1秒，确保关节完全停止后再测试下一个关节
            time.sleep(1)

        # 测试完成后，发送回原点指令，将机械臂恢复到初始位置
        self.arm_6001.go_home()

    def directmotion_mode_test(self):
        """
        执行机械臂直接运动模式（队列模式）测试。

        该函数演示了机械臂的直接运动模式（Direct Motion Mode），也称为队列模式。
        在此模式下，可以连续插入多个运动指令到控制器队列中，控制器会按顺序自动执行。

        测试流程：
        1. 初始化机械臂到运行模式（repeat mode）
        2. 启用直接运动模式（如果尚未启用）
        3. 设置运动速度参数
        4. 定义两条轨迹数据（traj1和traj2）
        5. 演示队列模式的各种控制操作：
           - 插入轨迹指令
           - 暂停运动
           - 继续运动
           - 停止但保持伺服使能
           - 返回零点位置
        6. 最后演示封装的队列运动功能

        直接运动模式特点：
        - 支持连续轨迹点插入
        - 可以暂停/继续/停止控制
        - 适用于预定义路径的批量执行

        Args:
            self: 类实例引用，包含机械臂对象和相关配置

        Returns:
            None: 该函数不返回任何值，仅执行机械臂直接运动模式测试
        """
        # 打印提示信息，表明开始执行机械臂直接运动模式（队列模式）测试
        print(f"机械臂直接运动模式（队列）")

        # 初始化机械臂到运行模式（repeat mode），直接运动模式需要在运行模式下执行
        self.arm_init("repeat")

        # 查询当前直接运动模式的状态（True=已启用，False=未启用）
        status = self.arm_6001.directmotion_mode_inquire()

        # 如果直接运动模式未启用，则启用该模式
        if status == False:
            self.arm_6001.directmotion_mode_set(True)

        # 设置全局运动速度，注意必须在设置对应模式之后才能设置速度参数
        self.arm_6001.speed_set(self.speed)  # 需要在设置对应模式之后设置速度

        # 定义第一条轨迹数据 traj1，包含一个目标位置点 [30,0,0,0,0,0,0]（关节角度）
        self.traj1 = [[30.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        # 定义第二条轨迹数据 traj2，包含三个连续的目标位置点
        self.traj2 = [
            [0.0, 0.0, 30.0, 0.0, 0.0, 0.0],  # 第一个点：第3关节旋转30度
            [40.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 第二个点：第1关节旋转40度
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],  # 第三个点：第2关节旋转10度
        ]

        # 向直接运动队列中插入第一条轨迹 traj1，设置加速度=100%，减速度=100%，平滑度=5
        self.arm_6001.directmotion_insert_instrvec(self.traj1, acc=100, dec=100, pl=5)
        # 延时1.5秒，等待轨迹执行完成
        time.sleep(1.5)

        # 暂停当前的直接运动模式（暂停执行队列中的指令）
        self.arm_6001.directmotion_mode_suspend()
        # 延时1秒，确保暂停操作完成
        time.sleep(1)

        # 继续执行直接运动模式（从暂停处继续执行）
        self.arm_6001.directmotion_mode_start()

        # 向队列中插入第二条轨迹 traj2，同样设置加速度、减速度和平滑度参数
        self.arm_6001.directmotion_insert_instrvec(self.traj2, acc=100, dec=100, pl=5)
        # 延时1.5秒，等待轨迹执行完成
        time.sleep(1.5)

        # 停止直接运动模式但保持伺服电源开启（电机保持使能状态）
        self.arm_6001.directmotion_mode_stop_keep_power_on()
        # 延时1秒，确保停止操作完成
        time.sleep(1)

        # 重新启动直接运动模式
        self.arm_6001.directmotion_mode_start()
        # 插入返回零点位置的轨迹指令，将机械臂移回初始零点位置
        self.arm_6001.directmotion_insert_instrvec(
            [self.q_zero], acc=100, dec=100, pl=5
        )
        # 延时5秒，确保返回零点的操作有足够时间完成
        time.sleep(5)

        # 打印提示信息，表明开始使用封装的队列运动功能
        print(f"使用封装功能")

        # 调用封装的队列运动函数，简化直接运动模式的操作流程
        self.robot_func.robot_queue_move(self.arm_6001, self.traj2)

        # 测试完成后，发送回原点指令，将机械臂恢复到初始位置
        self.arm_6001.go_home()

    def motion_control_test(self):
        """
        执行机械臂运动控制测试（使用7000端口）。
        
        该函数演示了TCB机械臂SDK的高性能运动控制功能，通过7000端口进行批量轨迹控制。
        7000端口专为高实时性运动控制设计，延迟固定为1ms，适合连续路径规划和高频运动。
        
        测试流程：
        1. 初始化机械臂到运行模式
        2. 通过6001端口创建并配置作业文件（7000端口使用的前置条件）
        3. 定义关节空间轨迹数据（traj3）和笛卡尔空间轨迹数据（traj_test）
        4. 使用7000端口执行关节空间运动控制（coord="ACS"）
        5. 使用7000端口执行笛卡尔空间运动控制（coord="MCS"）
        6. 演示封装的运动控制功能
        
        关键概念说明：
        - **ACS (Axis Coordinate System)**: 关节坐标系，使用关节角度值
        - **MCS (Machine Coordinate System)**: 机器坐标系（笛卡尔坐标系），使用[x,y,z,rx,ry,rz]格式
        - **双端口架构**: 6001端口负责管理任务，7000端口负责高频运动控制
        
        注意：使用7000端口前必须先通过6001端口创建作业文件，这是安全机制要求。
        
        Args:
            self: 类实例引用，包含机械臂对象和相关配置
            
        Returns:
            None: 该函数不返回任何值，仅执行机械臂运动控制测试
        """
        # 打印提示信息，表明开始执行机械臂运动控制测试（使用7000端口）
        print(f"机械臂运动控制(7000端口)")

        # 初始化机械臂到运行模式（repeat mode），为7000端口运动控制做准备
        self.arm_init("repeat")

        # 停止当前正在运行的作业文件，确保系统处于干净状态
        self.arm_6001.stop_job_run()

        # 通过6001端口打开名为"tlibot"的作业文件，这是使用7000端口的前置条件
        self.arm_6001.jobfile_open("tlibot")

        # 通知控制器作业文件发送完成，准备接收7000端口的运动指令
        self.arm_6001.jobsend_done("tlibot")

        # 定义关节空间轨迹数据traj3，包含三个目标位置点（关节角度值）
        self.traj3 = [
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 第一个点：第1关节旋转10度
            [-10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 第二个点：第1关节反向旋转10度
            [0.0, 0.0, 90.0, 0.0, 0.0, 0.0, 0.0],   # 第三个点：第3关节旋转90度
        ]

        # 通过7000端口执行关节空间运动控制，coord="ACS"表示使用关节坐标系
        self.arm_7000.motion_control(self.traj3, coord="ACS")

        # 延时6秒，等待关节空间轨迹执行完成（7000端口运动通常较快，但复杂轨迹需要足够时间）
        time.sleep(6)

        # 定义笛卡尔空间单点目标位置，格式为[x,y,z,rx,ry,rz,r7]，注意第7个元素在笛卡尔模式下会被忽略
        q_test = [279.852, 0, 409.237, -3.141, 0, 0, 0]

        # 定义笛卡尔空间轨迹数据traj_test，包含6个连续的目标位置点（6维笛卡尔坐标）
        traj_test = [
            [279.852, 0, 409.237, -3.141, 0, 0],      # 起始点：X=279.852mm, Z=409.237mm, rx=-3.141rad
            [379.852, 0, 409.237, -3.141, 0, 0],      # X方向移动+100mm
            [179.852, 0, 409.237, -3.141, 0, 0],      # X方向移动-100mm（相对于起始点）
            [279.852, 200.0, 409.237, -3.141, 0, 0],  # Y方向移动+200mm
            [279.852, -200.0, 409.237, -3.141, 0, 0], # Y方向移动-200mm（相对于起始点）
            [279.852, 0, 409.237, -3.141, 0, 0],      # 回到起始点，形成闭合轨迹
        ]

        # 通过7000端口执行笛卡尔空间单点运动，将机械臂移动到q_test位置
        self.arm_7000.motion_control([q_test], coord="MCS")
        # 延时6秒，等待单点运动完成
        time.sleep(6)
       
        # 通过7000端口执行完整的笛卡尔空间轨迹运动，实现复杂的连续路径规划
        self.arm_7000.motion_control(traj_test, coord="MCS")
        # 延时1秒，等待轨迹运动完成（由于是连续路径，时间相对较短）
        time.sleep(1)

        # 打印提示信息，表明开始使用封装的运动控制功能
        print(f"使用封装功能")

        # 调用封装的运动控制函数，同时传入6001和7000端口对象，执行关节空间运动（coord=0）
        # traj3 + [self.q_zero] 表示执行完traj3后回到零点位置
        self.robot_func.motion_control_move(
            self.arm_6001, self.arm_7000, self.traj3 + [self.q_zero], coord=0
        )

        # 调用封装的运动控制函数，执行笛卡尔空间运动（coord=1）
        self.robot_func.motion_control_move(
            self.arm_6001, self.arm_7000, traj_test, coord=1
        )

    def servo_point_motion_control_test(self):
        """
        执行机械臂伺服点位运动控制测试（使用7000端口）。
        
        该函数演示了TCB机械臂SDK的伺服点位运动控制功能，这是一种高实时性的运动控制模式。
        伺服点位控制采用分帧流式发送方式，客户端需要逐帧发送轨迹点，控制器实时响应每个点。
        
        测试流程：
        1. 初始化机械臂到运行模式
        2. 启用伺服点位运动控制模式
        3. 获取当前位置作为起始点
        4. 生成100个连续的轨迹点（第1关节逐步增加0.5度）
        5. 逐帧发送轨迹点到7000端口，实现连续的伺服点位控制
        
        伺服点位控制特点：
        - **高实时性**：延迟固定为1ms，适合需要精确时间控制的应用
        - **分帧传输**：通过sum和count参数将完整轨迹分成多帧发送
        - **灵活性高**：可随时调整或中断运动轨迹
        - **客户端控制**：由客户端精确控制每个点的发送时间间隔
        
        注意：伺服点位控制模式与批量运动控制模式互斥，不可同时使用。
        
        Args:
            self: 类实例引用，包含机械臂对象和相关配置
            
        Returns:
            None: 该函数不返回任何值，仅执行伺服点位运动控制测试
        """
        # 打印提示信息，表明开始执行伺服点位运动控制测试（使用7000端口）
        print(f"伺服点位运动控制(7000端口)")
        # 初始化机械臂到运行模式（repeat mode），为伺服点位控制做准备
        self.arm_init("repeat")
        # 注释掉的代码：禁用伺服点位运动控制模式（用于调试或重置状态）
        # self.arm_7000.set_servo_point_motion_control(False)
        # 延时2秒，确保初始化完成并给系统足够时间稳定
        time.sleep(2)
        # 启用伺服点位运动控制模式，准备接收分帧轨迹数据
        self.arm_7000.set_servo_point_motion_control(True)
        # 通过6001端口查询当前关节位置，作为轨迹生成的起始点
        q0 = self.arm_6001.currentpos_inquiry(0)
        # 初始化空的轨迹列表，用于存储生成的100个轨迹点
        traj = []
        # 循环生成100个连续的轨迹点
        for i in range(100):
            # 复制当前位置作为基础，避免修改原始位置数据
            qi = q0.copy()  #
            # 在第1关节上逐步增加角度：每次增加0.5*(i+1)度，形成线性递增轨迹
            qi[0] += 0.5 * (i + 1)
            # 将生成的轨迹点添加到轨迹列表中
            traj.append(qi)

        # 逐帧发送轨迹点到7000端口，实现伺服点位运动控制
        for i in range(100):
            # 发送第i帧轨迹数据，参数说明：
            # robot=1: 机器人编号（通常为1）
            # end=0: 表示这不是最后一帧（end=1表示结束帧）
            # sum=len(traj): 完整轨迹的总点数（100个点）
            # count=i: 当前帧的索引（0-99）
            # PosVec=traj: 完整的轨迹数据（所有100个点）
            self.arm_7000.servo_point_motion_control(
                robot=1, end=0, sum=len(traj), count=i, PosVec=traj
            )
            # 注释掉的代码：较长的延时（1秒），不适合高实时性伺服控制
            # time.sleep(1)
            # 实际使用的延时：0.004秒（4ms），接近7000端口的1ms延迟特性
            # 这个时间间隔需要根据实际应用需求和系统性能进行调整
            time.sleep(0.004)


if __name__ == "__main__":
    test = MotionControlTest()
    # test.go_home_test()
    # test.movj_test()
    # test.movl_test()
    # test.movc_test()
    # test.jog_operation_test()
    # test.directmotion_mode_test()
    # test.motion_control_test()
    test.servo_point_motion_control_test()

    # test.robot_func.robot_stop(test.arm_6001)
    # test.arm_6001.disconnect()
    # test.arm_7000.disconnect()

    # 查询当前机械臂的关节角度位置（使用关节坐标系）
    q = test.arm_6001.currentpos_inquiry(0)
    print(f"当前关节角度位置：{q}")

    # 如果需要，也可以使用预设的关节角度值（当前被注释掉）
    # q = [0., 0., 0., 0., 0., 0.]

    # 使用正运动学函数计算当前关节角度对应的末端执行器位姿矩阵
    T = test.robot_func.kinematics.fkine(q)
    print(f"当前末端执行器位姿矩阵：\n{T}")

    # 将位姿矩阵转换为位置和姿态向量（位置 x, y, z 和欧拉角 a,b,c）
    pos = test.robot_func._matrix_to_pose(T)
    print(f"当前末端执行器位姿向量：{pos}")
    # 将姿态向量中的角度值保留 4 位小数，并存储到新列表中
    pos_ = [round(angle, 4) for angle in pos]
    print(f"当前末端执行器位姿向量：{pos_}")

    # ps1 输出通过正运动学计算得到的姿态值（欧拉角 a,b,c)，即列表中索引 3 及以后的元素
    print(f"通过正运动学计算得到的姿态值（欧拉角 a,b,c):  ",pos_[3:])
    # ps2 输出通过 6001 端口直接查询得到的姿态值（欧拉角 a,b,c），同样保留 4 位小数，这样可以验证运动学计算的准确性。
    print(f"通过 6001 端口直接查询得到的姿态值（欧拉角 a,b,c):  ",test.arm_6001.currentpos_inquiry(1)[3:])
