#!/usr/bin/env python3
import numpy as np
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, "..")) + "/utils"
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
# import tl_kinematics
from utils.tl_kinematics import RobotKinematics


class Test:
    def __init__(self):
        """
        初始化运动学测试类，配置 TCB 机械臂的 URDF 模型和测试参数。

        该函数负责：
        1. 设置工作目录路径
        2. 加载 TCB610 和 TCB710 两种型号机械臂的 URDF 模型
        3. 初始化机器人运动学对象
        4. 定义测试用的目标关节角度和初始关节角度

        注意：代码中先配置 TCB610，然后被 TCB710 的配置覆盖，实际使用的是 TCB710 的配置。
        这种设计可能是为了演示不同型号的配置方式，或者便于快速切换测试型号。

        URDF模型说明：
        - TCB610_06N: 6自由度机械臂
        - TCB710_06N: 7自由度机械臂（实际配置显示7个关节角度）

        Args:
            self: 类实例引用

        Returns:
            None: 该函数不返回任何值，仅初始化类成员变量
        """
        # 获取当前脚本文件所在的目录路径
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        # '/home/robot/arm_sdk/TCB_SDK/24.03/TCB_SDK/example'

        # 获取项目根目录路径（当前目录的上一级目录）
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, ".."))
        # '/home/robot/arm_sdk/TCB_SDK/24.03/TCB_SDK'

        # * ========== TCB610 6自由度机械臂配置 ==========
        # 构建 TCB610 机械臂 URDF 模型文件的完整路径
        self.urdf_path = (
            self.parent_dir + "/models/TCB610_06N_urdf/urdf/TCB610_06N_urdf.urdf"
        )
        # '/home/robot/arm_sdk/TCB_SDK/24.03/TCB_SDK/models/TCB610_06N_urdf/urdf/TCB610_06N_urdf.urdf'

        # ps 定义 URDF 模型中的基座坐标系名称
        base_frame = "tl_robot_link0"

        # ps 定义 URDF 模型中的末端执行器坐标系名称
        end_frame = "tl_robot_link_end"

        # 创建 TCB610 机械臂的运动学对象，用于正/逆运动学计算
        # self.robot = tl_kinematics.RobotKinematics(self.urdf_path, base_frame, end_frame)
        self.robot = RobotKinematics(self.urdf_path, base_frame, end_frame)

        # ps 定义 TCB610 的初始关节角度（零位姿态）
        self.q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ps 定义 TCB610 的测试目标关节角度（6个关节），单位：度
        self.q_target = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])

        # * ========== TCB710 7自由度机械臂配置 ==========
        # 重新构建 TCB710 机械臂 URDF 模型文件的完整路径（覆盖之前的TCB610路径）
        # self.urdf_path = (self.parent_dir + "/models/TCB710_06N_urdf/urdf/TCB710_06N_urdf.urdf")

        # 重新定义基座坐标系名称（与 TCB610 相同）
        # base_frame = "tl_robot_link0"

        # 重新定义末端执行器坐标系名称（与 TCB610 相同）
        # end_frame = "tl_robot_link_end"

        # 重新创建 TCB710 机械臂的运动学对象（覆盖之前的TCB610对象）
        # self.robot = tl_kinematics.RobotKinematics(self.urdf_path, base_frame, end_frame)
        # self.robot = RobotKinematics(self.urdf_path, base_frame, end_frame)

        # 定义 TCB710 的测试目标关节角度（7个关节），单位：度
        # [关节1=41.440°, 关节2=-33.645°, 关节3-7=0°]
        # self.q_target = np.array([41.440, -33.645, 0.0, 0.0, 0.0, 0.0, 0.0])

        # 注释掉的代码：可选的零位目标角度配置（用于不同测试场景）
        # self.q_target = np.array([0., 0., 0., 0., 0., 0., 0.])
        # 定义 TCB710 的初始关节角度（零位姿态，7个关节）
        # self.q_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def fkine_test(self):
        """
        执行正向运动学测试，计算给定目标关节角度下的末端执行器位姿。

        该函数使用预先定义的目标关节角度 (self.q_target) 调用机器人的正向运动学
        方法，计算出末端执行器相对于基座坐标系的齐次变换矩阵，并打印结果。

        Args:
            self: 类实例引用，包含机器人对象和目标关节角度等属性

        Returns:
            T1: numpy.ndarray - 4x4 齐次变换矩阵，表示末端执行器在基座坐标系中的位姿
        """
        T1 = self.robot.fkine(self.q_target)

        print(f"正向运动学解算的齐次变换矩阵: \n{T1}\n")

        return T1

    def ikfine_test(self):
        """
        执行逆向运动学测试，验证从末端位姿到关节角度的解算精度。

        该函数首先使用目标关节角度计算末端执行器的齐次变换矩阵，
        然后基于该变换矩阵进行逆向运动学求解，得到关节角度解，
        最后通过正向运动学验证逆解结果的准确性。

        测试流程：
        1. 使用目标关节角度计算末端位姿（正向运动学）
        2. 基于末端位姿求解关节角度（逆向运动学）
        3. 使用逆解结果再次计算末端位姿进行验证

        Args:
            self: 类实例引用，包含机器人对象、目标关节角度(self.q_target)和初始关节角度(self.q_init)等属性

        Returns:
            q_ik: numpy.ndarray - 逆向运动学解算得到的关节角度数组，形状与self.q_target相同
        """
        # 计算目标关节角度对应的末端执行器齐次变换矩阵
        T_end = self.robot.fkine(self.q_target)
        print(f"逆解前的齐次变换矩阵: \n{T_end}\n")

        # 基于末端位姿和初始关节角度进行逆向运动学求解
        q_ik = self.robot.ikine(T_end, self.q_init)
        print(f"逆向运动学解算的关节角度值:\n{q_ik}\n")

        # 验证逆解结果：使用逆解得到的关节角度重新计算末端位姿
        T1 = self.robot.fkine(q_ik)
        print(f"逆解之后再正解的齐次变换矩阵: \n{T1}\n")
        return q_ik

    def ikine_retry_test(self):
        """
        执行逆向运动学计算，不使用参考角度进行求解。

        该函数首先通过正向运动学计算目标关节角度对应的末端位姿，
        然后使用 ikine_retry() 方法进行逆向运动学求解，得到关节角度解。

        Args:
            self: 包含 robot 属性和 q_target 属性的类实例
                - robot: 机器人对象，具有 fkine() 和 ikine_retry() 方法
                - q_target: 目标关节角度配置

        Returns:
            q_ik_retry: 逆向运动学求解得到的关节角度结果
        """
        T_end = self.robot.fkine(self.q_target)
        q_ik_retry = self.robot.ikine_retry(T_end)

        print(f"逆向运动学(不用参考角度)的解算结果:\n{q_ik_retry}\n")

        return q_ik_retry

    def jacobian_test(self):
        """
        执行雅可比矩阵计算测试，计算给定目标关节角度下的雅可比矩阵。

        该函数使用预先定义的目标关节角度 (self.q_target) 调用机器人的雅可比矩阵
        计算方法，得到关节空间到任务空间的速度映射矩阵，并打印结果。

        雅可比矩阵在机器人运动学中具有重要作用：
        - 描述关节速度与末端执行器线速度和角速度之间的关系
        - 用于速度控制、力控制和奇异性分析
        - 是实现微分运动规划的基础

        Args:
            self: 类实例引用，包含机器人对象和目标关节角度(self.q_target)等属性

        Returns:
            J: numpy.ndarray - 雅可比矩阵，维度为(6, n)，其中 n 为目标关节角度的维度，
               矩阵的每一列表示对应关节对末端执行器线速度和角速度的影响
        """
        J = self.robot.jacobian(self.q_target)

        print(f"计算得出的雅可比矩阵: \n{J}\n")

        return J

    def straightline_test(self):
        """
        执行直线轨迹规划测试，生成从初始关节角度到目标位姿的直线运动轨迹。

        该函数首先使用目标关节角度计算末端执行器的目标位姿，
        然后调用机器人的直线轨迹规划方法，生成从初始关节配置到目标位姿的
        直线运动轨迹，并打印轨迹结果。

        直线轨迹规划在机器人应用中非常重要，用于确保末端执行器沿直线路径移动，
        这在许多工业应用场景（如焊接、喷涂、装配等）中是必需的。

        Args:
            self: 类实例引用，包含机器人对象、目标关节角度(self.q_target)和初始关节角度(self.q_init)等属性

        Returns:
            traj: 直线轨迹规划结果，包含轨迹点序列，具体格式取决于straightline方法的实现
        """
        T_end = self.robot.fkine(self.q_target)

        traj = self.robot.straightline(T_desire=T_end, q_initial=self.q_init)

        print(f"直线轨迹:")

        print(traj)


if __name__ == "__main__":
    test = Test()
    T = test.fkine_test()
    print("####################################################")

    test.ikfine_test()
    print("####################################################")

    test.ikine_retry_test()
    print("####################################################")

    test.jacobian_test()
    print("####################################################")

    test.straightline_test()
    print("####################################################")
