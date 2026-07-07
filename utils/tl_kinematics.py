#!/usr/bin/env python3
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from itertools import product
from scipy.interpolate import make_interp_spline

np.set_printoptions(suppress=True, precision=4)


class RobotKinematics:
    """
    RobotKinematics 封装了TCB机械臂的正向和逆向运动学计算。

    安装依赖：
    pinocchio库:
    pip install pin==3.7.0
    (https://stack-of-tasks.github.io/pinocchio/download.html)

    主要功能:
    - fkine(joint_angles, unit='degrees'): 计算末端执行器的正向运动学, 返回 4x4 齐次变换矩阵
    - ikine(T_desire, q_initial): 迭代求解逆运动学, 返回满足目标位姿的关节角度解
    - ikine_retry(T_desire, random_attempts=100, full_enum=True): 多次随机或全枚举初值尝试求逆解
    - jacobian(q): 根据关节角度计算雅可比矩阵
    - straightline(T_desire, q_initial): 提供基于正逆运动学的直线轨迹插值

    构造函数参数:
    - urdf_path (str): 机械臂 URDF 模型文件路径
    - base_frame (str): 机械臂基座参考系名称, 用于定义运动学计算的相对坐标系。
                    通常选择机械臂的固定基座或安装平台对应的frame。
                    例如："right_tl_robot_link0" 或 "base_link"
    - end_frame (str): 机械臂末端执行器参考系名称, 用于计算末端位姿和雅可比矩阵。
                    通常选择机械臂的末端连杆或工具中心点对应的frame。
                    例如："right_tl_robot_link7" 或 "tool0"

    使用示例:
    # 创建机械臂运动学对象
    robot_kines = RobotKinematics(
        urdf_path="/path/to/robot.urdf",
        base_frame="tl_robot_link0",  # 机械臂基座
        end_frame="tl_robot_link7"    # 机械臂末端
    )

    注意:
    1. 坐标系定义采用"基座到末端"的链式结构, 正向运动学计算的是 end_frame 相对于 base_frame 的位姿
    2. 输入输出关节角度默认单位为度, 可通过 unit 参数切换为弧度
    3. 关节链会自动从 URDF 模型中提取, 无需手动指定关节数量
    """

    def __init__(self, urdf_path: str, base_frame: str = None, end_frame: str = None):
        self.interpolation_points = 6  # 直线轨迹插值生成的点数
        self.base_frame = base_frame
        self.end_frame = end_frame

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(self.end_frame)

        try:
            self.base_frame_id = self.model.getFrameId(base_frame)
            self.end_frame_id = self.model.getFrameId(end_frame)

            self.active_joints = self._get_joint_chain()
            self.n_joints = len(self.active_joints)
            self.joint_limits = self._get_joint_limits()

            # print(f"base_frame_id:{self.base_frame_id}; end_frame_id:{self.end_frame_id}")
            # print(f"✓ FK: {base_frame} → {end_frame}")
            # print(f"  Active joints ({self.n_joints}): {[self.model.names[j] for j in self.active_joints]}")
            # print(f"\n  Joint limits (degrees):")
            # for i, joint_id in enumerate(self.active_joints):
            #     joint_name = self.model.names[joint_id]
            #     lower = np.rad2deg(self.joint_limits['lower'][i])
            #     upper = np.rad2deg(self.joint_limits['upper'][i])
            #     print(f"    {i}: {joint_name:20s} [{lower:7.2f}, {upper:7.2f}]")

        except Exception as e:
            # print(f"✗ Error: {e}")
            raise

    def _get_joint_chain(self) -> list:
        end_joint_id = self.model.frames[self.end_frame_id].parentJoint
        base_joint_id = self.model.frames[self.base_frame_id].parentJoint

        joint_chain = []
        current_joint = end_joint_id

        while current_joint >= 0:
            if current_joint <= base_joint_id:
                break

            joint = self.model.joints[current_joint]
            if joint.nq > 0:
                joint_chain.insert(0, current_joint)

            current_joint = self.model.parents[current_joint]

        return joint_chain

    def _get_joint_limits(self) -> dict:
        """从URDF模型中获取活动关节的限位"""
        lower_limits = np.zeros(self.n_joints)
        upper_limits = np.zeros(self.n_joints)

        for i, joint_id in enumerate(self.active_joints):
            q_idx = self.model.joints[joint_id].idx_q

            lower_limits[i] = self.model.lowerPositionLimit[q_idx]
            upper_limits[i] = self.model.upperPositionLimit[q_idx]

        return {"lower": lower_limits, "upper": upper_limits}

    def fkine(self, joint_angles: np.ndarray, unit: str = "degrees") -> np.ndarray:
        """
        计算末端执行器的正向运动学

        该函数根据给定的关节角度计算机器人末端执行器在基坐标系下的位姿变换矩阵。
        支持度和弧度两种角度单位输入，并通过 Pinocchio 库进行高效的运动学计算。

        Args:
            joint_angles (numpy.ndarray): 7 个关节角度值的数组，表示机器人的关节配置
                - 数组长度必须等于机器人的关节数量 (self.n_joints)
                - 每个元素对应一个活动关节的角度值
            unit (str): 角度单位标识符，默认为 'degrees'
                - 'degrees': 输入角度以度为单位，函数内部会自动转换为弧度
                - 'radians': 输入角度以弧度为单位，直接使用无需转换

        Returns:
            numpy.ndarray: 4 x 4 齐次变换矩阵，表示末端执行器相对于基坐标系的位姿
                - 矩阵包含旋转和平移信息
                - 通过基坐标系到世界坐标系的逆变换与末端到世界坐标系的变换相乘得到

        Raises:
            ValueError: 当输入的关节角度数量与机器人关节数不匹配时抛出
            ValueError: 当unit参数不是'degrees'或'radians'时抛出
        """

        # 检查输入的关节角度数量是否与机器人关节数匹配
        if len(joint_angles) != self.n_joints:
            raise ValueError(
                f"Expected {self.n_joints} joint angles for "
                f"{[self.model.names[j] for j in self.active_joints]}, "
                f"but got {len(joint_angles)}"
            )

        # 根据指定的单位将关节角度转换为弧度制（Pinocchio 库要求的输入格式）
        if unit == "degrees":
            joint_angles = np.deg2rad(joint_angles)
        elif unit != "radians":
            raise ValueError("unit must be 'degrees' or 'radians'")

        # 创建完整的关节配置向量，初始化为零
        # q_full 包含了模型中所有关节的配置，而不仅仅是活动关节
        q_full = np.zeros(self.model.nq)  # array([0., 0., 0., 0., 0., 0.])

        # 将输入的活动关节角度映射到完整的关节配置向量中
        # 根据每个活动关节在模型中的索引位置，将对应的角度值填入 q_full
        for i, joint_id in enumerate(self.active_joints):
            q_idx = self.model.joints[joint_id].idx_q
            q_full[q_idx] = joint_angles[i]

        # 执行正向运动学计算并更新所有连杆的位姿
        # forwardKinematics() 计算关节空间到笛卡尔空间的映射
        # updateFramePlacements() 确保所有参考帧的位置信息都是最新的
        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        # 获取基坐标系和末端执行器在世界坐标系中的变换矩阵
        T_world_base = self.data.oMf[self.base_frame_id].homogeneous
        T_world_end = self.data.oMf[self.end_frame_id].homogeneous

        # 计算末端执行器相对于基坐标系的变换矩阵
        # 通过基坐标系变换矩阵的伪逆与末端变换矩阵相乘得到相对位姿
        return self.svd_pinv(T_world_base) @ T_world_end

    def ikine(self, T_desire: np.ndarray, q_initial: np.ndarray) -> np.ndarray:
        """
        迭代求解逆运动学

        Args:
            T_desire(numpy.ndarray): 4x4目标位姿的齐次变换矩阵
            q_initial(numpy.ndarray): 初始关节角, 用于迭代初值, 单位为度

        Returns:
            关节角数组(单位度)或 None(无解)
        """
        if len(q_initial) != self.n_joints:
            raise ValueError(f"q_initial must be a {self.n_joints}-element array")
        if T_desire.shape != (4, 4):
            raise ValueError("T_desire must be a 4x4 matrix")

        Pd = T_desire[:3, 3]
        Rd = T_desire[:3, :3]
        q_active = np.deg2rad(q_initial)

        max_iter = 100
        epsilon = np.array([0.001, 0.001, 0.001])
        max_epsilon = np.array([0.01, 0.01, 0.01])

        if self.n_joints == 7:
            joint_list1 = [0, 2, 4, 6]
            joint_list2 = [1, 3, 5]
        elif self.n_joints == 6:
            joint_list1 = [0, 3, 5]
            joint_list2 = [1, 2, 4]

        for i in range(max_iter):
            T_current = self.fkine(q_active, "radians")

            Pe = T_current[:3, 3]
            Re = T_current[:3, :3]
            R_err = Rd @ Re.T
            delta_theta = self.rotation_matrix_to_vector(R_err)

            if np.linalg.norm(delta_theta) < np.finfo(float).eps:
                delta_r = np.zeros(3)
            else:
                delta_r = (
                    delta_theta
                    / np.linalg.norm(delta_theta)
                    * 2
                    * np.arctan(np.linalg.norm(delta_theta))
                )

            delta_p = Pd - Pe

            R_world_base = self.data.oMf[self.base_frame_id].rotation
            delta_p_world = R_world_base @ delta_p
            delta_r_world = R_world_base @ delta_r
            delta_x = np.hstack((delta_p_world, delta_r_world))

            scale_factor = 0.5 + 0.5 * (i / max_iter)
            delta_x[-3:] *= scale_factor

            J_active = self.jacobian(q_active, "radians")

            J_pinv = self.svd_pinv(J_active)
            delta_q = J_pinv @ delta_x

            for j in joint_list2:
                lower = self.joint_limits["lower"][j]
                upper = self.joint_limits["upper"][j]
                if q_active[j] >= upper and delta_q[j] > 0:
                    step = 0.3 * (abs(upper - lower))
                    delta_q[j] = -step
                elif q_active[j] <= lower and delta_q[j] < 0:
                    step = 0.3 * (abs(upper - lower))
                    delta_q[j] = step

            q_active += delta_q

            for j in joint_list1:
                q_active[j] = self.normalize_angle(q_active[j])

            for j in joint_list2:
                q_active[j] = self.saturate_angle(
                    q_active[j],
                    self.joint_limits["lower"][j],
                    self.joint_limits["upper"][j],
                )

            if np.all(np.abs(delta_p) <= epsilon) and np.all(
                np.abs(delta_r) <= epsilon
            ):
                return np.rad2deg(q_active)

            if i == max_iter - 1:
                if np.any(np.abs(delta_p) > max_epsilon) or np.any(
                    np.abs(delta_r) > max_epsilon
                ):
                    return None
                else:
                    return np.rad2deg(q_active)

    def ikine_retry(
        self, T_desire: np.ndarray, random_attempts=100, full_enum=True
    ) -> np.ndarray:
        """
        多次随机或全枚举初值尝试求逆解

        Args:
            T_desire(numpy.ndarray): 4x4目标位姿的齐次变换矩阵
            random_attempts(int): 随机尝试次数, 默认100
            full_enum(bool): 若随机失败, 是否进行全枚举尝试

        Returns:
            关节角数组(单位度)或 None(无解)
        """
        for _ in range(random_attempts):
            # 在每个关节的限位范围内随机生成初值
            q_init = np.zeros(self.n_joints)
            for j in range(self.n_joints):
                lower = np.rad2deg(self.joint_limits["lower"][j])
                upper = np.rad2deg(self.joint_limits["upper"][j])
                # 在限位范围的中间80%区域内随机
                margin = (upper - lower) * 0.1
                q_init[j] = np.random.uniform(lower + margin, upper - margin)

            q_solve = self.ikine(T_desire, q_init)
            if q_solve is not None:
                return q_solve

        if full_enum:
            # 枚举策略：在关节限位的几个特征点组合
            # 为每个关节生成特征点：下限、中点、上限
            feature_points = []
            for j in range(self.n_joints):
                lower = np.rad2deg(self.joint_limits["lower"][j])
                upper = np.rad2deg(self.joint_limits["upper"][j])
                mid = (lower + upper) / 2
                feature_points.append([lower, mid, upper])

            # 只枚举部分组合以避免计算爆炸（3^7 = 2187种）
            # 可以根据需要调整枚举策略
            all_combos = product(*feature_points)
            for q_init in all_combos:
                q_solve = self.ikine(T_desire, np.array(q_init))
                if q_solve is not None:
                    return q_solve

        return None

    def straightline(self, T_desire, q_initial):
        """
        生成从初始关节角度到目标位姿的直线轨迹，并返回对应的关节空间轨迹

        参数:
            T_desire (numpy.ndarray): 4x4 齐次变换矩阵，表示期望的末端位姿
            q_initial (list[float]): 初始关节角度（角度）

        返回:
            list[list[float]]: 插值后的关节角度轨迹，每个元素是 self.n_joints 个关节角度的列表

        异常:
            ValueError: 当 IK 求解失败，或关节角度超出范围时抛出。
        """
        T_start = self.fkine(q_initial)
        p_start = T_start[:3, 3]
        p_desire = T_desire[:3, 3]

        R_start = T_start[:3, :3]
        R_desire = T_desire[:3, :3]
        quat_start = Rotation.from_matrix(R_start).as_quat()
        quat_desire = Rotation.from_matrix(R_desire).as_quat()

        alpha = np.linspace(0, 1, self.interpolation_points)
        T_list = []

        for a in alpha:
            pos = (1 - a) * p_start + a * p_desire
            quat = self.slerp(quat_start, quat_desire, a)
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            T_list.append(T)

        q_list = []
        q_current = q_initial.copy()
        for T in T_list:
            q = self.ikine(T, q_current)

            if q is None:
                raise ValueError(f"IK failed at T = \n{T}")
            q_list.append([round(val, 4) for val in q])
            q_current = q

        q_array = np.array(q_list)
        for i in range(self.n_joints):
            lower = np.rad2deg(self.joint_limits["lower"][i])
            upper = np.rad2deg(self.joint_limits["upper"][i])
            q_array[:, i] = self.unwrap_angles(q_array[:, i])
            if np.any(q_array[:, i] < lower) or np.any(q_array[:, i] > upper):
                raise ValueError(f"Joint {i+1} out of range: {lower}° to {upper}°")

        return [[round(val, 4) for val in q] for q in q_array.tolist()]

    def spline(self, T_list, q_initial, method="cubic"):
        if method == "quintic":
            if len(T_list) < 6:
                raise ValueError("Quintic requires ≥6 points.")
            spline_args = {
                "k": 5,
                "bc_type": ([(1, 0.0), (2, 0.0)], [(1, 0.0), (2, 0.0)]),
            }
        elif method == "cubic":
            spline_args = {"k": 3, "bc_type": ([(1, 0.0)], [(1, 0.0)])}
        else:
            raise ValueError(f"Unknown method: {method}")

        positions = [T[:3, 3] for T in T_list]
        rotations = [T[:3, :3] for T in T_list]
        t = np.linspace(0, 1, len(T_list))
        cs_x = make_interp_spline(t, [p[0] for p in positions], **spline_args)
        cs_y = make_interp_spline(t, [p[1] for p in positions], **spline_args)
        cs_z = make_interp_spline(t, [p[2] for p in positions], **spline_args)

        quat_start = Rotation.from_matrix(rotations[0]).as_quat()
        quat_end = Rotation.from_matrix(rotations[-1]).as_quat()

        alpha = np.linspace(0, 1, len(T_list) * 10)
        quats_interp = [self.slerp(quat_start, quat_end, a) for a in alpha]

        T_interp = []
        for i, a in enumerate(alpha):
            pos = np.array([cs_x(a), cs_y(a), cs_z(a)])
            R_mat = Rotation.from_quat(quats_interp[i]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = pos
            T_interp.append(T)

        q_list = []
        q_current = q_initial.copy()
        for T in T_interp:
            q = self.ikine(T, q_current)
            if q is None:
                raise ValueError(f"IK failed at T = \n{T}")
            q_list.append([round(val, 4) for val in q])
            q_current = q

        return q_list

    def jacobian(self, q: np.ndarray, unit: str = "degrees") -> np.ndarray:
        """
        求解雅可比矩阵

        Args:
            q(numpy.ndarray): 7个关节角度, 单位为度或弧度
            unit(str): 角度单位, 默认 'degrees'
                - 'degrees': 度
                - 'radians': 弧度

        Returns:
            雅可比矩阵
        """
        if unit == "degrees":
            q = np.deg2rad(q)
        elif unit != "radians":
            raise ValueError("unit must be 'degrees' or 'radians'")

        q_full = np.zeros(self.model.nq)
        for j, joint_id in enumerate(self.active_joints):
            q_idx = self.model.joints[joint_id].idx_q
            q_full[q_idx] = q[j]

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacements(self.model, self.data)

        J_full = pin.computeFrameJacobian(
            self.model,
            self.data,
            q_full,
            self.frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        J_active = np.zeros((6, self.n_joints))
        for j, joint_id in enumerate(self.active_joints):
            v_idx = self.model.joints[joint_id].idx_v
            J_active[:, j] = J_full[:, v_idx]
        return J_active

    @staticmethod
    def rotation_matrix_to_vector(R, eps=1e-4):
        tr = np.trace(R)
        tr = max(min(tr, 3), -1)
        theta = np.arccos((tr - 1) / 2)

        if np.abs(theta) < np.finfo(float).eps:
            return np.zeros(3)
        elif np.abs(theta - np.pi) < eps:
            v = np.zeros(3)
            v[0] = np.sqrt(max(R[0, 0] + 1, 0)) / 2.0
            v[1] = np.sqrt(max(R[1, 1] + 1, 0)) / 2.0
            v[2] = np.sqrt(max(R[2, 2] + 1, 0)) / 2.0
            if R[2, 1] - R[1, 2] < 0:
                v[0] = -v[0]
            if R[0, 2] - R[2, 0] < 0:
                v[1] = -v[1]
            if R[1, 0] - R[0, 1] < 0:
                v[2] = -v[2]
            norm_v = np.linalg.norm(v)
            return theta * v / norm_v if norm_v > np.finfo(float).eps else np.zeros(3)
        else:
            return (theta / (2 * np.sin(theta))) * np.array(
                [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
            )

    @staticmethod
    def normalize_angle(theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def saturate_angle(theta, min_val, max_val):
        return np.clip(theta, min_val, max_val)

    @staticmethod
    def adaptive_lambda(lamda0, segmal_min, segmal_max):
        if segmal_min == 0:
            min_max = 1000
        else:
            min_max = segmal_max / segmal_min
        if segmal_min < min_max:
            return -lamda0 * np.sin(2 * segmal_max**2 / np.pi * segmal_min) + lamda0
        else:
            return 0

    @staticmethod
    def svd_pinv_set(segmal, lamda_val):
        return 0 if segmal < 1e-3 else segmal / (segmal**2 + lamda_val**2)

    def svd_pinv(self, J):
        U, S, Vt = np.linalg.svd(J)
        V = Vt.T
        Ut = U.T
        segmal_min = np.min(S)
        segmal_max = np.max(S)
        lamda0 = 0.01
        lamdas = self.adaptive_lambda(lamda0, segmal_min, segmal_max)

        Sk = np.zeros((V.shape[1], U.shape[0]))
        for i in range(len(S)):
            Sk[i, i] = self.svd_pinv_set(S[i], lamdas)
        return V @ Sk @ Ut

    @staticmethod
    def slerp(q0, q1, t):
        q0 = np.array(q0, dtype=np.float64)
        q1 = np.array(q1, dtype=np.float64)
        dot = np.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * q0 + s1 * q1) / np.linalg.norm(s0 * q0 + s1 * q1)

    @staticmethod
    def unwrap_angles(angles):
        angles = np.array(angles)
        return np.unwrap(angles * np.pi / 180) * 180 / np.pi
