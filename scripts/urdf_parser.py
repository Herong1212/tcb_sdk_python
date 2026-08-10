from urdf_parser_py.urdf import URDF

# 加载并解析 URDF 文件
urdf_file_path = "models/TCB610_06N_urdf/urdf/TCB610_06N_urdf.urdf"
robot = URDF.from_xml_file(urdf_file_path)

# 获取 link 的惯性参数
link1 = robot.link_map["tl_robot_link0"]
if link1.inertial:
    print(f"质量: {link1.inertial.mass}")
    print(f"质心: {link1.inertial.origin.xyz}")

    # 提取转动惯量
    inertia = link1.inertial.inertia
    print(f"ixx: {inertia.ixx}, iyy: {inertia.iyy}, izz: {inertia.izz}")
    print(f"ixy: {inertia.ixy}, ixz: {inertia.ixz}, iyz: {inertia.iyz}")
