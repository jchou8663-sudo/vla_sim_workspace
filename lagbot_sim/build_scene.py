"""从 Lagbot 的 Xacro 生成可独立加载的 MuJoCo 桌面抓放场景。

右臂的关节树、限位、惯量和视觉网格来自原始机器人描述；桌面、方块和简化
双指夹爪由本文件补充。底盘、升降机构及头部在此任务中保持固定。
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import xacro


DEFAULT_SOURCE = Path('/home/zhw/projects/lagbot_ws/src/lagbotwbc')
JOINTS = tuple(f'right_joint_{i}' for i in range(1, 8))


def _vec(text: str | None, default='0 0 0') -> list[float]:
    """将 URDF 中以空格分隔的三维坐标或 RPY 字符串转为浮点数。"""
    return [float(x) for x in (text or default).split()]


def _quat_from_rpy(rpy: list[float]) -> str:
    """把 URDF 的 roll-pitch-yaw（弧度）转换为 MuJoCo 使用的 w-x-y-z 四元数。"""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return ' '.join(str(x) for x in (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def _s(values) -> str:
    """将数值序列编码为 XML 属性需要的空格分隔字符串。"""
    return ' '.join(str(x) for x in values)


def build(source: Path, output: Path) -> Path:
    """展开机器人 Xacro，生成场景 XML 并复制所需网格，返回 XML 路径。

    ``source`` 是 lagbotwbc 源码包目录，``output`` 是可写输出目录。生成后的
    ``scene.xml`` 只引用输出目录下的网格；运行仿真时不再依赖 ROS 包索引。
    """
    robot_file = source / 'urdf/robots/rokae_omni4.urdf.xacro'
    if not robot_file.is_file():
        raise FileNotFoundError(robot_file)
    output.mkdir(parents=True, exist_ok=True)
    # Xacro 文件包含 ROS 的 $(find lagbotwbc) 查找语法。复制到临时目录并改写这类
    # 引用，使场景生成只依赖用户指定的源码，不依赖该包是否已经 colcon install。
    # 标定 YAML 按原文件的相对路径读取，因此也要保留对应目录结构。
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        copied_urdf = Path(temporary) / 'urdf'
        shutil.copytree(source / 'urdf', copied_urdf)
        shutil.copytree(source / 'config/descriptions', Path(temporary) / 'config/descriptions')
        for file in copied_urdf.rglob('*.xacro'):
            file.write_text(file.read_text().replace('$(find lagbotwbc)', str(source)))
        root = ET.fromstring(xacro.process_file(
            str(copied_urdf / 'robots/rokae_omni4.urdf.xacro'),
            mappings={'left_tool': 'none', 'right_tool': 'non_parallel_claw',
                      'head_joint_type': 'fixed'}).toxml())
    links = {link.attrib['name']: link for link in root.findall('link')}
    joints = {joint.attrib['name']: joint for joint in root.findall('joint')}
    assets = output / 'assets'
    assets.mkdir(exist_ok=True)

    # 以原 URDF 的右臂结构为骨架，另外创建固定工位和可抓取方块。
    # 原有视觉网格不参与碰撞；接触只由桌面、方块和简化手指承担。
    mj = ET.Element('mujoco', model='lagbot_table_pick_place')
    ET.SubElement(mj, 'compiler', angle='radian', autolimits='true', meshdir='assets')
    ET.SubElement(mj, 'option', timestep='0.002', gravity='0 0 -9.81', iterations='80')
    ET.SubElement(mj, 'visual')
    default = ET.SubElement(mj, 'default')
    ET.SubElement(default, 'joint', damping='2', armature='0.02')
    ET.SubElement(default, 'geom', friction='1 0.005 0.0001', condim='3')
    asset = ET.SubElement(mj, 'asset')
    ET.SubElement(asset, 'material', name='arm_mat', rgba='0.75 0.77 0.8 1')
    ET.SubElement(asset, 'material', name='table_mat', rgba='0.44 0.29 0.18 1')
    ET.SubElement(asset, 'material', name='block_mat', rgba='0.9 0.2 0.12 1')
    world = ET.SubElement(mj, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 3', dir='0 0 -1')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='3 3 0.1', rgba='0.7 0.7 0.7 1')
    ET.SubElement(world, 'geom', name='table', type='box', pos='0.58 -0.34 0.39',
                  size='0.42 0.38 0.39', material='table_mat')
    ET.SubElement(world, 'geom', name='goal_marker', type='cylinder', pos='0.64 -0.50 0.782',
                  size='0.05 0.002', rgba='0.1 0.8 0.2 0.45', contype='0', conaffinity='0')
    base = ET.SubElement(world, 'body', name='base_link', pos='0 0 0')
    ET.SubElement(base, 'geom', name='chassis', type='box', pos='0 0 0.20',
                  size='0.29 0.29 0.10', rgba='0.3 0.3 0.35 1', contype='0', conaffinity='0')
    # 原 URDF 的 waist_lift_joint 是移动关节。此版把它冻结在整机配置的
    # 1.15 m 启动高度，并累加 waist_mount_offset_joint 的固定偏移。
    waist_joint = joints['waist_lift_joint']
    waist_offset = _vec(waist_joint.find('origin').attrib['xyz'])
    mount_offset = _vec(joints['waist_mount_offset_joint'].find('origin').attrib['xyz'])
    waist_pos = [waist_offset[i] + mount_offset[i] for i in range(3)]
    waist_pos[2] += 1.15
    waist = ET.SubElement(base, 'body', name='waist_link', pos=_s(waist_pos))
    ET.SubElement(waist, 'geom', type='box', pos='0 0 0.03', size='0.08 0.11 0.25',
                  rgba='0.35 0.35 0.4 1', contype='0', conaffinity='0')

    def add_link(parent_xml: ET.Element, joint_name: str) -> ET.Element:
        """把指定 URDF 关节及其子 Link 转成 MuJoCo Body。

        关节原点成为子 Body 相对父 Body 的位姿；转动轴和上下限直接沿用 URDF。
        当前右臂链只含固定关节与转动关节，因此无需处理底盘轮系或升降驱动。
        """
        joint = joints[joint_name]
        link_name = joint.find('child').attrib['link']
        origin = joint.find('origin')
        body = ET.SubElement(parent_xml, 'body', name=link_name,
                             pos=_s(_vec(origin.attrib.get('xyz'))),
                             quat=_quat_from_rpy(_vec(origin.attrib.get('rpy'))))
        if joint.attrib['type'] == 'revolute':
            limit = joint.find('limit')
            ET.SubElement(body, 'joint', name=joint_name, type='hinge',
                          axis=joint.find('axis').attrib['xyz'],
                          range=f"{limit.attrib['lower']} {limit.attrib['upper']}")
        link = links[link_name]
        inertial = link.find('inertial')
        if inertial is not None:
            mass = float(inertial.find('mass').attrib['value'])
            if mass > 0:
                # 保留质量和惯量对角项。URDF 中少量非对角惯量在此简化模型中省略；
                # 手臂随后按目标关节角运动，不用这些参数验证真机动力学。
                inertia = inertial.find('inertia').attrib
                ET.SubElement(body, 'inertial',
                              pos=inertial.find('origin').attrib.get('xyz', '0 0 0'),
                              mass=str(mass),
                              diaginertia=_s([inertia['ixx'], inertia['iyy'], inertia['izz']]))
        visual = link.find('visual')
        if visual is not None and visual.find('geometry/mesh') is not None:
            # 将每个右臂视觉网格复制到输出目录，避免生成的 XML 仍引用源码树。
            # contype/conaffinity=0 表示网格只用于显示，不会与方块发生接触。
            mesh_tag = visual.find('geometry/mesh')
            source_mesh = source / mesh_tag.attrib['filename'].removeprefix('package://lagbotwbc/')
            copied_mesh = assets / source_mesh.name
            shutil.copy2(source_mesh, copied_mesh)
            mesh_name = f'{link_name}_mesh'
            ET.SubElement(asset, 'mesh', name=mesh_name, file=copied_mesh.name,
                          scale=mesh_tag.attrib.get('scale', '1 1 1'))
            visual_origin = visual.find('origin')
            ET.SubElement(body, 'geom', type='mesh', mesh=mesh_name, material='arm_mat',
                          pos=visual_origin.attrib.get('xyz', '0 0 0'),
                          quat=_quat_from_rpy(_vec(visual_origin.attrib.get('rpy'))),
                          contype='0', conaffinity='0', group='1')
        return body

    arm = add_link(waist, 'right_base_fixed')
    for joint_name in JOINTS:
        arm = add_link(arm, joint_name)
    flange = add_link(arm, 'right_flan_joint')
    # 原工具 Xacro 是不可开合的整体网格。这里用两个相向滑动的手指表示夹爪，
    # 并在法兰前方设置 TCP；手指尺寸是任务级近似值，不是实测夹爪几何。
    ET.SubElement(flange, 'geom', name='gripper_palm', type='box', pos='0 0 0.045',
                  size='0.026 0.048 0.025', rgba='0.2 0.23 0.28 1',
                  contype='0', conaffinity='0')
    for side, sign in [('left', -1), ('right', 1)]:
        finger = ET.SubElement(flange, 'body', name=f'finger_{side}', pos=f'0 {sign * 0.025} 0.075')
        ET.SubElement(finger, 'joint', name=f'finger_{side}_slide', type='slide',
                      axis=f'0 {sign} 0', range='0 0.03', damping='1')
        ET.SubElement(finger, 'geom', name=f'finger_{side}_pad', type='box',
                      pos='0 0 0.045', size='0.014 0.008 0.047', mass='0.08',
                      rgba='0.1 0.13 0.16 1', friction='2 0.02 0.002')
    ET.SubElement(flange, 'site', name='tcp', pos='0 0 0.145', size='0.008', rgba='0 1 0 1')
    obj = ET.SubElement(world, 'body', name='block', pos='0.52 -0.35 0.83')
    ET.SubElement(obj, 'freejoint', name='block_free')
    ET.SubElement(obj, 'geom', name='block_geom', type='box', size='0.025 0.025 0.025',
                  mass='0.08', material='block_mat', friction='1.5 0.005 0.001')
    ET.SubElement(obj, 'site', name='block_site', size='0.004')
    ET.SubElement(world, 'camera', name='overhead', pos='0.52 -0.35 2.0',
                  xyaxes='1 0 0 0 1 0', fovy='55')
    # 位置 actuator 保留了清晰的命令通道；第一版环境会在每个子步直接写入关节
    # 目标，实现理想位置跟随，因此这里的低增益不代表真机或物理伺服参数。
    actuators = ET.SubElement(mj, 'actuator')
    for joint_name in JOINTS:
        ET.SubElement(actuators, 'position', name=f'{joint_name}_target', joint=joint_name,
                      kp='1', kv='1', ctrlrange=joints[joint_name].find('limit').attrib['lower'] + ' ' +
                      joints[joint_name].find('limit').attrib['upper'])
    for side in ('left', 'right'):
        ET.SubElement(actuators, 'position', name=f'finger_{side}_target',
                      joint=f'finger_{side}_slide', kp='1', kv='1', ctrlrange='0 0.03')
    ET.indent(mj)
    path = output / 'scene.xml'
    ET.ElementTree(mj).write(path, encoding='unicode', xml_declaration=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=Path('generated'))
    args = parser.parse_args()
    print(build(args.source, args.output))
