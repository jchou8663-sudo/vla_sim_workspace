"""Lagbot 右臂桌面抓放环境：固定底盘与升降，只控制右臂和夹爪。

本环境提供 50 Hz 的任务步进、状态观测和成功判定。手臂采用理想关节位置跟随；
夹爪靠近方块并闭合后，以附着规则近似抓取。该规则用于验证模仿学习数据闭环，
不等同于真实接触、力控或夹爪动力学。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .build_scene import JOINTS


class PickPlaceEnv:
    """维护单个 MuJoCo 抓放任务实例及其可复位状态。"""

    def __init__(self, scene: str | Path = 'generated/scene.xml', seed: int = 0):
        """加载场景并建立关节、TCP 和方块的索引。

        ``seed`` 只控制方块初始 XY 位置的采样；用于复现训练或评估轨迹。
        ``goal`` 是方块中心的目标世界坐标，桌面高度由场景 XML 定义。
        """
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.joint_ids = [self.model.joint(name).id for name in JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[i] for i in self.joint_ids])
        self.vadr = np.array([self.model.jnt_dofadr[i] for i in self.joint_ids])
        self.tcp_id = self.model.site('tcp').id
        self.block_id = self.model.body('block').id
        self.block_qadr = self.model.joint('block_free').qposadr[0]
        self.finger_qadr = np.array([self.model.joint(f'finger_{s}_slide').qposadr[0]
                                     for s in ('left', 'right')])
        self.goal = np.array([0.64, -0.50, 0.805], dtype=np.float64)
        self.source = np.zeros(3)
        self.steps = 0
        self.attached = False
        self.attach_offset = np.zeros(3)
        self.reset()

    def reset(self, randomize: bool = True) -> np.ndarray:
        """复位仿真，采样方块位置，并将右臂摆到方块上方。

        先以一组合法关节角作为 IK 种子，再求方块上方 14 cm 的 TCP 位姿。
        求得的初始关节角会保存在观测中，因为相邻方块位置可能对应不同 IK 支路。
        ``randomize=False`` 用于固定方块位置，便于复现单次问题。
        """
        mujoco.mj_resetData(self.model, self.data)
        home = np.array([-1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.qadr] = home
        self.data.ctrl[:7] = home
        self.data.qpos[self.finger_qadr] = 0.03
        self.data.ctrl[7:] = 0.03
        # 方块只在桌面上的小矩形区域变化：X ±4.5 cm、Y ±4 cm。
        xy = self.rng.uniform([-0.045, -0.04], [0.045, 0.04]) if randomize else [0, 0]
        self.source = np.array([0.48 + xy[0], -0.33 + xy[1], 0.805])
        self.data.qpos[self.block_qadr:self.block_qadr + 3] = self.source
        self.data.qpos[self.block_qadr + 3:self.block_qadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        home = self.solve_ik(self.source + [0, 0, 0.14])
        self.initial_q = home.copy()
        self.data.qpos[self.qadr] = home
        self.data.ctrl[:7] = home
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.attached = False
        self.attach_offset[:] = 0
        return self.observe()

    def observe(self) -> np.ndarray:
        """返回 36 维状态向量，供示范记录和策略推理使用。

        字段顺序固定：关节角 0:7、关节速度 7:14、手指实际张开比例 14、
        方块位置 15:18、目标位置 18:21、TCP 位置 21:24、附着标志 24、
        初始方块位置 25:28、初始关节角 28:35、时间比例 35。
        当前含有仿真真值；接入相机时须重新定义可观测输入。
        """
        block = self.data.xpos[self.block_id]
        tcp = self.data.site_xpos[self.tcp_id]
        return np.concatenate([
            self.data.qpos[self.qadr],
            self.data.qvel[self.vadr],
            [float(np.mean(self.data.qpos[self.finger_qadr])) / 0.03],
            block, self.goal, tcp,
            [float(self.attached)],
            self.source,
            self.initial_q,
            [self.steps / 525.0],
        ]).astype(np.float32)

    @property
    def tcp(self) -> np.ndarray:
        """返回当前 TCP 的世界坐标副本，避免调用方修改 MuJoCo 内部数组。"""
        return self.data.site_xpos[self.tcp_id].copy()

    @property
    def block(self) -> np.ndarray:
        """返回方块中心的世界坐标副本。"""
        return self.data.xpos[self.block_id].copy()

    def step(self, action: np.ndarray, substeps: int = 10) -> tuple[np.ndarray, bool]:
        """执行一个 50 Hz 策略步，并返回新观测与是否成功。

        动作前 7 维是右臂关节目标（rad），第 8 维沿用 WBC 夹爪约定：
        0 张开、1 闭合。MuJoCo 物理步长为 0.002 s；默认执行 10 个子步。
        越过 URDF 关节限位的目标会先截断，非有限数值会直接拒绝。
        """
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,) or not np.all(np.isfinite(action)):
            raise ValueError('动作必须包含 8 个有限数值：7 个关节目标和 1 个夹爪命令')
        limits = self.model.jnt_range[self.joint_ids]
        target_q = np.clip(action[:7], limits[:, 0], limits[:, 1])
        closure = float(np.clip(action[7], 0, 1))
        opening = 1.0 - closure
        # WBC 的 1 表示闭合，而此处手指滑动关节的 0 m 表示闭合。
        # 因此先反转归一化命令，再映射到每侧最大 3 cm 的滑动行程。
        self.data.ctrl[:7] = target_q
        self.data.ctrl[7:] = opening * 0.03
        self.data.qpos[self.qadr] = target_q
        self.data.qvel[self.vadr] = 0
        self.data.qpos[self.finger_qadr] = opening * 0.03
        mujoco.mj_forward(self.model, self.data)
        # 只有 TCP 足够接近方块时闭合，才将方块附着到 TCP。记录当前相对位置
        # 可以避免附着瞬间方块跳变；再次张开即解除，方块随后受重力落到桌上。
        if closure < 0.35:
            self.attached = False
        elif closure > 0.65 and not self.attached and np.linalg.norm(self.block - self.tcp) < 0.055:
            self.attached = True
            self.attach_offset = self.block - self.tcp
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
            # 每个物理子步后恢复期望关节角，使机械臂近似理想位置控制器。
            # 这避免未经标定的臂质量／伺服参数使实验偏离学习流程验证目标。
            self.data.qpos[self.qadr] = target_q
            self.data.qvel[self.vadr] = 0
            self.data.qpos[self.finger_qadr] = opening * 0.03
            if self.attached:
                # 保持抓取时的方块相对 TCP 位姿，并清零自由关节速度。
                # 此操作是简化抓取约束，不是由指尖接触力计算出来的结果。
                self.data.qpos[self.block_qadr:self.block_qadr + 3] = self.tcp + self.attach_offset
                block_dof = self.model.joint('block_free').dofadr[0]
                self.data.qvel[block_dof:block_dof + 6] = 0
            mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        return self.observe(), self.success()

    def success(self) -> bool:
        """方块中心到目标的水平距离小于 5 cm，且高度误差小于 4 cm。"""
        return bool(np.linalg.norm(self.block[:2] - self.goal[:2]) < 0.05
                    and abs(self.block[2] - self.goal[2]) < 0.04)

    def solve_ik(self, target_pos: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray:
        """用 MuJoCo 的 TCP Jacobian 求带阻尼的 6D 逆运动学。

        TCP 姿态固定为局部 Z 轴朝下；位置目标在世界坐标系下给出，单位为米。
        每次迭代联立位置误差与姿态旋转向量，限制单步关节增量，并在原始关节
        限位内截断。``seed`` 可保持相邻示范点沿同一 IK 支路连续运动。
        """
        work = mujoco.MjData(self.model)
        work.qpos[:] = self.data.qpos
        q = self.data.qpos[self.qadr].copy() if seed is None else np.asarray(seed).copy()
        desired = Rotation.from_euler('x', np.pi)
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))
        limits = self.model.jnt_range[self.joint_ids]
        for _ in range(180):
            work.qpos[self.qadr] = q
            mujoco.mj_forward(self.model, work)
            pos_error = target_pos - work.site_xpos[self.tcp_id]
            current = Rotation.from_matrix(work.site_xmat[self.tcp_id].reshape(3, 3))
            rot_error = (desired * current.inv()).as_rotvec()
            if np.linalg.norm(pos_error) < 0.012 and np.linalg.norm(rot_error) < 0.15:
                return q
            mujoco.mj_jacSite(self.model, work, jac_pos, jac_rot, self.tcp_id)
            J = np.vstack([jac_pos[:, self.vadr], jac_rot[:, self.vadr]])
            # 姿态误差权重降低，优先保证指尖位于方块或放置点附近；
            # 阻尼项抑制奇异位姿附近的关节增量。
            error = np.r_[pos_error, rot_error * 0.35]
            delta = J.T @ np.linalg.solve(J @ J.T + 0.02 * np.eye(6), error)
            q = np.clip(q + np.clip(delta, -0.10, 0.10), limits[:, 0] + 0.01, limits[:, 1] - 0.01)
        raise RuntimeError(f'目标 {target_pos} 的 IK 未收敛，位置残差={np.linalg.norm(pos_error):.3f} m')
