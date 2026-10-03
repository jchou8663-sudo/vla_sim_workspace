"""脚本示教器：将桌面抓放路径转换为与 WBC 语义一致的关节目标。"""

from __future__ import annotations

import numpy as np

from .env import PickPlaceEnv


def rollout(env: PickPlaceEnv, *, record: bool = True):
    """执行一轮八阶段抓放，并可记录每步的观测与示范动作。

    观测在发送动作前采样，因此每一行数据表示「当前状态 → 下一步命令」。
    每阶段先以当前关节解为种子求目标 IK，再用三次平滑曲线插值关节角。
    阶段时长总计 525 步；按环境默认 50 Hz 控制频率约为 10.5 秒。
    返回值分别是观测数组、动作数组以及最终任务成功标志。
    """
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    block = env.block.copy()
    goal = env.goal.copy()
    # 每行依次为 TCP 目标、夹爪命令、持续步数。夹爪沿用 WBC 约定：
    # 0 张开、1 闭合。保持抓取点或放置点不动的阶段用于完成夹爪动作。
    waypoints = [
        # 从方块上方接近，并保持夹爪张开。
        (block + [0, 0, 0.14], 0.0, 90),
        (block + [0, 0, 0.005], 0.0, 55),
        # 在方块处闭合、上提，再搬运至目标上方。
        (block + [0, 0, 0.005], 1.0, 45),
        (block + [0, 0, 0.14], 1.0, 65),
        (goal + [0, 0, 0.14], 1.0, 110),
        (goal + [0, 0, 0.025], 1.0, 60),
        # 放下方块，张开夹爪并将 TCP 撤回目标上方。
        (goal + [0, 0, 0.025], 0.0, 40),
        (goal + [0, 0, 0.14], 0.0, 60),
    ]
    seed = env.data.qpos[env.qadr].copy()
    for target, closure, duration in waypoints:
        q_goal = env.solve_ik(target, seed)
        q_start = env.data.qpos[env.qadr].copy()
        for tick in range(duration):
            progress = (tick + 1) / duration
            # smoothstep 在阶段首尾的速度为零，减少相邻目标切换时的突变。
            blend = progress * progress * (3 - 2 * progress)
            action = np.r_[q_start + blend * (q_goal - q_start), closure].astype(np.float32)
            if record:
                observations.append(env.observe())
                actions.append(action)
            env.step(action)
        # 下一阶段继续使用上一阶段的逆解，避免无谓地切到另一条 IK 支路。
        seed = q_goal
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32), env.success()
