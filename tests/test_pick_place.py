from pathlib import Path

import mujoco

from lagbot_sim.build_scene import JOINTS, build
from lagbot_sim.env import PickPlaceEnv
from lagbot_sim.expert import rollout


SOURCE = Path('/home/zhw/projects/lagbot_ws/src/lagbotwbc')


def test_generated_scene_uses_right_arm_joints(tmp_path):
    """生成的 MuJoCo 场景应保留原右臂关节顺序和所需视觉网格。"""
    scene = build(SOURCE, tmp_path)
    model = mujoco.MjModel.from_xml_path(str(scene))
    assert tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                 for i in range(7)) == JOINTS
    assert model.nu == 9  # 7 个右臂目标 actuator 加 2 个手指滑动目标 actuator
    assert model.nmesh == 8  # 从 Lagbot 原始描述复制的 8 个右臂网格


def test_scripted_demonstrations_reach_goal(tmp_path):
    """不同初始方块位置下，脚本示范都应完成一轮抓放并生成完整轨迹。"""
    scene = build(SOURCE, tmp_path)
    for seed in range(5):
        env = PickPlaceEnv(scene, seed=seed)
        observations, actions, success = rollout(env)
        assert success, f'随机种子 {seed}，方块位置={env.block}'
        assert observations.shape == (525, 36)
        assert actions.shape == (525, 8)
