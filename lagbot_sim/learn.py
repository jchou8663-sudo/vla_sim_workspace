"""采集 LeRobot 示范数据，训练小型行为克隆策略，并在 MuJoCo 中评估。

命令行提供 collect、train、eval 三个阶段。LeRobot 负责保存统一格式的数据；
本版使用独立的 PyTorch 多层感知机作为轻量训练基线。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import mujoco
import numpy as np
import torch

from .env import PickPlaceEnv
from .expert import rollout


def policy_features(obs: np.ndarray) -> np.ndarray:
    """从 36 维原始观测提取 18 维策略输入。

    输入包含初始方块 XY、初始七关节角、时间比例，以及四个频率的正弦／余弦
    时间特征。原始观测中的实时方块位置和附着标志不进入第一版策略，因而当前
    模型学习的是固定时序的轨迹，不具备物体受扰动后的闭环纠偏能力。
    """
    source = obs[..., 25:27]
    initial_q = obs[..., 28:35]
    phase = obs[..., 35:36]
    waves = [fn(2 * np.pi * phase * frequency) for frequency in (1, 2, 4, 8)
             for fn in (np.sin, np.cos)]
    return np.concatenate([source, initial_q, phase, *waves], axis=-1).astype(np.float32)


class Policy(torch.nn.Module):
    """将归一化策略输入映射为 7 个关节目标和 1 个夹爪命令的 MLP。"""

    def __init__(self, obs_dim: int = 18, action_dim: int = 8):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def collect(scene: Path, output: Path, episodes: int, lerobot_path: str | None = None):
    """逐轮运行脚本示教器，保存本地 NPZ 与可选的 LeRobot 数据集。

    仅成功示范会写入数据集；任何一轮失败都会抛错，让调用方检查场景或示教器。
    ``lerobot_path`` 用于加载工作区内额外安装的 LeRobot；正常安装时无需设置。
    """
    output.mkdir(parents=True, exist_ok=True)
    dataset = None
    if lerobot_path:
        sys.path.insert(0, lerobot_path)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        # LeRobot 的 feature 形状必须与环境输出及动作契约一致；采样频率
        # 由 MuJoCo 的 0.002 s 物理步长 × 每个策略步 10 个子步得到。
        features = {
            'observation.state': {'dtype': 'float32', 'shape': (36,), 'names': None},
            'action': {'dtype': 'float32', 'shape': (8,), 'names': None},
        }
        dataset = LeRobotDataset.create('local/lagbot_pick_place', fps=50,
                                        features=features, root=output / 'lerobot',
                                        robot_type='lagbot_right_arm', use_videos=False)
    except ImportError as exc:
        print(f'LeRobot 不可用（{exc}）；本次仅写入 NPZ 示范文件')
    successes = 0
    for index in range(episodes):
        env = PickPlaceEnv(scene, seed=index)
        observations, actions, ok = rollout(env)
        if not ok:
            raise RuntimeError(f'第 {index} 轮脚本示范失败')
        successes += int(ok)
        np.savez_compressed(output / f'episode_{index:04d}.npz',
                            observation=observations, action=actions)
        if dataset is not None:
            # 一轮示范对应一个 LeRobot episode；task 是该 episode 的文本任务描述。
            for obs, action in zip(observations, actions, strict=True):
                dataset.add_frame({'observation.state': obs, 'action': action,
                                   'task': '抓起桌上的方块并放到绿色目标区。'})
            dataset.save_episode()
        print(f'示范 {index + 1}/{episodes}：成功，共 {len(actions)} 帧')
    if dataset is not None:
        # finalize 会写入数据和元信息的尾部；缺少这一步会导致数据集无法完整读取。
        dataset.finalize()
    (output / 'summary.json').write_text(json.dumps({'episodes': episodes, 'successes': successes}, indent=2))


def load_episodes(directory: Path):
    """按文件名顺序读取 NPZ 示范，并拼接成训练用的逐帧数组。"""
    files = sorted(directory.glob('episode_*.npz'))
    if not files:
        raise FileNotFoundError(f'目录 {directory} 中没有示范文件')
    obs, act = [], []
    for path in files:
        with np.load(path) as episode:
            obs.append(episode['observation'])
            act.append(episode['action'])
    return np.concatenate(obs), np.concatenate(act)


def train(dataset: Path, output: Path, epochs: int, seed: int = 0):
    """训练逐帧行为克隆模型，并保存参数与输入／输出归一化统计量。

    使用和示范相同的 ``policy_features`` 提取规则。关节角与夹爪值的尺度不同，
    因此观测和动作分别按训练集均值／标准差归一化；推理时必须使用同一统计量。
    """
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    obs, act = load_episodes(dataset)
    obs = policy_features(obs)
    obs_mean, obs_std = obs.mean(axis=0), obs.std(axis=0) + 1e-5
    act_mean, act_std = act.mean(axis=0), act.std(axis=0) + 1e-5
    x = torch.from_numpy((obs - obs_mean) / obs_std)
    y = torch.from_numpy((act - act_mean) / act_std)
    model = Policy(x.shape[1], y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    for epoch in range(epochs):
        # 每轮随机打乱示范帧，并用均方误差拟合脚本示教器发送的动作。
        order = torch.randperm(len(x))
        for batch in order.split(512):
            loss = torch.nn.functional.mse_loss(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(f'训练轮次 {epoch + 1}/{epochs}：末批 MSE={loss.item():.5f}')
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'obs_mean': obs_mean, 'obs_std': obs_std,
                'act_mean': act_mean, 'act_std': act_std}, output)
    print(f'模型已保存至 {output}')


def evaluate(scene: Path, checkpoint: Path, episodes: int, seed_offset: int = 1000,
             show_viewer: bool = False, gif: Path | None = None):
    """在独立随机种子上部署策略，统计整轮抓放的最终成功次数。

    评估不调用脚本示教器。每个 50 Hz 策略步读取观测、做一次网络推理，
    再把动作交给环境执行。夹爪输出在 0.5 处二值化，确保与训练示范的
    「完全张开／完全闭合」命令一致。

    ``show_viewer`` 启动 MuJoCo 桌面窗口，并按真实时间播放。``gif`` 用
    离屏渲染保存单轮动画，每 5 个策略步取一帧，即 10 fps；无桌面显示时
    可在启动 Python 前设置 ``MUJOCO_GL=egl``。
    """
    if gif is not None and episodes != 1:
        raise ValueError('保存 GIF 时请使用 --episodes 1；每个文件只记录一轮评估')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = Policy(len(saved['obs_mean']), len(saved['act_mean']))
    model.load_state_dict(saved['model'])
    model.eval()
    successes = 0
    for index in range(episodes):
        env = PickPlaceEnv(scene, seed=index + seed_offset)
        with ExitStack() as stack:
            view = None
            if show_viewer:
                # viewer 必须在有权限访问图形桌面的终端运行。摄像机朝向桌面，
                # 每次 sync 都把环境中的最新关节和方块状态推送到窗口。
                from mujoco import viewer as mj_viewer

                view = stack.enter_context(mj_viewer.launch_passive(env.model, env.data))
                view.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                view.cam.lookat[:] = [0.45, -0.30, 0.77]
                view.cam.distance = 1.35
                view.cam.azimuth = 115
                view.cam.elevation = -30
                view.sync()
            renderer = None
            frames = []
            if gif is not None:
                # GIF 使用同一个斜俯视角；首帧在策略执行前采集，便于对照
                # 方块的起点与最后一帧的目标位置。
                from PIL import Image

                renderer = stack.enter_context(mujoco.Renderer(env.model, height=384, width=512))
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = [0.45, -0.30, 0.77]
                camera.distance = 1.35
                camera.azimuth = 115
                camera.elevation = -30
                renderer.update_scene(env.data, camera=camera)
                frames.append(Image.fromarray(renderer.render()))
            start = time.perf_counter()
            for tick in range(525):
                obs = policy_features(env.observe())
                x = torch.from_numpy(((obs - saved['obs_mean']) / saved['obs_std']).astype(np.float32))
                with torch.no_grad():
                    action = model(x).numpy() * saved['act_std'] + saved['act_mean']
                action[7] = float(action[7] >= 0.5)
                env.step(action)
                if view is not None:
                    if not view.is_running():
                        print('MuJoCo 窗口已关闭，停止评估')
                        return successes
                    view.sync()
                    # 仿真本身通常快于 50 Hz，等待到下一策略步的墙上时间，
                    # 否则整轮动作会在几秒内播完，不便观察。
                    time.sleep(max(0.0, start + (tick + 1) / 50 - time.perf_counter()))
                if renderer is not None and (tick + 1) % 5 == 0:
                    renderer.update_scene(env.data, camera=camera)
                    frames.append(Image.fromarray(renderer.render()))
            if gif is not None:
                gif.parent.mkdir(parents=True, exist_ok=True)
                frames[0].save(gif, save_all=True, append_images=frames[1:],
                               duration=100, loop=0, optimize=True)
                print(f'仿真动画已保存至 {gif}')
        ok = env.success()
        successes += int(ok)
        print(f'评估 {index + 1}/{episodes}：{"成功" if ok else "失败"}，方块位置={env.block.round(3)}')
    print(f'成功次数：{successes}/{episodes}')
    return successes


def main():
    """解析命令行并运行采集、训练或仿真评估阶段。"""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    collect_parser = sub.add_parser('collect')
    collect_parser.add_argument('--scene', type=Path, default=Path('generated/scene.xml'))
    collect_parser.add_argument('--output', type=Path, default=Path('runs/dataset'))
    collect_parser.add_argument('--episodes', type=int, default=20)
    collect_parser.add_argument('--lerobot-path')
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--dataset', type=Path, default=Path('runs/dataset'))
    train_parser.add_argument('--output', type=Path, default=Path('runs/policy.pt'))
    train_parser.add_argument('--epochs', type=int, default=100)
    eval_parser = sub.add_parser('eval')
    eval_parser.add_argument('--scene', type=Path, default=Path('generated/scene.xml'))
    eval_parser.add_argument('--checkpoint', type=Path, default=Path('runs/policy.pt'))
    eval_parser.add_argument('--episodes', type=int, default=5)
    display = eval_parser.add_mutually_exclusive_group()
    display.add_argument('--viewer', action='store_true', help='打开 MuJoCo 窗口并实时播放')
    display.add_argument('--gif', type=Path, metavar='PATH', help='将单轮评估保存为 GIF 动画')
    args = parser.parse_args()
    if args.command == 'collect':
        collect(args.scene, args.output, args.episodes, args.lerobot_path)
    elif args.command == 'train':
        train(args.dataset, args.output, args.epochs)
    else:
        raise SystemExit(0 if evaluate(args.scene, args.checkpoint, args.episodes,
                                       show_viewer=args.viewer, gif=args.gif) == args.episodes else 1)


if __name__ == '__main__':
    main()
