# Lagbot 桌面抓放模仿学习：第一版

本项目从 `lagbotwbc` 的右臂 Xacro 和 meshes 生成 MuJoCo 场景，验证「示范采集 → 数据保存 → 行为克隆训练 → 仿真部署评估」的完整流程。任务是将桌上的方块抓起，放到绿色目标区域。**底盘、升降机构和头部固定，只有右臂与仿真夹爪参与动作。**

这是仿真验证工程，不会连接或控制真机。当前策略使用仿真器直接提供的物体位置和初始关节角，不使用相机图像；夹爪闭合后的物体附着也是简化规则，并非经过标定的接触物理模型。因此，本文中的成功率只代表此场景、此初始位置范围内的结果。

## 已实现的流程

1. **生成场景**：展开原始 Xacro，读取右臂的关节树、关节限位和惯量，并复制 8 个右臂视觉网格。腰部固定在 `1.15 m`，再加入桌子、方块、目标区和双指仿真夹爪。
2. **采集示范**：脚本示教器用 MuJoCo Jacobian 求末端逆运动学（IK），依次经过物体上方、抓取点、目标上方和放置点。每个时间步保存观测及对应命令。
3. **保存数据**：同时生成便于本地训练的 `.npz` 文件，以及 LeRobot 0.6 格式的数据集。LeRobot 字段为 `observation.state` 和 `action`，采样频率为 `50 Hz`。
4. **训练与评估**：用小型 PyTorch 网络进行行为克隆（BC），保存模型参数，再让模型独立控制仿真机器人完成整轮抓放。

### 动作约定

每个动作是长度为 8 的向量：

| 位置 | 含义 | 单位或范围 |
| --- | --- | --- |
| `0:7` | `right_joint_1` 至 `right_joint_7` 的目标位置，顺序固定 | rad |
| `7` | 夹爪命令：`0` 张开，`1` 闭合 | `[0, 1]` |

这与 WBC 的 Joint Stream 和末端开度接口保持相同的动作语义。仿真中两个手指使用相反方向的滑动关节；动作值会被换算为各手指的目标位移。当前代码**没有**把策略命令发布到 WBC。

### 观测与策略输入

`observation.state` 共 36 维，依次包含：右臂 7 个关节角、7 个关节速度、夹爪实际张开比例、方块位置 3 维、目标位置 3 维、TCP 位置 3 维、是否已附着 1 维、方块初始位置 3 维、右臂初始关节角 7 维、当前时间步比例 1 维。

第一版策略只取**方块初始 XY 位置、右臂初始关节角和时间步比例**，并为时间加入正弦／余弦特征。这使模型学习示范轨迹，也说明它目前依赖准确的初始状态和固定任务时序；物体被扰动后的自主纠偏不在本版验证范围内。

## 安装与运行

使用 Python 3.12，并安装 `requirements.txt` 中的依赖。默认机器人源码路径是 `/home/zhw/projects/lagbot_ws/src/lagbotwbc`；若放在别处，生成场景时用 `--source` 指定。以下命令均从本项目根目录执行：

```bash
python -m pip install -r requirements.txt
python -m lagbot_sim.build_scene --source /home/zhw/projects/lagbot_ws/src/lagbotwbc
python -m lagbot_sim.learn collect --episodes 60
python -m lagbot_sim.learn train --epochs 180
python -m lagbot_sim.learn eval --episodes 50
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

普通 `eval` 只输出成功率，不启动图形窗口。要在有图形桌面的终端实时观看一轮抓放：

```bash
python -m lagbot_sim.learn eval --episodes 1 --viewer
```

窗口按约 `50 Hz` 的策略频率播放，一轮约 `10.5 s`。如果通过远程终端运行，或无法连接图形桌面，可用 EGL 离屏渲染生成 GIF；该模式每 5 步取一帧，约 `10 fps`：

```bash
MUJOCO_GL=egl python -m lagbot_sim.learn eval --episodes 1 --gif runs/eval.gif
```

GIF 只支持一次评估；多次成功率统计继续使用不带 `--gif` 的 `eval` 命令。图形窗口需要终端能访问桌面显示服务器；`DISPLAY` 有值本身不保证可连接。

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于避免本机 ROS 2 的 pytest 插件自动加载缺失的可选包；它不影响本项目测试内容。如果 Hugging Face 的默认缓存目录不可写，可先把 `HF_HOME` 设为一个可写目录。

生成的 MuJoCo 场景和复制的网格在 `generated/`；示范文件、LeRobot 数据集和训练后的 `runs/policy.pt` 在 `runs/`。两个目录都被 Git 忽略，可按上述命令重新生成。

## 已完成的测试

- MuJoCo 场景加载成功：右臂 7 个关节、两个手指滑动关节、8 个复制的右臂网格。
- 脚本示范采集：**60/60 次成功**。
- LeRobot 数据读取：**60 条轨迹、31,500 帧**；观测形状为 `(36,)`，动作形状为 `(8,)`。
- 训练 180 轮后，用未参与训练的随机种子 `1000–1049` 执行：**50/50 次成功**。
- 自动测试：`2 passed`，覆盖场景生成与多个初始位置的示范抓放。

成功判定要求方块最终位于目标中心周围 `5 cm` 内，且高度误差小于 `4 cm`。这组测试没有验证真实夹爪接触、相机识别、运动延迟或真机安全性。

## 接入真机前需要完成

将仿真直接提供的物体位置替换为经标定的相机观测；验证 TCP、夹爪几何、碰撞和抓取接触；实现 WBC 适配层，检查右臂 Stream 模式并完整发布关节目标与夹爪命令，同时处理限速、数据超时和停止。完成这些环节后，才能评估仿真数据对真机任务的帮助。
