# B2 Blind Lateral Ascent

Unitree B2 的 PPO 盲爬上山策略：**22cm 台阶、20kg 固定负重**，三段共 24 级上行台阶，上山后停稳。

## 保留的权重和效果

`weights/policy.pkl`，约 2.3MB，训练样本数 156,925,952。文件原样保留，包含策略/价值网络参数和观测归一化统计。没有优化器状态或物理池，适合推理、评测和新建微调实验。

本次以两个相同的地形测试种子 `1040031`、`1041031` 对比最终候选，每个种子、每项任务测试 20 个回合：

| 候选权重 | 上山成功 | 站立成功 |
|---|---:|---:|
| **best_training.pkl（本次保留）** | **30 / 40（75%）** | 40 / 40 |
| final.pkl | 28 / 40（70%） | 40 / 40 |

保留文件来自训练第 156,925,952 个样本的 `best_training.pkl`，原样复制为 `weights/policy.pkl`，没有修改网络参数。两个候选只相差 2 个成功回合，这不构成统计上显著优越的证据；本次按测得的成功总数选取一份。

原 `final.pkl` 的历史独立验收曾为上山 17/20（85%），本次未稳定复现。当前发布模型的两组上山结果均为 15/20（75%），没有达到原 80% 验收门槛，因此本仓库**不标记最终目标已通过**。不同 GPU 的浮点差异可能影响回放和验收次数。

逐回合结果见 [evaluation/acceptance.json](evaluation/acceptance.json)，候选对比见 [evaluation/comparison.json](evaluation/comparison.json)，模型信息和 SHA-256 见 [weights/model.json](weights/model.json)。

## 安装和快速检查

原运行环境：Linux、Python 3.12、NVIDIA GPU，JAX CUDA 13。需要支持对应 CUDA 运行时的驱动。回放编码还需要系统安装 `ffmpeg`，默认回放字体为 DejaVu Sans。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py check
```

检查命令在 CPU 上加载随仓库提供的可信权重，预期输出 `observations: 1325`、`actions: 12`、`finite: true`。依赖按原环境固定版本；完整环境版本留档见 `configs/requirements-lock.txt`，其中原机器的本地依赖路径已替换为固定 Git 提交。

## 重新评测与导出视频

```bash
# 上山、站立各 20 次，生成实际物理轨迹、场景和 summary.json。
python run.py evaluate

# 将第一次上山测试按原速渲染为 outputs/ascent.mp4。
MUJOCO_GL=egl python run.py render

# 另一个独立种子，输出到新的目录。
python run.py evaluate --seed 1041031 --output outputs/evaluation_new_seed
```

`outputs/` 中的运行产物不提交到 Git。视频读取新评测的物理轨迹，没有预设动作或隐藏重置。查看每次运行的 `summary.json` 判断成功，生成视频本身不代表验收通过。

## 策略输入和动作

Actor 使用 25 帧、每帧 53 维历史：陀螺仪、重力方向、关节位置/速度、上次动作、四脚接触、速度/转向指令、支撑脚估计的身体速度，以及累计横向位移估计。`configs/b2_payload_training.json` 保留原课程与货架建模参数，其中 30kg 是完整课程和载荷比例的参考值；本发布权重的任务是第 14 档、20kg。Actor 不读取地形扫描、台阶高度、负重质量、仿真世界位置或仿真提供的底盘线速度。

导航指令由仿真任务生成；“盲爬”描述的是低层策略的观测范围，并不表示已实现真机自主导航。速度和横向位移估计会随打滑和传感器偏差漂移。

动作顺序为 `FR / FL / RR / RL`，每腿依次为髋外展、髋俯仰、膝关节（`hip / thigh / calf`）；策略 50Hz，物理 250Hz，每个物理子步重算 PD 力矩。关节目标为默认姿态加 `0.6 × action`，PD 参数和力矩上限保持原实现。

## 新建微调训练

```bash
python -m stairs.train_taishan \
  --run runs/b2_lateral_finetune \
  --task-version b2_blind_lateral_v1 \
  --warm-start weights/policy.pkl --warm-start-mode actor-critic \
  --start-stage 14 --target-stage 14 \
  --num-envs 64 --horizon 32 --epochs 4 --width 192 \
  --exploration-profile refine --actor-lr 0.00005 \
  --steps 20000000 --max-stage-steps 20000000
```

这会继承模型参数和归一化统计，重新初始化优化器、回合和物理池。完整恢复所需的 `latest.pkl` 未打包，因此不能用本仓库精确续接原训练进程。最终策略由多轮本地盲爬实验迁移训练得到，本仓库不声称提供完整历史训练过程的复现材料。

## 目录

```text
weights/                  唯一模型 policy.pkl 与模型说明
stairs/                   PPO、B2 环境、传感器估计、课程和场景生成
scripts/                  评测、回放和场景外观脚本
configs/                  物理/载荷配置和原环境版本留档
vendor/unitree_b2/        官方 B2 MJCF、必需网格和原许可证
tests/                    观测隔离、速度估计、横向记忆等测试
evaluation/acceptance.json 最终独立验收结果
run.py                    加载检查、评测、视频回放入口
```

部分基类和通用 PPO 模块沿用原工程，为保留的策略提供继承和训练依赖。第三方 Playground 由依赖安装，不包含其 Git 历史。B2 网格是显示机器人所必需的资源，不是额外权重。

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -v
```

模型质量边界沿用原实验：刚性固定负重，没有液体晃动、绑带柔性、热衰减或电池模型；官方 MJCF 惯性合计约 83.5kg，保留原惯性，尚未校准真机动力学。此仓库没有完成实体机器人部署验证。

第三方来源与许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。项目自有代码和训练权重尚未指定仓库级开源许可。
