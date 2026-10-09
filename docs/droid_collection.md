# 本机 / 远程 OpenPI RGB-D 数据采集

采集入口 `policies/pi0_family/collect.py` 使用 DROID Franka + Robotiq 机器人。
默认采集单个 `BananaInBowlTask`、一路 **320×180 第三人称 RGB-D**；另一个
320×180 腕部 RGB 相机仅作为 OpenPI 输入。策略输入由现有客户端 resize/pad
到 224×224。只运行一个仿真环境，没有额外 viewport 相机和视频编码。

第三人称相机在采集脚本中设置 `focal_length=2.21`，成像面尺寸为
`horizontal_aperture=5.376`、`vertical_aperture=3.024`。在 320×180 下，
预期内参为 `fx=fy≈131.55`、`cx=160`、`cy=90`，接近本机两条 DROID
轨迹中四路第三人称相机的像素焦距（约 131.02～132.93）；主点仍在图像中心，
并非精确复现某台真实相机的完整标定。更改分辨率时，像素内参随之变化，
以每条轨迹实际保存的 `intrinsic/third_person.npy` 为准。

## 启动

OpenPI 与 RoboLab 使用各自的虚拟环境。当前机器已经安装好环境和缓存权重。
终端一运行：

```bash
cd /home/shengyu/Documents/PENG/github/openpi
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
.venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=/home/shengyu/.cache/openpi/openpi-assets-simeval/pi05_droid_jointpos
```

等到服务监听 8000 端口，再在终端二运行：

```bash
cd /home/shengyu/Documents/PENG/github/RoboLab
OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python policies/pi0_family/collect.py \
  --headless --task BananaInBowlTask --num-episodes 1 \
  --rendering_mode quality \
  --width 320 --height 180 --output-dir datasets
```

用 `--max-steps 20` 可做短测试；达到这个人为上限的轨迹会标记为
`status=truncated, complete=false`。完整采集时去掉这个参数。
`--num-episodes` 顺序采集多个 episode，每条的 seed 从 `--seed`（默认 1）递增。
目录使用 UTC 时间、随机标识和序号，不覆盖以前的数据。

用 `watch -n 1 nvidia-smi` 观察联合运行和首次推理编译的显存。
50% 只是 JAX 内存池预分配比例，不是整个进程的严格显存上限。若模型显存不足，
重启 OpenPI，将两个 XLA 设置替换为：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

这会减少缓存内存，但可能显著变慢。如果两个程序依然不能共存，将 OpenPI
部署在内网机器，采集命令增加 `--remote-host <IP> --remote-port 8000` 即可。
必须使用配套的 `pi05_droid_jointpos` 配置与 checkpoint，不能将其他动作语义的
OpenPI 权重直接接入。客户端连接失败会退出；推理超时默认 300 秒，包含首次 JIT，
可用 `--inference-timeout` 调整。

## 每条轨迹

```text
<episode>/
  metadata.json
  env_cfg.json
  timestamps.npy                       # [T]，仿真秒，从 0 开始，间隔 1/15
  inference_seconds.npy                # [T]，每步客户端调用耗时；缓存动作也记录
  images/third_person/000000.png        # uint8 RGB，320×180
  depths/third_person/000000.png        # uint16，毫米；0 表示无效
  depths/metadata.json
  intrinsic/third_person.npy           # [3,3]，保存图像对应的像素内参
  extrinsic/third_person.npy           # [4,4]，base -> camera
  observations/joint_position.npy      # [T,7]，panda_joint1...7，弧度
  observations/gripper_position.npy    # [T,1]，连续闭合度，0=开、1=闭
  observations/cartesian_position.npy  # [T,6]，基座坐标 TCP xyz/rpy
  action/joint_position.npy            # [T,7]，绝对关节目标，弧度
  action/gripper_position.npy          # [T,1]，执行的二值闭合命令
  TCP/third_person/state.npy           # [T,7]，相机坐标 xyz/rpy/gripper_open
  TCP/third_person/metadata.json
  terminal_state.npz                   # 最后一次已提交动作执行后的仿真状态
  trajectory.hdf5                     # RoboLab 原生场景/机器人记录，用于审计
```

这是 DROID **类似格式**，当前 4RC 读取器固定要求两路 320×180 相机，因此不能
直接用该读取器读取这里的单相机数据。

## 时间与坐标

控制频率 15 Hz，物理频率 120 Hz，每个动作推进 8 个物理步。每个索引包含同一
时刻的 RGB、深度和实际状态，以及从该时刻开始执行的动作。每个执行步都采集，
不是每次策略重规划才采集。等待推理期间不推进物理时间。
只有 `env.step` 成功且未发生自动 reset 才提交该帧。最终状态不附加虚构动作。
原生 HDF5 的 `states` 是 step 后状态，不应直接当作这些 PNG 的同索引标签。

基座为 `panda_link0`；相机使用 OpenCV/ROS 光学坐标（右、下、前）。
使用 Isaac Lab `quat_w_ros` 时四元数顺序是 **wxyz**。

```text
T_camera_base = inverse(T_world_camera) @ T_world_base
p_camera = T_camera_base @ p_base
```

`extrinsic/third_person.npy` 保存 `T_camera_base`；`metadata.json` 的
`camera_to_base` 保存逆变换。内参直接来自实际渲染相机，不能沿用 1280×720 的内参。
当前格式要求相机相对基座静止，运行中检测到标定变化会中断，避免错误标签。

深度是光轴方向的 Z-depth（`depth` / `distance_to_image_plane`），不是到光心的
欧氏距离。有限正深度四舍五入为毫米；超出 65.535 米、非有限、非正值写为 0。
有效且可表示深度的编码误差不超过 0.5 毫米。

TCP 对齐仓库根目录 `generate_tcp.py` 处理 DROID 的定义：参考 URDF 在
`finger_joint=0.8 rad` 时的两指接触面中心，偏移约为
`(0,0,0.1442549775197502)` 米，姿态与该 URDF 的 `panda_link8` 一致
（该 URDF 的 Robotiq 安装关节为单位变换）。这是固定虚拟工具点，不随夹爪张合移动，
也不保证等于仿真夹爪在任意张合状态下的实际接触面中心。

USD 的 Robotiq `base_link` 与上述 URDF 坐标系既有平移差，也有旋转差。
代码使用 USD 的 `/panda/panda_link8/panda_hand_joint` 两侧局部变换构造
`T_usd_base_link_tcp`，从实际仿真 body pose 计算 TCP，不使用关节目标代替实际状态。
完整精度变换保存在根目录和 `TCP/third_person` 的元数据中；其近似值为：

```text
T_usd_base_link_tcp ≈ [[0,  0, 1, 0.126080955],
                     [0, -1, 0, 0],
                     [1,  0, 0, 0],
                     [0,  0, 0, 1]]
```

仅沿 USD +X 偏移 0.1442549775 米会产生约 18.174 毫米的位置误差；沿用 USD
`base_link` 姿态会产生约 180° 的姿态误差。`EEF_OFFSET_ROT` 也不能替代这里的
完整旋转修正。新数据标记 `format=robolab_droid_like_v2` 和
`tcp_definition_id=droid_generate_tcp_v2`。`tcp_offset_in_robotiq_base_m` 指参考
**DROID URDF** 坐标系，`tcp_offset_in_usd_base_link_m` 指 **USD** 坐标系。
旧数据不会自动改写；新版校验器会明确拒绝旧 TCP 定义，避免混用。

位置单位米，角度单位弧度，RPY 使用 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。
夹爪闭合度来自实际 `finger_joint/(pi/4)` 并裁剪到 [0,1]；TCP 标签的
`gripper_open = 1 - closedness`。动作是执行命令，不是下一帧实际状态或笛卡尔目标。
参考脚本中的 0.8 rad 仅用于确定固定工具点，并不是这里的闭合度归一化分母。
仿真与真实夹爪的中间闭合度仍可能存在机械标定差异。

`complete=true` 代表正常任务结束，包括成功和超时失败；异常、中断及人为截断均
标记不完整。训练前检查该字段。进程被强杀时目录仍标记不完整，不能作为完整样本使用。

## CPU 验证

不启动 Isaac Sim 的几何、PNG 编码与数据对齐测试：

```bash
.venv/bin/python -m unittest discover -s tests -p test_droid_dataset.py -v
```

对实际采集结果进行逐帧文件、坐标变换和原生 HDF5 动作/状态时序的交叉校验：

```bash
.venv/bin/python policies/pi0_family/validate_dataset.py \
  output/droid_like/<episode> --preview /tmp/collection_preview.png
```

预览第一行绿色十字为 TCP 投影；第二行是米单位深度。环境全景背景没有对应的
实体几何，因此该区域深度为无效值，这是预期行为。校验器也能检查人为截断轨迹，
但检查通过不改变它的 `complete=false` 状态。
