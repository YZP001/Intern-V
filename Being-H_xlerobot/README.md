# Being-H xlerobot

目标：用 `datasets/local_xlerobot_box2basket_left_head` (LeRobot v3.0) 对 `Being-H05-2B_libero_robocasa` 做 LoRA 微调；推理在服务器上跑，笔记本只负责连接 XLerobot (COM4/COM3 + 摄像头 0/1) 并调用推理服务。


## 1) 服务器：LoRA 训练 (4x4090)

### 1.1 准备数据集路径

把笔记本的 `datasets/local_xlerobot_box2basket_left_head` 拷贝到服务器同路径或任意路径，然后设置环境变量：

- Linux bash:
  - `export VLA_DATASETS_DIR=/workspace/datasets`

确保最终存在：

- `/workspace/datasets/local_xlerobot_box2basket_left_head`

（你也可以把 `VLA_DATASETS_DIR` 直接设成数据集目录本身：`/workspace/datasets/local_xlerobot_box2basket_left_head`，代码也能识别。）

### 1.2 安装依赖

在服务器 Python 环境里（建议 Python 3.10）：

- 推荐直接安装我们提供的 server 依赖文件：
  - `pip install -r Being-H_xlerobot/requirements_server.txt`

说明：
- 上游 `Being-H/requirements.txt` 固定写了 `opencv-python==4.12.0`，但 pip 上的 OpenCV wheel 版本是 `4.12.0.88` 这种带 build 后缀的形式，所以会报 “No matching distribution found”。`Being-H_xlerobot/requirements_server.txt` 已经把它改成 pip 可用的 `opencv-python-headless==4.12.0.88`（更适合无桌面的服务器）。
- `flash-attn` 在部分服务器上（CUDA 版本/torch 版本组合）会无法安装或需要从源码编译，成本很高。本项目在 `Being-H_xlerobot/flash_attn/` 提供了一个兼容 fallback（用 PyTorch SDPA 实现）来避免因为 `flash_attn` 缺失而导入失败；因此 **不是必须安装**。

如果你 **一定要安装** `flash-attn`（加速推理/注意力）：

- 你的 torch/cu 组合如果没有预编译 wheel，pip 会自动走源码编译；源码编译经常因为内存不足导致 `cicc Killed`。
- 对于 4090（sm_89），建议只编译 8.9 架构并限制并行度：

```bash
# 可选：给编译留足空间（/tmp 需要足够磁盘；内存不够就加 swap）
mkdir -p /workspace/tmp
export TMPDIR=/workspace/tmp

# 关键：只编译 4090 需要的架构 + 降低并行度，避免 OOM
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=4
export NVCC_THREADS=1

pip install -v flash-attn --no-build-isolation --no-cache-dir
python -c "import flash_attn; print('flash_attn ok', flash_attn.__version__)"
```

### 1.3 运行训练

示例（4 卡）：

- 直接用 Hugging Face 模型名（脚本会自动下载到缓存）：
  - `torchrun --nproc_per_node=4 Being-H_xlerobot/train_lora_beingh_xlerobot.py --base_model_path BeingBeyond/Being-H05-2B_libero_robocasa --dataset_config_file Being-H_xlerobot/dataset_configs/xlerobot_box2basket_left_head.yaml --output_dir /path/to/out/xlerobot_box2basket_lora --max_steps 2000 --learning_rate 1e-4 --gradient_accumulation_steps 1 --num_workers 4`

  默认下载位置：Hugging Face 的默认缓存目录（通常是 `~/.cache/huggingface/hub`）。如果你想手动指定缓存目录，再加 `--hf_cache_dir /path/to/hf_cache` 即可。

- 已经提前下载好/本地目录（也可）：
  - `torchrun --nproc_per_node=4 Being-H_xlerobot/train_lora_beingh_xlerobot.py --base_model_path /path/to/Being-H05-2B_libero_robocasa --dataset_config_file Being-H_xlerobot/dataset_configs/xlerobot_box2basket_left_head.yaml --output_dir /path/to/out/xlerobot_box2basket_lora --max_steps 2000 --learning_rate 1e-4 --gradient_accumulation_steps 1 --num_workers 4`

训练输出（在 `--output_dir` 下）：

- `experiment_cfg/xlerobot_posttrain_metadata.json` (推理所需 normalization 统计信息)
- `lora_adapter/` (LoRA: `adapter_config.json` + `adapter_model.safetensors`)
- `finetuned_model/` (默认会导出：把 LoRA merge 到 base 权重后的 self-contained 目录，可直接用于推理)
- `run_config.json` / `train.log`

## 2) 服务器：启动推理服务 (ZMQ)

示例（单卡推理）：

- 推荐（直接加载 self-contained 微调模型）：
  - `python Being-H_xlerobot/run_server_xlerobot.py --model_path /path/to/out/xlerobot_box2basket_lora/finetuned_model --host 0.0.0.0 --port 5555 --device cuda:0`

- 兼容模式（base + LoRA + 外部 metadata）：
  - `python Being-H_xlerobot/run_server_xlerobot.py --model_path /path/to/Being-H05-2B_libero_robocasa --lora_path /path/to/out/xlerobot_box2basket_lora/lora_adapter --metadata_path /path/to/out/xlerobot_box2basket_lora/experiment_cfg/xlerobot_posttrain_metadata.json --host 0.0.0.0 --port 5555 --device cuda:0`

注意：

- 确保服务器防火墙放行 `5555/tcp`，笔记本能访问到服务器 IP。

## 3) 笔记本：连接 XLerobot 并验证

机器人连接信息（按你的描述）：

- 串口：
  - COM4: 左臂 + 头部电机
  - COM3: 右臂 + 底盘
- 摄像头：
  - 0: 中央/头部摄像头
  - 1: 左臂摄像头
  - 2: 右臂摄像头（本任务不用）

运行客户端（默认：head=0, left_wrist=1；每 16 步重新规划一次）：

- `python Being-H_xlerobot/robot_client_xlerobot.py --server_host <SERVER_IP> --server_port 5555 --port1 COM4 --port2 COM3 --head_cam 0 --left_wrist_cam 1 --hz 10 --replan_every 16 --steps 160`

可选：

- 需要手动校准时加 `--calibrate`
- 安全限制（每步最大关节变化）可调 `--max_relative_target`；设为 `0` 或 `-1` 禁用

## 4) 文件索引

- 训练脚本：`Being-H_xlerobot/train_lora_beingh_xlerobot.py`
- 推理服务：`Being-H_xlerobot/run_server_xlerobot.py`
- 笔记本控制/验证：`Being-H_xlerobot/robot_client_xlerobot.py`
- XLerobot 数据集注册：`Being-H_xlerobot/configs/dataset_info.py`
- XLerobot 模态定义：`Being-H_xlerobot/configs/data_config.py`
- LeRobot v3 数据读取适配：`Being-H_xlerobot/dataset/lerobot_v3_iterable_dataset.py`
- 数据集配置 YAML：`Being-H_xlerobot/dataset_configs/xlerobot_box2basket_left_head.yaml`
