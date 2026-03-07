# 服务器专属：G-RIS 环境与数据极速配置指南

本指南专为你今晚在 A100/H100 服务器上**直接下载数据**与**配置环境**而设计。
**绝不要再用本地电脑下载再跨网传输了！** 请在通过 SSH 连上服务器后，直接在服务器终端复制粘贴运行以下命令。

## 任务 1：极速下载 MS-COCO 原始图片 (13GB+)

默认服务器系统为 Linux (Ubuntu/CentOS)。建议使用 `wget` 或多线程工具 `aria2c`。

### 方案 A：最稳妥的单线程下载 (系统自带 wget)
请先进入你打算存放数据集的目录 (例如 `mkdir datasets && cd datasets`)，然后执行：

```bash
# -c 参数表示支持断点续传（如果网络断开，重新运行此命令会接着下）
wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
```

### 方案 B：满带宽多线程狂飙 (强烈推荐安装 aria2)
如果你的服务器带宽有 1000M，用这个只需几分钟就能下完 13GB：

```bash
# 1. 如果你有 sudo 权限，先安装 aria2
sudo apt-get install aria2  # Ubuntu
# 或者当你只有 conda 时： conda install -c conda-forge aria2

# 2. 开 16 个线程极速下载
aria2c -x 16 -s 16 -c http://images.cocodataset.org/zips/train2014.zip
aria2c -x 16 -s 16 -c http://images.cocodataset.org/zips/val2014.zip
```

**下载完后解压：**
```bash
unzip train2014.zip
unzip val2014.zip
# 注意：解压 13GB 碎文件可能会消耗几十秒到几分钟，耐心等待。
```

---

## 任务 2：获取 gRefCOCO / RefCOCO 标注文件

通常这些 JSON 标注文件都托管在 GitHub 或 Google Drive 上。

1. **获取官方 gRefCOCO 标注 (以 ReLA 仓库为例)**:
```bash
# 克隆 gRefCOCO 官方库，里面通常包含标注获取脚本或链接
git clone https://github.com/henghuiding/ReLA.git
cd ReLA
```

2. **如果标注文件存在 Google Drive 上 (国外服务器专享神技 `gdown`)**:
很多开源库把标注放谷歌云盘。不要用浏览器下！在终端安装 `gdown`：
```bash
pip install gdown
# 使用分享链接的 file_id 下载 (你需要去 README 里找到对应的 file_id)
# 示例：gdown https://drive.google.com/uc?id=XXXXXXXXXXXXXXX
```

---

## 任务 3：后台挂机大法 (防断网秘籍)

因为今晚你要挂着下几十个 G 的数据还要配置环境。为了**防止你电脑休眠或者 SSH 断开导致服务器下载中断**，你必须学会**终端复用工具** `tmux` 或 `screen`。

```bash
# 当你刚连上服务器，输入：
tmux

# 此时屏幕闪了一下，你进入了一个“永不离线”的虚拟终端。
# 现在你可以在里面执行 wget 下载命令，或者运行 conda 训练脚本了。

# 👉 如何退出而不中断任务：按下键盘 Ctrl+B，松开，再按 D
# 此时你退回了原本的终端，但下载任务还在后台狂奔！你可以安心关电脑睡觉。

# 👉 明天早上如何重新看进度：再次连上服务器，输入：
tmux attach
# 你就会神奇地回到昨晚正在下载/训练的那个界面！
```

## 任务 4：一键配置 Conda 深度学习环境

假设你今晚克隆了类似于 `CRIS` 或 `ReLA` 的代码，你需要准备环境。

```bash
# 1. 创建全新的 Python 虚拟环境 (隔离冲突)
conda create -n gris_env python=3.8 -y
conda activate gris_env

# 2. 安装地表最强算力显卡对应的 PyTorch 版本 (A100/H100 支持较高的 CUDA 版本，如 11.8 或 12.1)
# 请先输入 nvcc -V 确认你的 CUDA 核心版本，假设为 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 安装依赖库 (去你克隆的代码目录下找 requirements.txt)
# pip install -r requirements.txt
```

**今晚请务必完成：COCO数据集下载完毕 + 环境 build 成功。祝你明早醒来直接开跑！**
