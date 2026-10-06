环境搭建
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)

- 建议使用[Anaconda](https://www.anaconda.com/)创建独立环境。当前 Pillow 和 PyTorch 版本要求 Python 3.10 及以上；本项目建议使用 Python 3.11，以兼容 `requirements.txt` 中的其他固定版本。

```bash
conda create -n bearing python=3.11
conda activate bearing
```

- 从项目根目录安装依赖，版本以根目录的 `requirements.txt` 为准：

```bash
python -m pip install -r requirements.txt
python -m pip check
```

- 核心依赖固定为 `torch==2.13.0`、`torchvision==0.28.0` 和 `Pillow==12.3.0`。torch 与 torchvision 必须使用这组配套版本。需要 CUDA 或 Linux/Windows CPU 专用安装时，请参考 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v2130)，选择对应平台的 **2.13.0** 命令，然后再执行上面的依赖安装命令。macOS 可直接使用上面的命令。

- 安全版本的选择依据 Pillow 的 [GHSA-xj96-63gp-2gmr](https://github.com/python-pillow/Pillow/security/advisories/GHSA-xj96-63gp-2gmr)、[GHSA-jjj6-mw9f-p565](https://github.com/python-pillow/Pillow/security/advisories/GHSA-jjj6-mw9f-p565)、[GHSA-5x94-69rx-g8h2](https://github.com/python-pillow/Pillow/security/advisories/GHSA-5x94-69rx-g8h2)，以及 PyTorch 的 [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p) 和 [GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47) 中列出的修复版本。请保持项目现有的 `weights_only=True` 权重加载方式。
