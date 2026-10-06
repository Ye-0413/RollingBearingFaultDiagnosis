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

- 核心依赖固定为 `torch==2.14.0`、`torchvision==0.29.0` 和 `Pillow==12.3.0`。torch 与 torchvision 必须使用这组配套版本。需要 CUDA 或 Linux/Windows CPU 专用安装时，请参考 [PyTorch 官方平台安装说明](https://pytorch.org/get-started/locally/)，先确认所选平台安装源提供这组固定版本的软件包，并在命令中保持上述版本，然后再执行上面的依赖安装命令。macOS 的预编译软件包要求 Apple Silicon 和 macOS 14.0 及以上，满足条件时可直接使用上面的命令。

- 安全版本的选择依据 Pillow 的 [GHSA-xj96-63gp-2gmr](https://github.com/python-pillow/Pillow/security/advisories/GHSA-xj96-63gp-2gmr)、[GHSA-jjj6-mw9f-p565](https://github.com/python-pillow/Pillow/security/advisories/GHSA-jjj6-mw9f-p565)、[GHSA-5x94-69rx-g8h2](https://github.com/python-pillow/Pillow/security/advisories/GHSA-5x94-69rx-g8h2)，以及 PyTorch 的 [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p) 和 [GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47)。JIT 问题采用 [PyTorch 2.14.0 官方发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.14.0) 中确认包含的修复；公告曾将 2.13.0 列为修复版本，但该修复未进入 2.13 发布分支，详见[公告更正](https://github.com/github/advisory-database/pull/9857)。请保持项目现有的 `weights_only=True` 权重加载方式。
