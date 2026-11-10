# Gemini 335 计算摄像学课程设计

**课题**　基于 Gemini 335 的浅景深合成与遮挡边界伪影抑制  
**小组**　2 人  
**计划周期**　2026-11-10 至 2026-12-09

我们用同一组 RGB-D 数据检验两种处理各自的作用：按深度分层渲染能否减少前后景颜色混合，以及修正低置信度深度能否进一步降低遮挡边界误差。

## 从这里开始

| 文件 | 用途 |
| --- | --- |
| [docs/proposal.md](docs/proposal.md) | 在线协作的开题正文 |
| [docs/weekly-plan.md](docs/weekly-plan.md) | 两人分工、进度和每周验收 |
| [docs/experiment-log.md](docs/experiment-log.md) | 记录预实验、参数、结果与失败样例 |
| [src/capture_rgbd.py](src/capture_rgbd.py) | Gemini 335 单组 RGB-D 采集与保存 |
| [src/validate_rgbd.py](src/validate_rgbd.py) | 离线核对深度和直边配准，登记样例与焦点 |

平时在 GitHub 上更新 Markdown，定稿时由一人把已合并的内容同步到 Word。采集原始帧与视频另行存放，实验记录注明数据编号和存放位置。

## Gemini 335 首次采集

固定 Gemini 335 和光照，在镜头前布置一个纸盒和一块颜色反差大的背景板，让纸盒有一段清晰、独立的直边。前后景先放在 0.5–2.5 m 内，记录从相机到对应表面的实测距离。关闭可能占用相机的预览程序，在本机 PowerShell 进入项目文件夹。电脑已安装依赖时可以直接运行。

```powershell
python .\src\capture_rgbd.py --lock-color-exposure
```

首次运行时可用 64 位 Python 3.12 创建项目环境并安装依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\src\capture_rgbd.py --lock-color-exposure
```

脚本使用相机默认的彩色和深度分辨率，跳过前 10 组完整帧，再保存一组数据到 `data/raw/<时间>/`。使用 `--lock-color-exposure` 时，脚本在预热后锁定当时的彩色曝光和增益，再等 10 组帧。退出时恢复原先启用的自动曝光。锁定失败时，可检查设备支持情况，或用不带该参数的命令采集并记录实际曝光设置。

| 输出 | 内容 |
| --- | --- |
| `rgb.png` | 彩色照片 |
| `depth_raw.npy` | 对齐前的原始深度整数，乘以 `meta.json` 的 `raw_depth_mm_per_unit` 得到毫米 |
| `depth_m.npy` | 对齐到彩色图的深度，单位米。用 `depth[y, x]` 读取彩色图坐标 `(x, y)` 的深度。无效值为 `NaN` |
| `meta.json` | 帧时间戳、分辨率、SDK 版本、深度比例、有效深度统计及曝光。曝光和增益使用 SDK 原始读数，帧未提供的字段为 `null` |

## 第一周核对与交接

打开采集到的 `rgb.png`，在纸盒和背景板内部各选一个像素，再沿同一段直边标两个端点。坐标用原图像素 `(x, y)`，左上角为 `(0, 0)`。端点要避开拐角，留出扫描线到图像边缘的空间。把下面的目录、坐标和两处实测距离替换为自己的数据。用虚拟环境时，将 `python` 换成 `.\.venv\Scripts\python.exe`。

```powershell
python .\src\validate_rgbd.py ".\data\raw\实际采集目录" --scene-id A-W1-01 --foreground 300 240 --background 540 240 --measured-m 0.85 1.60 --edge 410 160 410 320
```

- 对照输出的前后景深度与实测距离。脚本记录中心值、5×5 窗口的有效深度中位数、相对实测距离的差值和全图有效深度比例。
- 打开 `edge_check.png`。白线为 7 条扫描线，绿圈为彩色边，紫点为深度跳变，红叉为深度空洞，F 和 B 为两个焦点。彩色边在标注线附近 ±3 像素内查找，深度跳变至少达到实测前后景深度差的 25%。偏差为正表示深度边偏向背景。脚本会记录未匹配的截面和缺测坐标，空洞两端不会合并成一次跳变。
- 将整个样例目录打包交给 B。离线读取 `rgb.png`、`depth_m.npy` 和 `meta.json` 即可获取图像、米制深度和两个焦点。焦点在 `focus_points_xy.foreground` 与 `focus_points_xy.background`，深度用 `depth[y, x]` 访问。样例编号、配置版本及检查结果也写入 `meta.json`。
- 在 [实验记录](docs/experiment-log.md) 填实际读数、配准观察和共享数据位置。代码与记录通过 PR 提交，`data/raw/` 由 Git 忽略。

Windows 若已经识别相机，而 SDK 仍未找到设备，先退出 Viewer，再在本机 PowerShell 直接运行脚本。若本机仍失败，用 [Orbbec Viewer](https://www.orbbec.com/developers/orbbec-sdk/) 检查能否同时预览彩色和深度。

SDK 接口参考 [Orbbec Python SDK 文档](https://orbbec.github.io/pyorbbecsdk/source/5_Application_Guide/ApplicationGuide.html)。

## 两人怎么改文档

1. 打开要修改的 Markdown 文件，点击铅笔图标。
2. 完成修改后点 **Commit changes**，选创建新分支，给这次修改起个名字，例如 `docs/week1-capture`。
3. 点 **Propose changes** 并创建 **Pull Request（PR）**。标题写清改了什么。
4. 另一位组员在 **Files changed** 看改动。两人确认后，将 PR 合并到 `main`。

第一次可以各改一项不同的内容来熟悉流程。成员 A 填写采集与配准任务的验证方式，成员 B 填写 M0 的合成样例验收方式。每人开一个 PR，互相看完再合并。
