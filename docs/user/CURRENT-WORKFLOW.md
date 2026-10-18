# GS Studio 当前工作手册

本页是当前操作入口。桌面为 Tauri + React + Spark，CLI 与桌面调用同一操作层：原片/照片 → 预处理/遮罩 → 重建 → 分段 QA → 不可变训练包 → 独立训练与评估。预处理和训练使用同一 Torch 2.8/cu129 Python 环境的不同 worker 进程。开发及发布说明见[维护](../development/MAINTENANCE.md)与[Windows 目录包](../development/WINDOWS-RELEASE.md)，既有验收范围见[迁移记录](../development/PIPELINE-MIGRATION.md)。

## 1. 启动与输入

完整解压目录包，运行 `GS Studio.exe`。无需全局 Python。CLI 示例在包目录的 PowerShell 中执行；素材和项目在包目录外，实际路径由用户选择。

```powershell
$Studio = (Resolve-Path '.\gsstudio.ps1').Path
& $Studio --help
& $Studio status --json
```

运行需要可用 NVIDIA 驱动；当前实测 GPU 架构只有 8.9。RealityScan 需另行安装并配置，也可明确使用 COLMAP。公开包的 INSV 需要外接获许可的 Insta360 SDK/helper。所有 GPU 操作共用锁；`doctor --backend gsplat` 也会探测 GPU，应安排在空闲时段运行。

地点组织素材，场景表示连续空间，Capture 固定源文件与窗口。源文件按顺序记录路径、大小和 SHA256，只读引用；修改窗口/处理配置创建新 Capture 或 Run。Run 配置和输入哈希不在恢复时改写。

| source-type | 输入 | auto 训练预设 |
| --- | --- | --- |
| `insta360_insv` | 一/两个明确指定的 INSV，helper 拼接后投影视图 | gs |
| `equirect_video` | 标准 2:1 全景视频 | gs |
| `equirect_sequence` | 2:1 全景图片序列，显式序列 fps | gs |
| `perspective_video` | 普通视频，每帧一个透视视图 | gs |
| `perspective_images` | 普通照片，保留实际尺寸，不投影 | photo |

```powershell
$L = 'my-location'
$S = 'my-scene'
$C = 'capture-01'
$Source = (Resolve-Path '..\media\capture.insv').Path
& $Studio location init $L --name '地点'
& $Studio scene init $L $S --name '场景'
& $Studio capture init $L $S $C --source-type insta360_insv --source $Source --camera-model X6 --output-width 3840 --output-height 1920 --start-seconds 0 --end-seconds 45
& $Studio media probe $L $S $C
& $Studio ingest $L $S $C
```

普通照片用 `capture init ... --source-type perspective_images --source <照片目录>`。不要把透视重构的 16:9 视频当作全景。普通照片/视频的焦距可由重建估计，全景分视图保持固定投影内参；RealityScan 当前采用无畸变针孔路径，COLMAP 可估计畸变并去畸变。更多采集建议见 [SOP](CAPTURE-SOP.md)。

素材迁移用 `capture relink L S C --source <新路径>`，多文件按原顺序重复，大小与哈希全部一致才更新路径。不要把数据搬移变成重新分组或覆盖原片。

## 2. Run、审核与重建

```powershell
& $Studio preprocess $L $S $C --mask-review-gate
$R = '<实际输出的 Run ID>'
& $Studio mask $L $S $R
& $Studio mask-review $L $S $R --attempt primary
# 逐图审核 ok/exclude 并提交后，才固定集合。
& $Studio mask-finalize $L $S $R --attempt primary
& $Studio reconstruct $L $S $R
& $Studio qa report $L $S $R
& $Studio status $L $S $R --json
```

| 设置 | 当前默认 |
| --- | --- |
| Run 候选/初选 | 1 fps，每秒一张 |
| 全景投影 | 14 视图，110°，1746²，pitch −45°/0°/+45°，底裁 15% |
| 遮罩 | Mask R-CNN，score .25、probability .50、推理 gamma .75、膨胀 24 px、闭合 7 px |
| 整图剔除 | 遮罩比例严格大于 .005；等于保留 |
| 重建 | RealityScan；可在 preprocess 时显式 `--reconstruction-backend colmap` |
| 修复 | 弱区间前后各 2 秒，局部 2 fps，最多一次，按源帧索引去重 |
| retention | minimal；新 Run 可用 `--keep-intermediates` 保留调试输入 |

桌面新 Run 要求遮罩审核；CLI 上例明确开启 gate，CLI 默认不因界面规划暗改。全景先投影再遮罩，gamma 只影响人物检测。遮罩、审核身份和固定集合均带哈希；repair 有独立审核身份，primary 的决定不能代替 repair。外部 vision 默认关闭，发送图像到外部服务需要明确授权。

数据完整性、分段可训练性、全路线覆盖分别显示。分段至少十个采样、三个位置，实际分段注册率至少 70%；缺文件、相机错配、非有限位姿或错误时间戳会阻断。首尾缺失、长缺口和未注册区间单独报告。一个分段通过不能证明全路线完整，也不自动启动全量训练。

阅读实际报告后接受或拒绝：

```powershell
& $Studio qa approve $L $S $R --notes '实际验收意见'
# 或 qa reject ... --notes '实际拒绝原因'
```

QA 接受前训练入口拒绝。失败 Run 不手改成功，不平滑异常轨迹掩盖失败，不从归档找旧 schema 恢复命令。

## 3. 训练输入与预设

`train-options` 会写入/核验真实训练包，但不训练、不批准 QA。`--dry-run` 准备实际输入及命令，不执行训练。

```powershell
$Segment = 'segment-001' # 替换为实际通过的 ID
$Output = Join-Path (Resolve-Path '..\experiments').Path 'sample-01'
& $Studio train-options $L $S $R --segment $Segment
& $Studio train $L $S $R --segment $Segment --output $Output --dry-run
# 确认代表分段和资源后，去掉 --dry-run 执行。
```

| 默认值 | photo：照片/原始外部 COLMAP | gs：全景/视频分段 |
| --- | --- | --- |
| recipe | quality_v2 | quality_v2 |
| 步数 | 10000 | max(30000, 30×训练图数) |
| SH / 点数上限 / seed | 3 / 1000000 / 42 | 3 / 1000000 / 42 |
| 尺寸 | 最长边 1600，只缩小 | 原尺寸 |
| 光度 / bilateral / sparse depth 请求 | 全关 | 全开；深度取决于有效锚点 |

`--training-preset auto|photo|gs` 独立于 `--training-recipe quality_v2|observed_v5|observed_v6`。quality_v2/v5 冻结，v6 实验性。auto 使用 Capture 源类型；有效共享包保留可信 split，不靠文件名猜分组。旧 Run 或旧检查点不支持恢复。

`--steps`、`--photo-comp/--no-photo-comp`、`--use-bilateral-grid/--no-use-bilateral-grid`、`--use-sparse-depth/--no-use-sparse-depth` 是实际覆盖。启用 bilateral 必须同时启用 photo_comp，否则拒绝；关闭 photo_comp 关闭整个光度模块。`--duration-seconds 30` 只选通过分段内再次满足条件的最早完整窗口，不足则拒绝。预览周期不改变训练预算。

缩放创建独立有哈希的派生包，真实更新图像、二值遮罩、K、COLMAP 观测和 depth u/v；位姿、世界点和 depth z 保持尺度。颜色只由有效 TRAIN 观测重算，验证组不参与；至少四个各有两个不同训练视图观测的初始化点才可用。

深度可用性来自核心验证：`sparse_depth_available/observations/reason` 描述真实锚点，`sparse_depth_requested` 记录请求，`use_sparse_depth` 记录有效启用。无锚点时有效值必须 false；坏锚点或哈希失败拒绝。预设、显式覆盖、配方与源/派生包身份改变时新建实验。

## 4. 外部 COLMAP

```powershell
$Images = (Resolve-Path '..\input\images').Path
$Model = (Resolve-Path '..\input\sparse\0').Path
& $Studio external inspect --images $Images --model $Model --independent
& $Studio external prepare 'photos-import' --images $Images --model $Model --independent
& $Studio external options 'photos-import'
& $Studio external train 'photos-import' --dry-run
```

`--independent` 明确确认每图独立；全景/时序相关输入须用完整 `--groups <JSON>`，已有共享包用 `--package <目录>` 并保留可信 split。inspect 检查多模型、缺图、相机、遮罩、稀疏点和输入身份；prepare 可用 `--expected-input-identity` 绑定预览。导入写 `.external-imports/<id>`，源只读，不冒充 Run QA。迁移用 `external relink --new-root` 或 `--mapping-file` 逐项核验。

## 5. 暂停、恢复与成功证据

| 动作 | 语义 |
| --- | --- |
| `train-control <output> pause` | 安全步保存并退出，paused，释放 GPU |
| `train-control <output> suspend` | 热挂起，保留进程/显存/锁 |
| `train-control <output> unpause` | 同进程继续 |
| `train-control <output> checkpoint` | 安全边界立即保存，继续训练 |
| `train-control <output> stop` | 保存后提前结束，stopped，不算成功 |
| `train ... --resume` | 同身份 `checkpoints/latest.pt` 的新进程恢复 |

控制绑定包哈希与 generation；过期请求不能影响新任务。首次检查点前失败，只有显式 `--resume --restart-from-step-zero` 可从零重试；已有检查点时拒绝该标记。OOM 保留失败证据，不暗改尺寸或 SH。

Run 恢复先确认无活动进程/锁，再按实际阶段使用 `preprocess ... --run-id $R --resume`、`mask ... --resume`、`reconstruct ... --resume` 或 `qa report ... --resume`，不无条件重跑全部。

成功需要完整预算、评估完成及检查点/引擎/包/模型身份一致。manifest/config_hash/mask-final/segments 决定 Run 与 QA；dataset.json 固定训练输入；dispatch.json/train.log 记录真实调度；config.json、training.json、evaluation/metrics.json、checkpoints/latest.pt 和 model.ply 共同验证训练。暂停、提前结束、退出码零或 PLY 存在均不足。

查看实际片段的模糊、人物残留与伪影。全景质量当前偏低；分段训练完成不代表路线完整。模型预览与源相机按包身份和请求序号配对，坐标/编辑语义见[桌面说明](STUDIO.md)。

## 6. 保留与清理

minimal 只回收通过消费者核验的白名单中间文件，保留重建图/遮罩/COLMAP、位姿、模型、检查点、日志和溯源。共享文件可能是硬链接，不得原地改写。

`cleanup L S R` 仅预览，`--apply` 才在 Run 锁内执行消费者/哈希校验并记录逐项结果。清理失败保持 pending，不改变计算状态；不按退出码清理失败阶段，不删除原片、其他 Run、实验、全局缓存、SDK、密钥或环境。

## 7. 独立选帧

select-sharp 从显式 source/out 选清晰帧，只写 out，不创建 Capture/Run，也不训练。其独立候选默认 5 fps，Run 仍为 1 fps。按实际源类型明确设置 `--kind`：

```powershell
$Pick = Join-Path (Resolve-Path '..\experiments').Path 'pick-01'
& $Studio select-sharp --source $Source --kind insta360_insv --out $Pick --window 0..30 --candidate-fps 5 --mask-threshold 0.01
```

支持 probe/extract/select/project/mask/filter/all 分阶段重复执行并核验已有哈希。`--keep-candidates` 保留候选，默认选帧后删除；投影/遮罩仍遵循共享 GPU 锁。不同窗口使用不同输出，不用此工具绕过 QA 或资源调度。
