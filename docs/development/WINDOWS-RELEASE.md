# Windows v0.3.0-rc.1 源码预览与目录包构建

v0.3.0-rc.1 是 [GS-Studio](https://github.com/Astroite/GS-Studio) 的仅源码（source-only）预览基线，**没有 Windows 二进制附件**。源码提交/推送遵循[发布保护](SOURCE-PUBLISHING.md)。以下目录包步骤供源码构建者与私有组装使用；`public` 是打包配置名，不代表组装结果已取得分发许可或可供下载。

目录包结构是 Tauri 桌面 + React/Spark 前端 + 一个共享 Python 环境，预处理与训练各自运行 worker 进程。入口为 `GS Studio.exe` 与包内 `gsstudio.ps1`，不再使用 Qt/PyInstaller 的四套冻结入口。

## 依赖与配置

目标为 Windows x64、Python 3.10.22、Torch 2.8.0+cu129、torchvision 0.23.0+cu129 和匹配的原生扩展。`python` 与 `preprocess_python`、`core_runtime_root` 与 `preprocess_runtime_root` 必须分别指向同一个解释器与根目录。Python 依赖锁由打包配置的 `package.python_lock_file` 指定；原生扩展必须有源哈希、二进制哈希与目标 GPU 架构证据。实际 GPU 验证仅覆盖 compute capability 8.9，不能推定其他架构可用。

开发环境的 [52-distribution 联合锁](../../scripts/runtime/unified-requirements.lock.txt)由冻结的 44 项核心和 8 个带 SHA256 的流水线 wheel 组成；[provisioning 工具](../../scripts/provision_runtime.py)仅向新的不存在目录复制核心并离线添加这 8 项，不改原环境或升级核心。41 项运行时与 11 项构建/测试锁用于区分用途；裁剪包保留审计 metadata/许可证，不承诺带有测试或安装器载荷。复现、保留旧环境回滚及裁剪规则见[运行时瘦身](RUNTIME-SLIMMING.md)。

目录包组装输入包括解释器、扩展、工具、离线模型和 notices；公开分发须另行闭合下述第三方证据门槛。需要本机 NVIDIA 驱动；开发编译工具链不属于用户运行依赖。RealityScan 单独安装，通过运行时配置指向用户自己的安装；COLMAP 可作为另一重建后端。

COLMAP 保持固定的 4.2.1 工具身份及其合法链接的 Qt DLL，不要求使用无 Qt 构建，也不通过删 DLL 改造该工具。排除的是退役 GS Studio PySide/PyQt 界面及其绑定。旧 native trainer/editor runtime 仅供源码验证，不作为现行发布入口。

public 配置默认不分发 Insta360 SDK、其模型或依赖 SDK 的 helper 二进制。INSV 用户需自行取得许可、安装 SDK，并按 [helper 说明](../../tools/mediasdk-helper/README.md)从自有 bridge 源码构建，再配置 `media_helper` 与 `mediasdk_root`。可在 `%LOCALAPPDATA%\GSStudio\runtime.override.json` 中保存本机覆盖；不要把商业 SDK 加进源码或公开目录包。local profile 只供已有合法 SDK 的本机组装使用，不能据此推定有公开再分发许可。

public 包载荷路径相对包内 `runtime/runtime.json`，用户可写数据默认位于 `%LOCALAPPDATA%\GSStudio` 下的 projects/cache/logs。`GSSTUDIO_DATA_ROOT` 只覆盖项目根；各入口仍共用用户级 GPU 锁。`GSSTUDIO_RUNTIME` 可显式选配置，`GSSTUDIO_RUNTIME_OVERRIDE` 可选本机覆盖文件。保持整个目录包一起搬移。

## 构建与计划

在仓库根目录使用已有开发工具链构建前端及 Tauri：

```powershell
Push-Location desktop
try {
    npm ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
    npm test
    if ($LASTEXITCODE -ne 0) { throw 'frontend tests failed' }
    npm run tauri -- build --no-bundle
    if ($LASTEXITCODE -ne 0) { throw 'Tauri build failed' }
} finally { Pop-Location }
```

依赖来自 `desktop/package-lock.json` 与 `desktop/src-tauri/Cargo.lock`。不得用无锁安装重新解析前端依赖。JS notice 生成随前端构建执行；原生 NOTICE/LICENSE、Python/工具/模型许可证由明确的打包输入保留。

准备自己的 runtime 配置并先生成只读计划：

```powershell
$Config = (Resolve-Path '.\private\runtime.json').Path
$Output = Join-Path $env:TEMP 'GS-Studio-0.3.0-rc.1-new'
& .\desktop\scripts\package-gsstudio.ps1 -RuntimeConfig $Config -Output $Output -Profile public -PlanOnly
```

示例字段见 [runtime-package.example.json](../../desktop/scripts/runtime-package.example.json)，执行前按同一共享环境及脚本要求核对。计划成功不代表运行时齐备。环境、锁、离线权重、工具和 notices 准备并核验后，同一命令去掉 `-PlanOnly`、加 `-RuntimeReady` 才复制到新目录。组装不安装环境，不覆盖已有包；审查包清单、排除审计和 SHA256 后再验收。

## 验收与发布门槛

既有精简包的大小、清单哈希与检查范围保留在[瘦身记录](RUNTIME-SLIMMING.md)：public 载荷约 6.049 GiB，local-final 约 7.870 GiB，public 压缩卷约 2.551 GiB。它们是历史本地验证制品，不是新源码快照重建的公开附件；不能直接把历史包标记为新提交的构建。

每次目录包验收至少分开记录以下范围：

| 范围 | 所需记录 |
| --- | --- |
| 身份与载荷 | 源快照/提交、依赖锁、扩展源与二进制、完整包清单及其哈希；排除审计和许可证 |
| 独立运行 | 公共包无商业 SDK/秘密/私有路径；干净 PATH、无全局 Python/CUDA 编译器、中文/空格路径及整体搬移 |
| 功能与恢复 | 离线遮罩、真实 CUDA 前后向、五种输入的适用范围、QA 阻断、完整训练、暂停/恢复及故障证据 |
| 查看与编辑 | 原生选择、编辑状态、PLY/SPZ 回读、桌面视觉/交互；窗口启动另行记录 |
| 质量与资源 | 固定输入/划分与预算，逐视角画质、计时范围、显存口径及已知限制 |
| 再分发 | 对最终包内每个第三方载荷闭合来源、条款、notice、对应源码/构建资料及制品身份 |

只有与实际源文件、依赖锁和最终包哈希绑定的结果，才能标记该构建通过。历史集成包、精简候选包、最终包和搬移后的包各有独立范围，见[迁移记录](PIPELINE-MIGRATION.md)。CI 只提供源码/文档、有限 CPU、前端及 Rust 检查；不签名、发布或上传目录包，不使用自托管 GPU runner。

二进制分发另受[第三方分发证据](../distribution/third-party-sources.json)约束，目前状态为 `evidence-gathered-binary-release-blocked`。未闭合项包括 FFmpeg 对应源码/静态依赖构建资料、COLMAP 依赖及 Qt/其他许可绑定、权重分发条款、GPL `plyfile` 分发条件、CUDA/cuDNN 协议和 Python 原生依赖资源。Apache-2.0 仅覆盖自有代码；保留 LICENSE/NOTICE 或通过哈希检查不等于取得第三方分发权。Insta360 SDK 外接、RealityScan 另装的边界保持不变。
