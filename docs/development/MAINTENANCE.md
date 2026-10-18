# GS Studio 维护说明

操作以[当前手册](../user/CURRENT-WORKFLOW.md)为准；源码分层见[工程结构](STRUCTURE.md)，开发命令见[贡献指南](../../CONTRIBUTING.md)，提交与推送约束见[源码发布](SOURCE-PUBLISHING.md)，目录包门槛见[Windows 发布](WINDOWS-RELEASE.md)。历史验收只证明记录中的输入、依赖和制品，不提供当前安装或 Run 恢复命令。

## 当前实现

Tauri + React + Spark 是桌面主线。Rust 管理 JSONL worker 和后代进程；`src/gsstudio/application` 是 CLI 与桌面的统一操作层。预处理 worker、独立训练核心与原生编辑使用同一锁定 Python、Torch 2.8.0+cu129 / torchvision 0.23.0+cu129 和配套原生扩展，在不同进程中运行。发布包不在启动时下载权重或 JIT 编译扩展。

`scripts/gs-studio.ps1` 与根 `gsstudio.ps1` 是目录包代理；`scripts/session.ps1` 仅供已配置的源码开发环境使用，不改变全局环境。使用 `GSSTUDIO_RUNTIME` 显式选择开发配置。发布时 `python`/`preprocess_python` 必须解析为同一解释器；不得靠多套 site-packages 或环境回退凑出运行时。

无现行生产导入方的 Qt 界面和严格 Qt 专属测试已归档；混合 worker 协议测试保留非 Qt 用例。`pipeline/training/native.py` 与 `pipeline/editor/runtime.py` 只作为源码侧旧支持模块保留，现行发布入口使用独立引擎，不添加兼容路径。开发不再推荐旧 `studio` extra。COLMAP 4.2.1 的合法 Qt DLL 依赖应保留，不能按退役 GS Studio PySide/PyQt 一并排除。

主要运行时字段为 `python, preprocess_python, engine_root, pipeline_root, data_root, ffmpeg, ffprobe, colmap, media_helper, mediasdk_root, realityscan, realityscan_export_params, torch_home, torch_extensions_dir, cache_root, log_root, gpu_lock_path`。相对路径相对所在 JSON 解析。公共目录包使用可搬移的载荷路径，用户项目、缓存、日志与共享锁在用户目录；具体覆盖顺序以运行时 resolver 和目录包说明为准。

## 保留的脚本

| 脚本 | 用途与边界 |
| --- | --- |
| `desktop/scripts/package-gsstudio.ps1` | 当前 public/local 目录包计划与组装；不安装环境，输出到新目录 |
| `desktop/scripts/test-package-plan.ps1`、`test-gsstudio-cli.ps1` | 临时 fixture 中检查打包与 CLI；后者仍引用 `scripts/gsstudio-next.ps1`，因此保留该代理 |
| `engine/scripts/setup-build-*.cmd`、`build_studio_selection.*` | 原生编译源入口；部分开发默认值需按目标机器核对，不是用户安装步骤 |
| `engine/scripts/setup-runtime*`、`setup_record_runtime.py` | 独立引擎运行时构建/记录资料；不自动成为发布包依赖锁 |
| `engine/scripts/record_splat3_renderer_runtime.py`、`record_splat3_round4_runtime.py` | 原生制品与配置记录；当前 CPU 测试直接引用 |
| `engine/scripts/evaluate_reference_pngs.py`、`render_reference_cameras.py` | 显式标定与图像比较工具；被现有测试验证 |
| `engine/scripts/studio_synthetic.py`、`studio_selection_cases.py`、`validate_studio_*.py` | 合成数据和核心验证；GPU 脚本只在明确安排的本地任务运行 |
| `scripts/migration_retirement.py` | 独立维护审计工具，测试验证路径/消费者/锁保护；不是常规清理入口，也不读取旧 Run |

旧 PyInstaller/frozen、cu130 引导、无引用的激活准备脚本及一次性 RE/实验脚本已归档，见[索引](../archive/README.md)。保留脚本表示仍有可说明的职责或测试消费者；名称中出现旧内部实现名，不构成再次运行或默认启用的依据。

无当前调用方的外部 Postshot 后端/对照入口和孤立的历史私有验收脚本另见[快照整理清单](../archive/source-retirement-20261018.json)。它们不再作为当前开发或验收步骤；保留的原生源码、数值夹具及仍被测试引用的工具不得按名字批量清理。

## 状态与训练契约

只有一个当前 Run 配置。旧 schema Run/checkpoint 不读取、不恢复。配置、输入哈希、阶段证据、分段 QA 与训练结果共同决定状态；不能手改失败状态或绕过哈希校验。有效共享数据包是一种独立输入，不是旧 Run 兼容通道。

Run 默认每秒一个候选、一张初选，弱区间局部 2 fps 补抽最多一次。遮罩审核、分段可训练性与路线覆盖独立；产品的审核功能保留，历史任务授权不构成新任务授权。

photo/gs 是输入预设，quality_v2/observed_v5/observed_v6 是独立配方。前两者数学冻结，v6 实验性。photo 使用最长边 1600 的真实派生图像/遮罩/内参/观测更新，源图、位姿、世界尺度和可信分组不变。深度请求与有效启用分开记录；无有效锚点不得显示启用。

保存暂停退出并释放 GPU；热挂起保留资源。恢复使用同身份 `checkpoints/latest.pt`，失败无检查点重试须显式从零开始。完整预算、评估及包/引擎/检查点/模型哈希核验是成功条件；退出码零、PLY 存在或提前结束均不足。

## 检查与清理

按更改选择检查，记录实际结果，不复用历史通过计数作为新提交的结果：

| 范围 | 检查与证据 |
| --- | --- |
| 源码、文档与发布边界 | `python .github/check_source.py`、`git diff --check`、源码 payload 守卫及守卫单元测试；具体入口见贡献指南 |
| 产品接口与桌面代码 | `tests/unit`、`tests/contract`、锁定前端测试/构建及 Rust 测试；需要已配置的开发依赖 |
| CUDA、真实素材与目录包 | 显式安排本地任务，绑定输入、源文件、锁与最终包哈希，分别记录功能、质量、恢复和分发条件 |

[CI](../../.github/workflows/source-checks.yml)只覆盖源码/文档卫生、有限 CPU 契约、锁定前端及 Rust 单元检查。它不验证 CUDA、真实重建、画质、离线包或交互行为。部分 `engine/tests` 会执行 CUDA，不能无筛选地用整套核心测试作 CPU 检查。文档整理不触发 GPU、训练、重建或安装。

输出放 `tests/output` 或仓库外。原生 Windows 日志为 UTF-8，worker stdout 仅 JSONL，诊断写 stderr。不要把合成 fixture 成功当作训练接受结果；既有接受范围见[迁移记录](PIPELINE-MIGRATION.md)。

运行时 cleanup 必须在 Run 锁内核验 retention 白名单、消费者和哈希。它不删除原片、其他 Run、环境、SDK、密钥或全局缓存，也不凭退出码清理失败阶段。源码归档另有逐文件清单，不能拿它当运行数据清理规则。

目录整理保留未提交工作。先确认生产导入方、测试消费者、许可证与文档链接，再决定迁移；归档先复制并核验路径、大小和 SHA256，再逐文件移除，公开索引不含私人位置或内容。源码发布只包含审查后的 Git 快照，不推送旧 refs、私有备份或 LFS 对象；本地原始数据的保留与远端发布是独立操作。
