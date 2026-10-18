# GS Studio 工程结构

GS Studio 的源代码分为桌面、统一操作层和独立数值引擎。它们构成一个产品和一条 Capture → Run → Segment → Experiment 流水线。项目、环境、SDK、权重、构建输出和原始证据不属于源码树。

公开源码从经过检查的快照建立独立 Git 历史。开发入口是[贡献指南](../../CONTRIBUTING.md)，提交与推送的保护边界见[源码发布](SOURCE-PUBLISHING.md)；既有集成与包验证只在[历史验收记录](PIPELINE-MIGRATION.md)中陈述。

| 位置 | 职责 |
| --- | --- |
| `desktop/src/` | React 界面、Three.js/Spark 视口和前端状态契约 |
| `desktop/src-tauri/` | Tauri/Rust supervisor、JSONL 传输、进程树与本地资产访问 |
| `desktop/scripts/` | 当前目录包组装、排除策略、notice 生成与包计划验证 |
| `src/gsstudio/domain/` | 清单、配置、身份和状态规则 |
| `src/gsstudio/application/` | CLI 与桌面共用的查询、创建、审核、调度和交付操作 |
| `src/gsstudio/interfaces/{cli,worker}/` | Typer 命令及结构化 JSONL 操作适配；launcher 启动目录包 |
| `src/gsstudio/pipeline/` | 输入、遮罩、重建、QA、训练包准备、独立引擎调度及 retention |
| `src/gsstudio/infrastructure/` | 工具适配、持久化、路径、运行时配置和锁 |
| `src/gsstudio/_vendor/` | 带来源及许可证的第三方代码；不受根许可证重新授权 |
| `engine/local_postshot/` | 数据验证、训练、恢复、原生预览、编辑和编解码；包名为内部实现名 |
| `engine/native/`、`engine/editor_native/` | 保留的 C++/CUDA 编译源及模块许可证 |
| `engine/scripts/` | 原生构建、运行时记录、可复用验证和合成数据工具 |
| `tools/` | 自有 MediaSDK bridge 源码和 RealityScan 导出参数；不包含商业 SDK |
| `tests/{unit,contract,manual}/`、`engine/tests/` | 单元、接口、数值与显式人工/GPU 验证；合成 fixtures 保留在 tests |
| `scripts/` | 已验收包的启动代理、源码 session 及仍被测试引用的维护工具 |
| `.githooks/`、`scripts/check_source_payload.py` | 源码提交/推送检查；防止媒体、模型、运行载荷与 LFS 对象进入 Git |
| `.github/` | 源码卫生、有限 CPU 契约、锁定前端和 Rust 的只读 CI |
| `docs/` | 当前使用/维护文档、架构契约及小型归档索引 |

根目录只保留产品说明、许可证/notice、贡献及变更记录、构建配置和启动代理。`desktop`、`src`、`engine`、`tools` 的职责不因历史来源而混合；不复制环境或 SDK 到这些目录。`tests/output` 与开发构建输出被忽略，测试需要的少量合成二进制夹具按精确路径和哈希保留，不构成真实数据的例外。

预处理、核心训练和编辑各用独立进程，但 `python` 与 `preprocess_python` 指向同一锁定 Python 环境。独立进程用于生命周期、故障和 GPU 调度隔离，不代表两套 Torch。运行时相对路径按其 JSON 配置文件目录解析；所有入口共用用户级 GPU 锁。

退役的 `src/gsstudio/interfaces/desktop/` Qt 界面与三个 Qt 专属契约测试已归档；混合 worker 测试保留现行协议覆盖。精确范围见[后续归档清单](../archive/source-retirement-20261017-002.json)。

`pipeline/training/native.py` 与 `pipeline/editor/runtime.py` 仍被有用的数值、锁和编辑测试使用，作为仅供源码验证的旧支持模块保留；现行发布入口不调用它们，不增加旧 Run 或检查点兼容。模块算法未改。COLMAP 4.2.1 自身链接的 Qt DLL 属于工具依赖，不属于已退役的 GS Studio PySide/PyQt 界面。

顶层旧 `tests/test_*.py`、Tk prototype、旧冻结构建和一次性实验已移到仓库外。逐文件原路径、SHA256 和理由见[首轮归档清单](../archive/source-retirement-20261017.json)。现行生产接口与核心数值测试、编译源、合成夹具和被测试引用的工具保持原位。

后续快照审查归档了无当前调用方的外部 Postshot 后端/对照包装器，以及绑定历史私有工作流的孤立手工验证脚本，见[2026-10-18 清单](../archive/source-retirement-20261018.json)。独立训练入口、现行运行时记录器、数值测试及最小夹具仍保留；这些归档不会把历史数据或旧源码祖先带入新 Git 历史。
