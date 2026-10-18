# GS Studio 工程方向

更新：2026-10-17。源码布局见[工程结构](../development/STRUCTURE.md)，协议见[引擎契约](ENGINE-CONTRACT.md)。

| 边界 | 当前要求 |
| --- | --- |
| 产品入口 | Tauri/Rust 管理 React/Spark 桌面；CLI 与桌面共享 application |
| 运行时 | 一个锁定 Torch2.8/cu129 Python 环境，多个独立 worker 进程；包不依赖源码路径或启动时 JIT |
| 数据 | 原片只读，清单/哈希/阶段/QA 共同决定状态；旧 Run/checkpoint 不读、不恢复 |
| 数值 | quality_v2/v5 冻结；v6 显式实验，不因文档/结构整理改变数学 |
| 发布 | public 包外接商业 SDK；RealityScan 另装；保留第三方许可证和排除审计 |
| 检查 | 托管 CPU/前端/Rust CI；真实 GPU、画质和目录包验收独立记录 |

Qt 界面经依赖核实已无退役界面外的生产导入，已与严格 Qt 专属测试归档。混合 worker 测试保留现行协议覆盖。旧 native trainer 和 editor runtime 仍被有效测试使用，作为仅供源码验证的支持模块保留，现行发布入口不调用；删除前须处理实际导入方并保留数值证据。COLMAP 4.2.1 自身的合法 Qt DLL 依赖保持完整。原生编译源和合成测试不随历史实验归档。

本候选最终接受状态由[迁移记录](../development/PIPELINE-MIGRATION.md)承载；未来架构计划不改变操作门禁或授权。
