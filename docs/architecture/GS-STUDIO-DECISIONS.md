# 当前架构决策

更新：2026-10-17。这里是现行决策摘要，早期决策原文已保存在外部归档，见[索引](../archive/README.md)。

1. 桌面使用 Tauri + React + Spark；Python application 统一 CLI/桌面业务，不建立第二套 Run 流水线。
2. 一个锁定 Torch2.8/cu129 环境供多个独立进程使用。Rust 管理进程树，JSONL stdout 只传协议，日志走 stderr。
3. Run、数据包、配方、检查点与模型均以完整身份校验。旧 Run/检查点不兼容，失败不得手改成功。
4. quality_v2 与 observed_v5 的数值基线冻结，observed_v6 为未晋升实验。输入预设不等于数值配方。
5. 自有代码采用 Apache-2.0；第三方源码/二进制保留各自授权。公开包外接 Insta360 SDK，RealityScan 单独安装。
6. 原始素材、运行时、SDK、缓存、个人配置和原始分析证据不进公开源码。归档先复制核验哈希，运行数据清理仍使用独立 retention 保护。
7. CI 检查源码和可行 CPU/前端/Rust 行为；GPU 和产品接受结果必须绑定最终制品，历史接受不自动继承。

实际用法与限制见[工作手册](../user/CURRENT-WORKFLOW.md)和[迁移记录](../development/PIPELINE-MIGRATION.md)。
