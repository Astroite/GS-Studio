# GS Studio 项目方向

更新：2026-10-17。产品是本地 Windows Gaussian Splatting 工作台，统一 Capture/Run/QA/训练与模型编辑。当前桌面为 Tauri + React + Spark，命令以[工作手册](../user/CURRENT-WORKFLOW.md)为准。

v0.3.0-rc.1 聚焦公开源码、明确第三方许可、公共包外接 Insta360 SDK，以及单共享 Torch2.8/cu129 运行时。它不是重新设计数值配方或兼容旧 Run 的版本。最终候选验收见[迁移记录](../development/PIPELINE-MIGRATION.md)。

后续方向包括改善全景质量与覆盖解释、扩展 GPU 架构验证、补齐更多真实采集设备和桌面交互证据。它们尚未验收，不改变现行默认值，也不授权自动训练或上传用户图片。

早期 Qt、原型和下游工具规划已被当前产品主线取代，见[历史摘要](../archive/2026-10-08-design/README.md)。工程边界见[工程方向](GS-STUDIO-ENGINEERING-ROADMAP.md)。
