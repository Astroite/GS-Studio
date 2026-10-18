# 2026-10-17：源码发布整理

目标为 v0.3.0-rc.1：公开说明统一为当前桌面和共享运行时，添加自有代码 Apache-2.0、第三方许可边界和托管 CI。没有在文档整理中执行重建、训练、GPU 或环境安装。

退役的顶层 gsdb 测试、Tk prototype、旧冻结/cu130 构建、无引用的激活脚本、一次性实验/RE 与原始 reference/evidence 均先复制并校验 SHA256，再逐文件移除。整理前现行文档另有原文备份。精确路径、哈希及原因见[清单](../source-retirement-20261017.json)。

后续依赖检查确认 Qt 界面无界面外生产导入方，已将该界面、三个 Qt 专属测试和旧桌面协调文档归档。混合 worker 测试仅移除退役 UI 用例；有用的非 Qt/数值测试、合成/vendor fixtures、原生编译源及被测试引用的工具保留。旧 native trainer/editor runtime 作为源码侧验证支持模块保留，算法未改，不是现行发布入口。后续批次的原文、路径和哈希见[清单](../source-retirement-20261017-002.json)。环境、SDK、缓存和原片不在归档删除范围。

最终候选版接受仍由任务负责人绑定实际结果，见[迁移记录](../../development/PIPELINE-MIGRATION.md)。
