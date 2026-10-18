# 归档索引

公开仓库只保留简短的有日期摘要与逐文件哈希索引；原文、私人路径/授权、一次性脚本及原始基准/逆向证据保存在外部迁移档案。归档不是当前命令、兼容接口或新任务授权。

| 日期/类别 | 内容 |
| --- | --- |
| [2026-10-08 设计](2026-10-08-design/README.md) | 被当前 Tauri 产品主线取代的早期设计与冻结包记录 |
| [2026-10-16 集成](2026-10-16-integration/README.md) | 既有集成接受范围，不能替代本候选版接受 |
| [2026-10-17 源码发布整理](2026-10-17-source-release/README.md) | 本轮退役范围与保留边界 |

[source-retirement-20261017.json](source-retirement-20261017.json)逐项记录原相对路径、字节数、SHA256、归档相对路径与理由。`archive` 表示经复制/哈希核验后移除原文件；`backup_before_edit` 表示只保存整理前原文，当前路径继续使用。外部档案以 `slim-20261017-001` 标识，公开索引不含本机磁盘路径。

[source-retirement-20261017-002.json](source-retirement-20261017-002.json)记录后续 Qt 界面、Qt 专属测试和旧桌面协调文档的归档，以及公开维护/来源说明修改前的原文。该批次对应外部档案的 `followup-002`；源码中的旧 native/editor 支持模块继续保留。

[source-retirement-20261018.json](source-retirement-20261018.json)记录 16 个无当前调用方的外部 Postshot 后端、对照入口和历史手工验收脚本的退役，以及三处来源元数据／测试说明修改前的备份。源快照为经过 SHA256 核验的 `repository-before.bundle` Git bundle，公开清单记录其哈希；归档集合标识为 `fresh-repo-20261018-001`。现行自动化数值测试、原生编译源、合成夹具及有效构建辅助脚本继续保留。

当前可用的行为契约见[引擎契约](../architecture/ENGINE-CONTRACT.md)，可公开的数值及限制见[基准方法](../development/BENCHMARKS.md)。
