# GS Studio 维护入口

操作 Capture、Run、重建或训练前，阅读[当前工作手册](docs/user/CURRENT-WORKFLOW.md)。开发与检查见[维护说明](docs/development/MAINTENANCE.md)和[贡献指南](CONTRIBUTING.md)；[项目方向](docs/architecture/GS-STUDIO-PROJECT-ROADMAP.md)及[工程方向](docs/architecture/GS-STUDIO-ENGINEERING-ROADMAP.md)不取代当前操作手册。

- 使用原生 Windows。产品入口为 Tauri/React 桌面和同一流水线的 CLI；源码 session 只配置当前开发进程，不修改全局环境。
- 预处理、训练与编辑使用同一锁定 Torch 2.8/cu129 环境的独立 worker 进程。不要混合不同环境的 site-packages，也不在发布包启动时下载权重或 JIT 编译。
- 保留已有未提交改动、运行环境、SDK、密钥、缓存和原始素材。项目数据与原始分析证据不进入公开源码；公开文档不记录个人机器路径或任务特定授权。
- 只有一条流水线和一套当前 Run 配置；不增加旧 schema Run 或检查点兼容分支。
- Run 清单、输入哈希、阶段证据、分段 QA 和训练结果共同决定状态。不得将失败手改为成功或绕过完整性校验。
- 分段可训练与全路线覆盖独立。保留产品审核流程；历史例外、测试数据或固定场景队列不构成新任务授权。图像发送到外部服务需要明确授权。
- 默认每秒一个候选、一张初选；局部 2 fps 补抽最多一次。运行数据清理必须使用 retention 白名单、消费者校验和 Run 锁，不删除原片，不凭退出码清理失败阶段。
- quality_v2 与 observed_v5 数值基线冻结，observed_v6 为实验性。保留当前数值测试、合成夹具和原生编译源；不要通过删除有效测试消除依赖。
- COLMAP 4.2.1 及其合法链接的 Qt DLL 保持完整；退役 GS Studio PySide/PyQt 界面不属于发布运行时。
- 文档整理只执行静态及适用的 CPU 检查，不自动运行重建、训练或 GPU 检查。报告真实失败和跳过；源码 CI 不能替代最终目录包验收。
- 归档先复制、核验 SHA256 和路径边界，再逐文件移除；在[归档索引](docs/archive/README.md)记录原路径、哈希及理由，不从历史归档提取当前默认命令。
