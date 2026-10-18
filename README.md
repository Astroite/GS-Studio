# GS Studio

Windows 本地 3D Gaussian Splatting 工作台。Tauri + React 桌面使用 Three.js + Spark 浏览模型；CLI 与桌面共用 Python 流水线，预处理与训练在独立 worker 进程中使用同一套 Torch 2.8 / CUDA 12.9 运行时。

**v0.3.0-rc.1 为仅源码（source-only）预览基线，不提供 Windows 二进制附件。** [GS-Studio](https://github.com/Astroite/GS-Studio) 从审查后的源码快照建立独立 Git 历史；视频、模型、工程、运行环境及历史 LFS 对象不属于该快照。从源码开始请看[贡献指南](CONTRIBUTING.md)和[Windows 构建说明](docs/development/WINDOWS-RELEASE.md)，后续版本见[发布页](https://github.com/Astroite/GS-Studio/releases)。

支持普通照片、普通视频、全景视频/序列及外接 Insta360 SDK 的 INSV 输入，包含遮罩审核、RealityScan/COLMAP 重建、分段 QA、训练、模型编辑和 PLY/SPZ 导出。原片只读，训练与恢复核验输入和检查点身份。

## 本地构建或私有目录包的使用

1. 将 Windows x64 目录包完整解压到可写目录，启动 `GS Studio.exe`。保留相邻的 `runtime`、工具和 notices。
2. 选择项目目录，导入素材，完成遮罩审核、重建和分段 QA，再训练通过的代表分段。
3. 查看训练相机与渲染对照，编辑模型并导出到新文件。

在目录包中打开 PowerShell 可使用同一流水线的 CLI：

```powershell
& .\gsstudio.ps1 --help
& .\gsstudio.ps1 status --json
```

需要 NVIDIA 驱动及支持随包 CUDA 12.9 运行时的显卡。实际 GPU 验证目前仅覆盖计算能力 8.9。RealityScan 需单独安装；也可配置 COLMAP。public 打包配置排除 Insta360 商业 SDK 和依赖它的 helper 二进制；使用 INSV 前按[配置说明](docs/development/WINDOWS-RELEASE.md)接入自己获许可的安装。

## 范围与限制

`quality_v2` 是默认冻结配方，`observed_v5` 是另一冻结选项，`observed_v6` 仍为实验性。全景代表场景质量偏低，有天空、植被与遮挡伪影；通过流水线不等于画质或全路线覆盖通过。SPZ 是有损编码，超出表示范围的属性会裁切。旧 Run、旧检查点和 `.psht` 不支持导入或恢复。

既有目录包的 CPU/GPU、遮罩、暂停恢复和启动结果保留为[历史验收记录](docs/development/PIPELINE-MIGRATION.md)，对应体积与精确数值见[运行时瘦身记录](docs/development/RUNTIME-SLIMMING.md)。它们不代表新快照重新完成目录包或产品视觉验收。二进制分发仍受[第三方分发证据](docs/distribution/third-party-sources.json)的未闭合项目限制。

- [操作手册](docs/user/CURRENT-WORKFLOW.md) · [桌面使用](docs/user/STUDIO.md) · [文档索引](docs/README.md)
- [开发贡献](CONTRIBUTING.md) · [源码发布保护](docs/development/SOURCE-PUBLISHING.md) · [变更记录](CHANGELOG.md) · [Windows 打包](docs/development/WINDOWS-RELEASE.md)
- 自有代码采用 [Apache-2.0](LICENSE)；它不重新许可第三方组件（包括 GPL 的 `plyfile`），见[第三方说明](THIRD_PARTY_NOTICES.md)。
