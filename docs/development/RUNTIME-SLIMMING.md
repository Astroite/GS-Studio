# 运行时瘦身历史记录

记录范围：2026-10-17 的本地运行时合并、组装与验收，摘要整理于 2026-10-18。GS-Studio 的 v0.3.0-rc.1 是仅源码预览基线，不附 Windows 二进制文件。下面的测试和包身份属于既有制品，不是新 Git 快照重建或重新执行的结果。构建步骤见 [Windows 发布说明](WINDOWS-RELEASE.md)，整合背景见[迁移记录](PIPELINE-MIGRATION.md)；原始日志、私有素材、机器路径和安装来源资料保留在仓库外。

## 单一环境与可复现边界

预处理、训练和编辑各用独立 worker 进程，共用锁定的 Python 3.10.22、Torch 2.8.0+cu129 / torchvision 0.23.0+cu129 环境。`python`/`preprocess_python` 必须解析为同一解释器；不得拼接不同环境的 site-packages。quality_v2 与 observed_v5 数学冻结，observed_v6 仍实验性，只有一套当前 Run 配置，不引入旧 Run/checkpoint 兼容分支。

[锁说明](../../scripts/runtime/README.txt)与以下源码共同描述复现输入：

- [core-constraints.txt](../../scripts/runtime/core-constraints.txt)：原 44-distribution 核心精确版本。
- [pipeline-additions.lock.txt](../../scripts/runtime/pipeline-additions.lock.txt)与 [pipeline-wheels.json](../../scripts/runtime/pipeline-wheels.json)：8 个额外 wheel 的精确版本、文件身份和 SHA256，不重装原生核心。
- [unified-requirements.lock.txt](../../scripts/runtime/unified-requirements.lock.txt)：合并的 52-distribution 开发环境；分为 [41 项运行时](../../scripts/runtime/runtime-requirements.lock.txt)和 [11 项构建/测试](../../scripts/runtime/build-test-requirements.lock.txt)。

[provision_runtime.py](../../scripts/provision_runtime.py)的 `verify` 只读核验解释器/安装元数据；`plan`/`provision` 要求明确的冻结核心、离线 wheelhouse 及不存在的新输出目录。`provision` 复制核心、核验复制的 DLL/PYD 哈希，再用 `--no-index --no-deps --require-hashes` 安装这 8 个 wheel，核验全部 52 个精确版本并执行 `pip check`。失败输出保留供检查，原环境不修改。它不下载权重、不导入 GPU 模块，也不重建私有原生 wheel；版本锁不能替代这些制品的构建与来源证据。

目录包通过[裁剪策略](../../desktop/scripts/runtime-prune-policy.json)移除构建/测试/安装器载荷，保留需要的运行模块、distribution metadata 和许可证。metadata 中仍能看到开发 distribution，不表示 `pytest`、`pip` 或编译工具仍可运行。保留 `setuptools`/`tomli` 与 `ninja` Python wrapper/metadata 满足冻结导入契约，不在启动时 JIT 编译或下载。

原双环境与旧包保留为受保护的回滚依据，不作为新包的隐式回退路径，不混用其模块。staging、审计副本和回滚目录均不计入最终安装体积；历史本地激活有独立证据，文档整理不执行切换。

## 体积口径与来源

载荷数来自各包的 `package-inventory.json.files`，不包含清单文件自身。GiB = bytes / 2³⁰；表内值四舍五入至三位。

| 制品 | 载荷文件数 | 载荷字节 | GiB |
| --- | ---: | ---: | ---: |
| 原始包 | 65,701 | 20,169,853,443 | 18.785 |
| public 候选包 | 9,133 | 6,495,443,306 | 6.049 |
| local-final | 9,298 | 8,449,816,529 | 7.870 |

public 根目录含清单的实际合计为 **9,134 文件、6,497,272,221 字节**；两个压缩卷合计 **2,738,873,151 字节（2.551 GiB）**，归档测试与解压通过。压缩卷只是本地验证制品，未上传为预发布附件。

public 清单 SHA256 为 `ddb6a69a8cfffa7c25e0be1fbfdc148797ec6b2433a4226d2d957e56abd24494`；local-final 清单 SHA256 为 `393be8a260166b40ffdbae148dde36f4e104bc71d9c979a88ddb6323d4489285`。local-final 逐文件完整性和 public 搬移解压后的逐文件完整性均已核验。脱敏数值亦见 [JSON 汇总](runtime-slimming-results.json)。

包内 `package-inputs.json` 记录组件、锁与裁剪策略的身份；`package-inventory.json` 记录保留载荷的相对路径、大小、SHA256；`package-exclusions.json` 记录排除理由、大小和哈希。同级 `.audit` 的输入、排除、`package-transformations.json` 及 `originals` 保留原始来源和转换证据，其中私有安装路径不进入公开源码。完整来源/大小/SHA 报告依此复核，不用磁盘上所有备份的总和代表安装包。

裁剪排除重复环境、退役 GS Studio PySide/PyQt、editable hooks、缓存、测试/示例/文档、构建头文件和静态/调试/编译产物。原生编译源、当前数值测试及合成 fixture 继续保留在源码中。运行所需 DLL/PYD 不按通用后缀删除；COLMAP 4.2.1 及其合法链接的 Qt DLL 保持完整。public 配置另排除商业 Insta360 SDK、其模型与依赖 SDK 的 helper 二进制；local profile 可含本机获许可的 SDK，但不据此推定公开再分发权。RealityScan 单独安装。

## 历史检查结果与制品边界

以下汇总既有日志与检查结果，精确数值和日志 SHA256 保留在[脱敏 JSON](runtime-slimming-results.json)。证据名称指仓库外制品，不链接私有路径。文档整理没有重新运行 GPU、训练、重建或安装。

| 证据 | 结果与边界 |
| --- | --- |
| `final-runtime-tests.log` / XML | 39 passed、1 deselected |
| `final-frontend-tests.log` | 34 passed |
| `final-rust-tests.log` | 7 passed，包括真实目录包 worker bootstrap |
| `engine-all.log` / XML | 354 passed、48 skipped；跳过为显式启用 GPU、旧 PTX 或平台条件 |
| vendor 独立测试 | 35 passed |
| `final-pipeline-all.log` / XML | 最终全套 **269 passed，0 failed / skipped**；较早失败及受影响复测计数在 JSON 中单列，不与最终结果相加 |
| `mask-comparison-unified.json` | 21 个真实样本；最低 IoU **0.9945465644712745**，无基线高置信目标遗漏，原审核遮罩未改写 |
| `gpu-unified/validation.json` | 合成 GPU 全部已测模式的完整检查点、控制、编辑和 PLY/SPZ 通过 |
| `package-local-002/result.json` | 实际裁剪 local 候选包 CPU/GPU 核心、保存暂停/恢复、遮罩和 probe 通过；封闭运行时，无 JIT/下载，受保护哈希不变。对象是 local 候选包，不自动覆盖 local-final 重组装 |
| `public-smoke-002/result.json` | public 候选包独立运行、可选工具缺席与可搬移默认配置通过；不替代搬移后检查 |
| `size-and-relocation.json` | public 压缩包在中文及空格路径重新解压后，完整文件列表、字节数及逐文件 SHA256 通过 |
| `public-relocated-002/result.json` | 搬移后的 public 包独立运行时、Torch/torchvision 导入、冻结引擎身份、可选工具缺席及 JIT/下载阻断通过；使用修正 UTF-8 参数后的测试，包内容未改 |
| `gui-relocated-001` / `gui-active-001` | 中文及空格路径下的 public 包和正式入口下的 local 包均通过原生窗口、随包 worker 启动与终止；不是视觉交互 QA |
| `gui-local-001` | local-final 原生窗口、目录包 worker 启动与终止通过；不是视觉交互 QA |
| `local-final-integrity.json` | 9,298 载荷文件、8,449,816,529 字节逐文件完整性通过，清单哈希如上；记录候选包到最终包 7 个变更路径。完整性通过不扩大为重跑候选包 GPU/产品验收 |
| `activation.json` | 历史本地激活通过，激活清单匹配 local-final；旧包回滚、原始素材、项目、原开发环境及 SDK/密钥目录保留，GPU 锁与无活动任务已核对；不表示公开发布 |
| `sdk-stitch-final/result.json` | 真实 X6 两帧，动态拼接/FlowState、去重本地 SDK 通过，源哈希不变；不扩大到所有相机或完整视频验收 |

### 固定场景 120 步短回归

South Building 固定 112 个训练视图 / 16 个留出相机，seed 42、quality_v2、120 步。计时口径为完整训练调用墙钟秒数，每臂一次；源与受保护输入的身份保持不变。

| 臂 / 外部证据 | 训练调用秒 | 留出 PSNR dB | 留出 SSIM |
| --- | ---: | ---: | ---: |
| baseline / `real-baseline/run.json` | 20.1189771 | 15.116953790187836 | 0.3890410605818033 |
| unified / `real-unified/run.json` | 18.7149577 | 15.116954326629639 | 0.3890410494059324 |
| pruned / `real-pruned/run.json` | 19.506149 | 15.116954863071442 | 0.38904108852148056 |

这仅支持该固定输入的短程一致性判断，**不是 10k 步质量验收，也不是稳健速度基准**，不据单次时间计算一般性加速结论。更早的三对 10k 步 South Building 配对回归保留在[基准汇总](BENCHMARKS.md)，两者范围独立。

## 后续构建与分发边界

既有 public 搬移检查以 `public-relocated-002/result.json`、`gui-relocated-001/result.json` 和完整解压哈希核验为准。该项没有重跑搬移后的 South Building GPU 训练，也不扩大为完整产品视觉验收。后续构建必须重新绑定实际源快照、锁和最终包身份，再按变更范围安排检查。

[第三方分发清单](../distribution/third-party-sources.json)状态仍为 `evidence-gathered-binary-release-blocked`。独立分发审计报告保留在仓库外；源码预览不附二进制。FFmpeg 对应源码/静态依赖、COLMAP 与 Qt 等依赖许可绑定、预训练权重条款、GPL `plyfile` 分发条件、CUDA/cuDNN 协议及 Python 原生依赖证据仍须闭合。自有代码采用 Apache-2.0 不改变第三方条款，哈希和功能通过不授予再分发权。

全景代表场景质量偏低，SPZ 有损且可能裁切属性；真实 GPU 验证仅覆盖架构 8.9。旧 Run/checkpoint 与 `.psht` 不支持，分段可训练不等于全路线覆盖。原生窗口检查不等于完整桌面视觉/交互验收，受控故障不等于真实断电/磁盘耗尽。仅源码范围就绪、运行证据、产品质量与二进制分发条件须分别表述。
