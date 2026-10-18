# 从公开输入构建 Windows 运行环境

`scripts/runtime/bootstrap_windows.py` 从固定公开下载和本仓库源码构建新环境，不复制已有 Python 环境或预编译的私有扩展。它用于开发构建，产品启动不会调用它。原有 `scripts/provision_runtime.py` 仍只适用于复制已验证核心、补入八个锁定依赖的离线复现，不能作为从零构建证据。

## 前置工具与隔离目录

需要原生 Windows x64、可执行脚本的 Python 3.10 或更新版本、Git、系统 curl、Visual Studio 2022 的 C++ 工具与 Windows SDK。当前构建基线是 MSVC 14.44、CUDA 12.9.0（NVCC 12.9.41）、架构 8.9；其他 GPU 架构未获得同等验证。构建脚本显式接收工具链位置，不假设 VS 的具体版本类型。

运行时解释器固定为 Python 3.10.22、Astral BUILD 20261009。Torch 2.8.0+cu129、torchvision 0.23.0+cu129、gsplat 1.5.3 和完整 52 包锁保持不变。公开来源、SHA256 与 Git commit 位于 [public-build-inputs.json](../../scripts/runtime/public-build-inputs.json)：44 个官方索引轮子、一个源包，以及 gsplat、GLM、fused-ssim、SPZ、zlib 和 zstd 的固定源码。

`python-devtools==2` 上游只有 951 字节源码包；脚本在新环境离线构建其空占位轮子，不替换成另一个项目或版本。CPython 的完整包、裁剪包和本地已安装文件可能存在签名或裁剪差异；版本、BUILD 或构建成功不证明二进制逐字节相同。

工作目录必须是仓库外的新目录。可选下载缓存与工作目录使用不同目录，缓存只读并逐文件核验哈希。不复用源码目录中的 build/dist，也不修改现有运行环境、SDK 或项目数据。失败产物保留供诊断。

## 分阶段命令

在仓库根目录的 PowerShell 中设置实际位置：

```powershell
$Workspace = Join-Path $env:TEMP 'gsstudio-source-build'
$VcVars = '<VS 2022 安装位置>\VC\Auxiliary\Build\vcvarsall.bat'
$CudaHome = '<CUDA 12.9 安装位置>'
$BuildArguments = @('--workspace', $Workspace, '--vcvars', $VcVars,
    '--vcvars-version', '14.44', '--cuda-home', $CudaHome,
    '--architecture', '8.9', '--jobs', '2')

python -I -B -X utf8 scripts/runtime/bootstrap_windows.py plan @BuildArguments
python -I -B -X utf8 scripts/runtime/bootstrap_windows.py prepare @BuildArguments
python -I -B -X utf8 scripts/runtime/bootstrap_windows.py build @BuildArguments
python -I -B -X utf8 scripts/runtime/bootstrap_windows.py verify @BuildArguments
```

`plan` 只检查路径、锁和源码摘要，不创建目录、不下载、不导入 Torch。`prepare` 是唯一联网阶段：下载公开工件、核验 SHA256、检出固定 Git commit 和 GLM。它可对相同身份的工作目录重复执行，补齐下载；内容变化或部分提取/编译失败可能需要检查保留文件后改用新目录。

已有的公开下载缓存可通过 `--download-cache <目录>` 只读复用；缓存中的错误哈希会报错，脚本不会覆盖它。`--git <git.exe>` 可显式固定 Git 路径。`build` 和 `verify` 不解析在线依赖，不加载旧 site-packages。所有原生扩展先将源码复制到各自新作业目录，再用 [build_native.py](../../engine/scripts/build_native.py) 构建。

没有 CUDA toolkit 的构建机可将参数中的 `--cuda-home` 和路径替换成 `--download-cuda`。脚本从 [NVIDIA 官方 CUDA 12.9.0 清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.0.json) 固定下载七个编译器、运行时及头文件组件，解压至工作目录，不运行系统安装器、不安装驱动、不修改系统 PATH。每个组件的许可资源单独保留。这是构建工具来源记录，不构成最终二进制分发许可结论。

脚本在子进程内配置 VS/CUDA。安装时禁用依赖解析，版本验证要求完整 52 包精确匹配；原生检查加载预编译扩展，不调用训练或运行时 JIT。成功分别写入 `prepared.json`、`built.json`、`verified.json`；失败退出非零并保留 `logs` 和各 `builds/<模块>/build.log`。成功文件的范围必须一起阅读，编译成功不会记录为 GPU 测试通过。

## 干净 Windows CI 与验收边界

[Clean Windows build](../../.github/workflows/clean-windows-build.yml) 仅手动触发，在 GitHub 托管的 `windows-2022` 新虚拟机上运行，不使用已有应用环境。它显式检查 MSVC 14.44，缺失时停止，不自动换成新编译器；通过独立下载的 CUDA 编译固定公开输入，并在禁用 GPU 的环境中检查 52 包版本和预编译模块导入。另一新虚拟机使用 Rust 1.95.0、Node 22、npm/Cargo 锁编译 Windows 桌面程序。

该工作流没有二进制、下载缓存或压缩包上传步骤。普通源码 CI 也执行 [构建脚本契约](../../tests/build_scripts/test_windows_bootstrap.py)，但不会自动下载 CUDA。

托管构建机没有可用 NVIDIA GPU，因此绿色 CI 只证明指定 commit 的公开输入构建、桌面编译和 CPU 导入，不能证明真实训练或最终目录包可用。后续必须在本地新环境完成实际前后向、固定种子训练和渲染对照、暂停恢复，再将这些二进制纳入最终包的身份和搬移验收。第三方材料、模型权重、外置 SDK、RealityScan 授权及最终运行包验收按 [Windows 发布说明](WINDOWS-RELEASE.md) 单独处理。
