# 第三方分发材料与发布门禁

当前为源码预览版，二进制附件继续暂缓。机器可读清单为 [third-party-sources.json](third-party-sources.json)；自身源码的 Apache-2.0 不覆盖第三方代码、模型权重或运行库。

## 已收集范围

2026-10-19 的本地证据检查覆盖 267 个资源、38 份归档、371 个候选二进制身份及 170 份需随包保留的通知。材料留在仓库外；源码仅包含来源、哈希、相对路径和验证工具。该数量表示证据完整性，不能代替最终包验收或分发条款确认。

- **Python：** 确认 3.10.22 / 20261009 的 `install_only_stripped` 变体。47 个保留的解释器/标准库原生文件与官方归档逐字节一致；7 个额外 CRT DLL 与供应商 MSVC 14.44.35211.0 Redist 一致。普通 `install_only` 变体的多数文件哈希不同，不能互换后沿用旧身份。保留独立构建配方、原生依赖通知及完整 Microsoft 条款。
- **GLM：** 补齐 gsplat 固定子模块 `33b4a621a697a305bc3a7610d290677b96beb181` 的完整源码和原始通知，官方 submodule 记录核对一致；选择通知提供的 MIT 许可。新扩展仍需绑定本轮实际构建身份。
- **CUDA/cuDNN：** 14 个 CUDA、8 个 cuDNN 候选 DLL 分别绑定完整版本协议、补充条款和通知。CUDA Attachment A 与 cuDNN runtime DLL 条款保持分开；应用分发条件仍待处理。
- **COLMAP：** 除既有 COLMAP/Qt/CGAL 材料，补齐 GMP 6.3.0、MPFR 4.2.2、SuiteSparse 7.13.0 源码；三份归档的 SHA512 与固定 vcpkg 配方一致。10 个相关 DLL 身份绑定候选源码/通知。保留 SuiteSparse 分模块许可，包括 SPQR GPL 与 CHOLMOD 混合模块条款，没有给整个工具套用单一宽松许可。
- **FFmpeg：** 固定 core commit，补齐 7/82 个提供方列名外部库的来源材料：GMP、bzip2、GnuTLS、libiconv 的匹配版本 MSYS2 source package 包含 PKGBUILD/补丁/原始源码；libass、vidstab、rubberband 绑定声明的上游版本。源码归档通过哈希校验，不等于已证明这些配置生成了现有静态二进制。

## 剩余七项阻塞

| 项目 | 仍需完成 |
| --- | --- |
| FFmpeg 静态对应源码 | 完整外部库和传递依赖源码、实际补丁、编译/安装配方；82 个列名库中的其余来源材料，以及四个未列版本库。提供方 9.0.1 Release 只有六份二进制归档；仓库树没有构建配方，公开 build-suite 分支停在 2020，不能冒充 2026 配方。 |
| COLMAP 依赖绑定 | 供应商 installed/share/status、triplet、features、构建/补丁记录，剩余依赖通知和有效许可判断。实际 ONNX Runtime 1.30 与 registry port 1.23.2 不一致，不能只按 baseline 推断。保留 Qt 替换权利及 GPL 相关义务。 |
| AlexNet 权重 | 适用于精确 checkpoint 的分发条款；代码和数据集许可不代替权重授权。 |
| Mask R-CNN 权重 | 同上，保留精确文件身份，不覆盖现有模型或已审核遮罩。 |
| plyfile / 组合程序 | 明确并满足 GPL 组合分发与对应源码条件，处理与专有运行库的关系。 |
| NVIDIA 应用分发条件 | 最终包保留完整协议/通知，确认应用分发条款一致；DLL 在许可列名中不是此项的全部要求。 |
| Python 原生资源 / CRT | 新构建候选重新核验原生身份和通知；确认 CRT 适用授权与应用分发条件。已有包的字节证据不能自动证明新候选通过。 |

清单的 `source_archive_verified` 仅表示声明来源的归档身份已固定、字节已检查；`binary_build_binding_verified: false` 明确保留尚未证明的二进制构建关系。不得把候选源码资料标成完整对应源码交付。

## 可复核命令

在仓库根目录执行，路径替换为本机选定的仓库外目录。命令不执行第三方二进制，不修改已验收的目录包。

```powershell
python scripts/release/third_party_sources.py self-test
python scripts/release/third_party_sources.py verify --source-dir ..\distribution-sources --evidence-only
python scripts/release/third_party_sources.py notices --source-dir ..\distribution-sources --output-dir ..\new-package-notices
python scripts/release/third_party_sources.py verify --source-dir ..\distribution-sources --package-root ..\final-public-package --report ..\final-distribution-check.json
```

`notices` 先核验全部输入，再创建全新目录，保留每份通知的原始字节和哈希清单。构建人员把通知目录纳入新候选包后，再运行最终校验。它不覆写旧证据、不修改已验收包，也不解除发布阻塞。

默认 `verify` 是发布门禁：开放阻塞、缺失/变更文件、新增未审原生文件、缺失通知、私有 SDK 或缺失最终包均导致失败。`--evidence-only` 只证明证据字节完整，返回成功也会保留 `release_ready: false`。保留实际未通过项，继续暂缓二进制上传。
