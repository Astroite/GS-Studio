# 独立引擎行为契约

这是根据当前自有实现整理的协议/行为摘要，不包含专有程序反汇编、捕获或运行依赖。`engine/local_postshot` 是内部模块名；桌面与 CLI 通过相同 application 和独立 worker 使用它。

## 协议与进程

UTF-8 JSONL 每行一个请求：`{"id":"request","method":"studio.catalog","params":{}}`。响应为 result 或带 code/message/details 的 error；长任务通过 event 发送 progress/log/metric/preview。stdout 只传协议，原生日志到 stderr。核心 worker 转发 studio 操作到流水线 worker，二者使用同一解释器环境的不同进程。

目录、状态和控制保持可响应；变更操作受 Run/项目锁约束，GPU 操作共用用户级锁。Rust supervisor 管理退出和后代进程；过期 generation 控制不能影响新任务。

## 数据与坐标

输入按实际相机、图像、遮罩、稀疏点、可信 split 与哈希验证。训练包不可变，缩放产生带身份的派生包，不改源图或可信分组。颜色只用有效 TRAIN 观测计算；深度必须通过实际锚点验证，请求与有效启用分别记录。

规范世界/模型坐标为右手 RUB、Y-up。训练 `w2c` 把规范世界转为 RDF/OpenCV 相机；K 为无畸变针孔像素内参。COLMAP 溯源位姿需经实际 scene normalization，不能直接代替规范相机。内部与 3DGS PLY quaternion 为 w,x,y,z，SPZ 边界显式转换为其要求的 x,y,z,w。

## 数值与恢复

quality_v2 与 observed_v5 保留冻结数学；observed_v6 显式实验。输入预设 auto/photo/gs 与 recipe 独立。训练预算、实际开关、源/派生包、源码和扩展身份参与实验标识。缺失原生扩展不得静默换算法。

完整 checkpoint 保存模型、优化器、调度器、策略状态、RNG、采样位置、step 和数据/配置/扩展身份。保存暂停在安全步退出；热挂起保留资源；恢复校验同身份 `checkpoints/latest.pt`。失败无检查点时必须显式从零重试，不能伪装为恢复。这里不提供旧 Run/checkpoint 兼容契约。

成功要求完整预算、评估完成，以及包/引擎/检查点/模型哈希一致。逐验证相机 PSNR/SSIM 必须有限；退出码、日志文本或导出文件存在不足以证明成功。

## 编辑与格式

变换是 position、Euler XYZ 弧度 rotation 与统一 scale；crop 在原模型局部坐标、变换前执行，边界包含。删除、历史、裁剪和变换由渲染/导出一致应用，旋转同时作用于协方差及 SH。源模型与训练 checkpoint 不覆盖，导出选新路径。

Spark 是交互视口，原生渲染/选择承担精确选择与验证。SPZ 使用官方 codec 的有损格式，属性可能量化/裁切；PLY 不保证携带完整训练/查看策略，恢复仍需完整实验。

本契约由[当前测试目录](../development/STRUCTURE.md)验证适用部分。CPU fixture 不提供 GPU、质量或目录包验收结论；边界及基准口径见[基准方法](../development/BENCHMARKS.md)。
