# LoopLoRA：实验诊断与下一阶段执行计划

**本文档为事前制作的参考文档,并非权威, 实际执行中应当根据实际情况调整计划并对本文档进行更新**

**版本：1.0 · 日期：2026-10-29 · 新研究代号：SLOT-LOOP-v1**

**审查基线：** `UnbearableFate/LoopLora`，commit `a255e814740ff85ab3b4ec4836c88f0a8f802b2d`。本文依据该快照的实现、预注册、实验报告和部分原始指标整理；训练权重仍需执行者在原 HPC 环境核实。此后仓库新增结果应作为新证据登记，不能默认为已包含在本文中。

**文档性质：** 可交付给研究负责人、实现者和实验执行者的工作规格。历史结果与新实验建议分别标注。新增 API、配置、CLI、预算和阈值属于待实现、待锁定的执行约定；本文不表示这些实验已经完成，也不表示新增脚本已经存在。

**核心建议：** 先验证“同一组学习得到的 Key/Value，是否因为矩阵存储与读取方式而损失信息”；再验证训练后的多次读取是否产生可组合的状态转移。优先推进**显式 KV 槽位 + 共享低维循环读取器**，并保留矩阵记忆作为严格对照。只有完成读取与转移验证后，才扩展动态停止、固定字节预算和真实任务。

### 执行记录（Orchestrator 持续更新）

- **2026-10-29 17:50 JST** 用户要求按本文继续改进，本文提交为 4e6778a。
  - 审查基线 a255e81 之后 master 只新增了文档。LSQT 的 U 对照组（作业 3725694）仍在运行，结果出来后作为新证据登记。
  - 分派：任务书 `docs/tasks/WP-L13_slot_loop.md`（000d045），由新 worker sn-slot 在 `LoopLora-wt/slot`（分支 wp/slot）执行。
  - 范围调整：S0 和 S1A 合并为一份预注册 `docs/prereg/SLOT-LOOP-v1.md`，预算分别 ≤ 1.0 和 ≤ 4.5 node·h。S1B 在 S1A 之后另行登记，S2/S3 未授权。第一步只起草预注册，由 Orchestrator 审核并登记到 master 后才写代码。
  - 补充决定：
    - recency-matched SoftSlot（§6.2）在 S0 中预先登记为描述性条件；
    - 正式计算前先对新代码做一次静态评审；
    - S1A 开封确认集之前，由独立评审重跑分析脚本（§14.2）。
- **2026-10-29 18:0x JST** 用户授权 S2、S3 及本计划全部任务（“S2和S3以及新计划的所有任务都一起授权，按计划推进”）。
  - 各阶段仍按本文的进入条件依次推进（§0.3、§6.5、§7.4、§8.3、§10.5、§14.3），不越过未通过的门。
  - S1B、S2、S3 各写一份增补预注册，各有独立数据命名空间、判定规则和预算表，由 Orchestrator 在看到该阶段数据前登记。S1B 预算沿用本文 11.5 node·h；S2、S3 的预算按实测步时在增补中给出，由 Orchestrator 批准。
  - S0 代码合并后，S2/S3 的准备工作（标签与读取器代码、测量工具）可交给第二个 worker 并行。
- **2026-10-29 18:10 JST** 预注册 `docs/prereg/SLOT-LOOP-v1.md` 登记于 master 5f04a5c，blake2b-128 `56e8c611c01e0ef9b7ae32db97b0e334`（sn-slot 草稿 e7d96df，加上 Orchestrator 登记决定 1–9）。
  - 登记前复核：数据表中全部基础种子和布局摘要 `b7f93c1c…` 均用标准库 blake2b 重新推导，结果一致。
  - S1B 的进入条件按 §0.3 理解：S1A 在自己的 dev 上有效、确认集隔离即可，不要求 S1A 判定为 ADVANCE。S1A 和 §7.4 消融都为阴性时，停止扩展槽位（§7.4、§14.3），S1B 只比较 Matrix-L1 和 Matrix-L4。
  - Matrix 与 Native 必须逐位一致，否则在读取任何校准或 dev 数据之前停止。
  - sn-slot 开始第 2 步：实现、计算节点测试、静态评审 STOP。
- **2026-10-29 19:34 JST** LSQT 的 U 对照组（作业 3725694）完成，合并于 b60f144。LT-U 36.0 %；RG001 比 R-r128 高出的 +12.6 pp 按点估计拆为单位查询 +2.5 pp（标准误 2.1）和存储 +10.1 pp（标准误 2.2），单 seed，只做描述。
  - 这条证据的意思：在线性键、从头训练的设置下，按经过 U 的拆分，增益的点估计大部分落在存储一项。没有训练“ridge 存储 + 学习查询”的臂，所以交互项分不开。
  - 它不改变 S0 要回答的问题（换读取算子能否更好地读出 S-RE 的现有表示），S0 照常进行。
- **2026-10-29 21:44 JST** 用户放宽资源规则：“不必在意文档中描述的预算, 保持每单个run在16 node * h即可”。
  - 本文和各预注册里写的预算不再约束作业或阶段，包括 §14.1，以及 SLOT-LOOP-v1 的 S0 ≤ 1.0、S1A ≤ 4.5 node·h 和“会超出上限的作业须经 Orchestrator 同意”。S1A 的 2 h 扩展也不再单独审批，release 审核照旧。
  - 每个作业 ≤ 16 node·h（节点数 × walltime），由 `scripts/submit.py` 检查。walltime 仍按实测 × 1.5 申请。
  - SLOT-LOOP-v1 预注册文件不改，因为代码绑定了它的 blake2b。本条记录取代其预算一节。
  - 当时 S0 已用 0.701 node·h，其中两轮评审修复后的重测约 0.2。
- **2026-10-29 22:20 JST** S0 完成，判定 R2（wp/slot 3475ecc，合并为 master 2f7abbc）。
  - 评审：一轮静态评审，两轮交叉评审（rv-code 和 Opus 子代理）。之后按用户要求（“不要搞这么多次review,核心代码没问题就行”）不再增加评审轮次。剩下两项改为 S1A 前的跟进，不改 `src/`：release 摘要绑定、S1A runbook。
  - preflight（作业 3728339）9 项全部通过。S-RE 原路径逐位复现：723/1000，NLL 1.642127275466919。Matrix 与 Native 的 logits 和梯度逐位一致。lock 为 2f22aba，`src/` 摘要 `fe548da4c06d77b7f4a8ae65349650f9`。
  - S0 dev（1,000 题，全部有效问题位置），区间均为描述性：

    | 读取方式 | EM | 相对 Matrix | 读取误差 |
    | --- | ---: | ---: | ---: |
    | Matrix | 72.0 % | 基准 | 0.184 |
    | SoftSlot（g=1，τ=0.1） | 74.5 % | +2.5 pp | 0.092 |
    | HardSlot | 78.1 % | +6.1 pp | 0.137 |

    HardSlot 的 NLL 更差。只读最终位置时，干净读取为 92.6 %。D128-3 复核的方向相同。
  - 按登记规则，判定为 R2：S1A 比较 SoftSlot-L1 与 Matrix-L1，两者都用 g=1，SoftSlot 用 τ=0.1；HardSlot 只作诊断。S1A 由 wk-slot 进行。
  - S0 共用 0.778 node·h，含测试和重试。
- **2026-10-29 22:45 JST** S1B 增补预注册 `docs/prereg/SLOT-LOOP-v1-S1B.md` 登记，blake2b-128 `6f2e776f2c6cfde131380fb343854784`。底稿是 wk-loop 的 73250eb，另加登记决定 1–6；其中 11 个种子和布局摘要已用标准库复算。
  - 分支由 S1A 的判定机械决定：
    - ADVANCE：M1、M4、S1、S4；
    - NOT ADVANCED 且消融未标记：M1、M4；
    - 消融被标记：四臂都带 `A_query`。
  - 任务按 §8.1 用 T2 chain-end：E12，两条链长 8 和 4，关系 0，训练 hop 1–3，确认 hop 1–8。主对比用 hop 2–3。
  - 冻结的 S-RE writer 是在 T1 上训练的，迁移到 T2 未经验证。由 preflight 把关；另设一个停止条件：校准后的 Matrix-L1 若 hop-1 EM 低于 10 %，就在 pilot 前停止。
  - 确认循环增益的条件：L4−L1 的 Δ ≥ 3 pp，校正后的区间下界 > 0，每个 seed 的差都 > 0，hop-1 下降不超过 2 pp。四臂分支有三个对比，用 Bonferroni 98⅓ % 区间。
  - wk-loop 是新开的 codex worker，在 wp/loop 上并行实现。S1A 完成前，wp/loop 与 wp/slot 不互相合并。

---

## 0. 执行摘要与第一批交付物

### 0.1 需要回答的三个问题

1. **存储问题：** 当前单跳上限是否主要受到线性叠加、寻址和读取尺度的共同限制？
2. **循环问题：** 在可靠读取的条件下，共享循环是否学会“读出当前事实 → 形成下一跳地址”，而不只是增加一个低维非线性网络的深度？
3. **实用问题：** 在明确计算读次数、状态字节和端到端延迟后，循环方案是否仍比同条件的单次读取有价值？

### 0.2 第一批只做三件事

| 顺序 | 任务 | 必须产生的文件 | 完成条件 |
| --- | --- | --- | --- |
| 1 | 锁定代码、checkpoint、数据分片和参数冻结集合 | `manifest.lock.json`、`checkpoint_inventory.json` | 源 commit 与权重摘要可追溯，旧测试保护规则保留 |
| 2 | 建立兼容 S-RE/DPFP 的读取接口，复现原路径 | `identity.json`、针对性测试报告 | 同精度、同 batch、同读取位置下，原路径 logits/预测一致 |
| 3 | 运行冻结模型的 Matrix / SoftSlot / HardSlot 对照 | `s0_metrics.json`、`s0_samples.jsonl`、`decision_s0.md` | 完成同内容、L1、全部有效问题位置的比较，并明确是否需要继续训练 |

第一批不增加循环深度，不更换模型规模，不进行全参数网格搜索。这样可以在较小代价内确定下一步应改存储、改 query，还是改跨步状态。

### 0.3 推进顺序

| 阶段 | 内容 | 主要科学问题 | 进入下一阶段的依据 |
| --- | --- | --- | --- |
| S0 | 冻结 S-RE 的读算子替换 | 原表示能否被更好地读出？ | 有诊断信号，或仍存在可训练接口不匹配的合理解释 |
| S1A | T1、L1、冻结 writer 的同起点训练 | 新算子能否胜过同起点继续训练的矩阵对照？ | 新 dev 上有效；测试确认与新阶段数据隔离 |
| S1B | T2 上 Memory × Depth 的 2×2 实验 | 多次读取是否带来独立收益？ | 独立 L1 对照与同 checkpoint 深度扫描共同支持 |
| S2 | 角色对齐、逐转移监督、NULL/终止 | 循环状态是否具备可解释、可干预的多跳语义？ | 转移正确、终态稳定，且新增信息和监督已匹配 |
| S3 | 固定字节预算、负载与实际任务 | 优势能否在资源约束下保留？ | 后端独立部署、真实计时和容量边界验证 |

本计划中的 S0–S3 是新研究阶段，**不是** SQ2 中 S-DE/S-RE 等历史 arm 的重命名。旧研究的 80% writer gate、SQ2 判定和 LSQT 判定保持原样。

---

## 1. 已有证据：哪些结论成立，哪些还不能成立

### 1.1 必须分开的实验条件

**Learned writer 主线**主要是 T1 单跳、E16、四种关系、L1；SQ2/LSQT 的主要结果为一个训练 seed、1,000 个验证 episode。**Oracle reader 主线**主要是 T2 chain-end、E12、L4、三个训练 seed，以及一个 oracle 编码表/布局。两条线的 EM 不能串成同一条性能曲线。[R01][R02][R03]

仓库中的 seen/unseen entity-name pool 为一个预定义名字集合的划分。它不能直接代表开放词表实体、任意新 tokenizer token 或真实世界知识泛化。跨 seed 的置信区间也不自动覆盖新 writer 初始化、新 oracle 编码表或不同数据生成机制。

### 1.2 主要结果账本

| 证据 | 已观察到的结果 | 本计划采用的解释 |
| --- | --- | --- |
| P1L，0.6B，oracle r128 | M1-L4 对 B4-L1，hops 2–3：+21.55 pp，区间 [20.38, 22.76]；M1 的 h1–4 为 47.9 / 35.4 / 18.5 / 7.2% | 共享多次读取在此协议下有效；绝对多跳能力仍低，尚未证明 learned writer 下也成立 |
| P5L，1.7B，oracle r128 | 相同主对比 +25.78 pp，[24.53, 27.03]；h1–4 为 58.7 / 41.5 / 22.5 / 6.2% | 有跨模型规模的同协议支持；深度与长链稳定性仍是问题 |
| Context-retained B1/B2 | r32、hops 1–3 为 99.65–99.93% | 基础模型在保留上下文时具备解题能力；访问原文协议不同，不能作为公平的 memory-only 胜负比较 |
| B8W | warm-start untied − shared 为 +0.90 pp，[0.09, 1.71]；shared 继续训练本身增加 +5.09 pp | “共享是必要条件”没有得到支持；原 B8 失败含优化/初始化因素；既有 checkpoint 未必充分收敛 |
| DPFP | D128-3 为 60.8%；E8 另行训练为 73.0% | 表征与负载影响显著；不同 E 的结果不能当同一个模型直接外推 |
| SQ2 | S-RE 为 72.3%，优于同起点继续训练的 S-DE 65.4% | 存储规则确实有影响；尚未解决全部读取/解码问题 |
| LSQT | 最佳 LT-RG001 为 46.1%，读误差 0.512 | 从头训练线性键 LSQ 没有复现 DPFP+ridge 水平；处于其注册的中间区间，不能写成整条线已被证伪 |
| RFX | 旧 R-r128：原读 33.8%，ridge+learned-query 38.8%，delta+true-key 39.9%，ridge+true-key 84.2%，clean-value 84.4% | 在这个 checkpoint 上，存储与 query 存在共同瓶颈；oracle 干预是诊断，不是可部署算法 |
| LSR | 长跳可改善，但 h2 降低约 10.8/12.3 pp；平均读 5.60/5.66 次，上层网络经过 6.60/6.66 次 | 停止策略未形成准确率与成本的全面改善；不能省略短跳损失与上层多次前向成本 |
| TSM | 已尝试终态 stay loss 和 gate，未通过全部注册条件 | “加停止门/重复终点监督”已经试过；新方案应验证终止前的逐步转移 |

来源：[R01]–[R10]。上述数值是原报告的条件性结果；它们不是本计划新增实验的结果。

### 1.3 SQ2 是当前最有价值的起点，但不能过度解释

| arm | EM | 归一化读误差 | clean-value head | read-value head | key top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S-DE：delta | 65.4% | 0.359 | 93.7% | 54.6% | 85.9% |
| S-DC：delta + coefficient loss | 63.0% | 0.353 | 92.9% | 50.4% | 86.9% |
| S-RE：DPFP ridge | 72.3% | 0.196 | 93.6% | 66.2% | 87.1% |
| S-RC：DPFP ridge + coefficient loss | 69.5% | 0.199 | 92.9% | 66.4% | 86.8% |

S-RE 的“干净值可解码”与“实际读出值可解码”之间仍相差 27.4 pp，说明进一步研究读取过程有依据。但这两个 head 指标来自训练期辅助头，不能直接解释为最终 LM 答案准确率的可加分解，也不是最终性能的严格上界。[R03]

SQ2 的主注册对比是 S-RC 对 S-DE，提升 +4.1 pp、误差比 0.554，未达到其 +10 pp 和误差减半要求。S-RE 是事后表中最好的单元，不能据此改写原研究 verdict。其 step-0 直接改为 ridge 已达 68.7%，随后到 72.3%；这提示读算子本身值得优先诊断。[R03]

### 1.4 暂时不支持的归因

- **“循环本身没有用”**：oracle-reader 结果不支持这个总括结论。
- **“只要增加循环数就会学会推理”**：绝对长跳准确率、短跳退化和 TSM 结果不支持。
- **“共享权重的表达能力必然优于不共享”**：B8W 的同函数初始化与继续训练结果不支持。
- **“Key top-1 已高，所以记忆读出不是问题”**：SQ2/RFX 和第 2 节推导不支持。
- **“再做 ridge / coefficient loss / 简单 tying 就是新方案”**：仓库已有相关实验，应先解释差异。
- **“没有达到旧 80% gate，所以矩阵记忆或循环在理论上不可行”**：门槛是项目推进规则，不是理论不可能性证明。

---

## 2. 对未达预期原因的结构性诊断

### 2.1 当前实际模型已经不同于最初的双 BA 公式

以列向量记号写，仓库核心读取近似为：

\[
u=Ax,\quad z_0=u,\qquad
z_{s+1}=u+\rho\,\phi\!\left(Rz_s+UM\psi(Qz_s)\right),
\]

\[
y=Wx+\alpha Bz_L.
\]

具体 arm 的 feature map、归一化和 writer/storage 另由实现决定。一次问题读取的内循环中，M 固定；基础 Wx 不因低维循环而反复计算。现实现采用有效的残差初始化，不能用最初双 BA 在 B=0 时可能出现的梯度问题来解释所有现有失败。[R16][R17]

冻结 backbone 的参数不意味着可以在训练时把相关整个前向放入 `no_grad()`：读取分支仍需要通过上层冻结网络获得损失梯度。

### 2.2 地址正确与值正确不是同一件事

考虑两个单位 key，\(k_1^Tk_2=c\)，以及正交的 \(v_1,v_2\)。用无遗忘、单位更新率的 delta 写入：

\[
M_1=v_1k_1^T,\quad
M_2=M_1+(v_2-M_1k_2)k_2^T.
\]

即使 query 恰好为正确键 \(q=k_1\)，仍有：

\[
M_2k_1=(1-c^2)v_1+cv_2.
\]

当 c=0.8 时，正确 key 的点积是 1，错误 key 的点积只有 0.8，但读出是 \(0.36v_1+0.8v_2\)。因此 key top-1 可以正确，输出却被错误值主导。这是对干扰机制的数学示例，并非声称全部实际样本都遵循此二键条件。

同样不能从这个例子推导“任何矩阵都不可能存好”：当键集合具有相应线性独立性时，精确插值的线性映射可以存在。实际限制还涉及特征几何、条件数、正则化、训练目标和 query 泛化。

### 2.3 增加循环深度，不自动等于增加图上的一跳

只有在激活和 query feature map 均取恒等的简化情形下，令

\[
T=\rho(R+UMQ),\quad z_{s+1}=u+Tz_s,
\]

就有

\[
z_L=\sum_{j=0}^{L}T^ju.
\]

它说明重复读取可以形成关于记忆的高阶变换，但没有保证第 s 步对应第 s 个桥实体。当前输入 u 持续注入也可能帮助保留问题，或使状态始终混有起点；哪一种占主导需要通过轨迹和干预验证，不能只靠公式判断。

### 2.4 多个接口错位会掩盖改进

现有辅助寻址/读出损失主要作用于最终问题位置，但实际模型可在全部有效问题位置注入 memory branch。仅修改最后一个位置未必能修复其他位置传播到上层的表示。P1L-C 中 final-only 推理消融出现明显退化，说明读取位置是需要控制的因素；这不是“单位置训练必然失败”的证据。[R11]

另外，head、tail、当前实体、下一跳 query 未必处于可直接组合的表示空间。CTC 的显式解析器使用了特权结构，不能当作“纯文本 learned writer 已学会角色组合”的证明。[R12]

### 2.5 训练和监督问题应保持可检验

SQ2 中梯度裁剪在所有记录步生效，且 coefficient loss 没有带来预期收益；这值得记录各损失的梯度范数与方向。但不能仅凭 clip=1、原范数约 8–10，就断言 AdamW 的有效学习率严格下降到原来的十分之一。[R03]

新的逐转移损失必须监督终止前的桥实体/下一地址；不能再把已有 TSM 的终点 stay loss 改名为路径学习。进一步训练需要与同起点、同优化器策略的矩阵对照比较，不能只与历史最佳数值比较。[R05][R10]

---

## 3. 推荐方案与研究假设

### 3.1 第一优先：显式 KV 槽位 + 原低维读取器

第一阶段保持 writer 产生的 key/value 不变，替换持久状态与 read：

\[
\mathcal S=\{(k_i,v_i)\}_{i=1}^{E},\qquad
a_i(q)=\operatorname{softmax}_i(k_i^Tq/\tau),
\]

\[
m(q)=g\sum_i a_i(q)v_i,
\qquad
z_{s+1}=u+\rho\phi(Rz_s+Um(q_s)).
\]

\(q_s\) 必须使用原 adapter 实际 query feature map，不能擅自换成未扩展的 128 维 query。对于 S-RE，\(d_k=768,d_v=128\)。\(g>0\) 为统一读出增益，\(\tau>0\) 为温度；第一轮先固定，校准版本另报。

这个更改分离了两个操作：事实的可寻址保存，以及查询时的选择。代价是状态随容量 S 增长，且失去矩阵记忆的固定大小优势。因此优先在当前 E16 条件验证，并在 S3 才研究更大负载和容量溢出。

**研究假设 H1：** 在相同学习表示、相同支持事实、相同读取位置与同等训练安排下，显式选择能减少实际读出误差，改善完整词表答案 EM。

### 3.2 第二优先：为循环建立明确的转移语义

在 H1 的接口验证后，研究：

\[
c=\operatorname{QuestionEnc}(x_q),\qquad
p_0=\operatorname{AddressEnc}(x_q),
\]

\[
q_s=\Phi(F_{\rm addr}(p_s,c)),\quad
m_s=\operatorname{Read}(\mathcal S,q_s),
\]

\[
p_{s+1}=F_{\rm step}(p_s,m_s,c),\qquad
a_{s+1}=F_{\rm answer}(a_s,m_s,c).
\]

c 保留问题目标；p 表示下一次寻址所需状态；a 表示答案相关状态。这些角色应逐项引入，而不是一次把全部结构、损失、深度、位置都改掉。

**研究假设 H2：** 逐转移的一致性训练可以提高模型在自由 rollout 下的正确地址率，并使额外循环的收益随真实路径长度变化。

### 3.3 第三优先：有语义的 NULL 与终态

NULL 只表示“当前允许读取的记忆里，不存在指定实体、指定关系的有效出边”。闭世界完整图中，它可以支持 chain-end 停止；缺失事实、容量淘汰或开放世界任务中，它应支持 UNKNOWN，而不是自动把当前实体当答案。

**研究假设 H3：** 将读取失败/无出边与状态保持显式建模，可减少完成后的状态漂移，并改善短跳与长跳的共同表现。是否带来延迟改善必须实际计时，不能只按减少的低维循环次数估计。

### 3.4 创新主张的边界

显式 memory、多跳 attention、共享读取和路径监督均有充分先例，包括 End-To-End Memory Networks、Key-Value Memory Networks 和 Neural Logic Programming。[L01][L02][L03] 将它们放入低秩残差分支，本身不足以证明新的基础机制。

可争取的研究贡献是：**针对冻结 LLM 的小状态适配，给出地址、存储、跨步组合三类瓶颈的可验证分解，并提出在相同信息和资源约束下有效的接口与训练方案。** 论文主张应由同内容替换、条件匹配训练、因果中间态干预和成本测量共同支持。

最终投稿前，应另行核对 RecLoRA、Generative Adapter、δ-mem、MaLoRA/MaRA、QueryableLoRA 等工作的准确版本和最接近设置。不能以“动态”“循环”“低秩记忆”这几个词作为新颖性判据。此执行计划不预先宣布首创。

---

## 4. 资产核查：必须使用真实 checkpoint 和真实入口

### 4.1 主 checkpoint：S-RE

| 项目 | 已知值 |
| --- | --- |
| HPC 目录 | `/work/xg24i002/x10041/LoopLora/runs/p1l/sq2/arms/S-RE/` |
| 推理导出 | `writer.pt` |
| 续训资产 | `checkpoint.pt`，含优化器和训练期辅助头 |
| 推理 loader | `looplora.llm.sq2.load_inference(path)` |
| SQ2 续训步数 | 6,000 |
| 配置哈希 | `1bfb72b7d87bc0aff743c1e018e8680e` |
| 原科学运行 commit | `3d543b430a2533df0057647081b07d874de893db` |
| 原验证复现参考 | 723/1,000；NLL 1.642127275466919 |

核查来源：[R03][R18]。S-RE 文件的 blake2b-128 应由执行者在 HPC 读取真实文件后记录。本文没有访问该权重文件，不能替它填写一个未验证的摘要。

### 4.2 辅助 checkpoint：D128-3

目录为 `/work/xg24i002/x10041/LoopLora/runs/p1l/dpfp/arms/D128-3/`，loader 为 `looplora.llm.dpfp_run.load_inference(path)`。已登记推理文件摘要为 `31d98c01c604c05a54bc20e4b4c4f436`，训练 checkpoint 摘要为 `b6acff8b772a692311e630293add553f`；均为 blake2b-128。原步数 22,370，配置哈希 `2207a847e4967fb8e60d1663237fd979`。[R19]

S-RE 是主要起点，D128-3 仅用于验证诊断是否依赖某个 ridge-trained checkpoint。不同 checkpoint 的绝对差异不能归因于一次存储替换。

### 4.3 当前入口的实际限制

1. **`rfx.FinalRead` 不能直接处理 S-RE。** 它对 DPFP、ridge、unit-query 有明确限制，且只处理旧条件的 L1/final-token。应新建适配接口，保留其保护条件。[R20]
2. **`read_fix_run.validate()` 固定旧验证范围和 seed。** 新研究需要显式 manifest 数据接口，不能继续把反复用于筛选的 `val[0,1000), seed4242` 当新确认集。[R21]
3. **普通 `LLMExperiment._load_writer()` 不接受 SQ2 的导出结构。** SQ2 使用 adapter/config/arm/read_fix_version 等字段；通用 writer 入口期待另一套结构。需要显式导入器，保留完整标准化 buffers，并做一致性测试。[R17][R22]
4. **`stage='writer'` 会强制生成 T1 设置。** 不能仅把 YAML 中任务名改为 T2，就假定训练任务已经切换。[R17]
5. **`writer.pt` 不包含训练期 value head。** 如需 clean/read head 的离线诊断，应加载匹配 `checkpoint.pt` 的头；它只能用于指标审计，不能进入普通预测路径。[R21][R22]
6. **现有 `submit.py` 的多节点运行未实现。** 本计划使用单节点。科学作业传新实现的完整 commit；`--commit HEAD` 会在脚本绑定的主仓库解析，不能假定是当前临时 worktree 的 HEAD。[R23]
7. **原 `read_fix_run.fit()` 不是通用深度训练器。** 它沿用 L1 调用及固定的训练超参数。新 runner 必须把阶段配置真正传入前向、优化器和任务构造，并在 smoke 记录实际 loop/read 次数；不能仅增加 YAML 字段后仍调用旧训练逻辑。[R21]

---

## 5. 数据、信息访问与统计协议

### 5.1 新研究使用独立 manifest

每一个 episode 的身份由以下字段共同确定：生成器版本、task/layout digest、base seed、split、index。只换一个文件名并不产生新数据。现生成器由这些字段派生 episode seed，应继续保留该规则。[R24]

建议采用新的研究命名空间：

```text
study_id = "slot-loop-v1"
train_base_seed(stage, run_seed)
    = derive_seed(20261029, study_id, "train", stage, run_seed)
eval_base_seed(purpose, stage)
    = derive_seed(20261029, study_id, purpose, stage)
```

这是新 seed 约定，需由实现者将解析后的整数写入 manifest。训练数据的 base seed 在同 seed、不同 arm 之间相同；calibration/dev/test 的 base seed 与训练不同，且不含 run_seed、arm 或 checkpoint。所有训练 seed 与 arm 读取完全相同的评估 episode IDs，才能做第 5.4 节的跨 seed 配对。记录并检查生成出的 episode IDs 与既有数据登记是否重复。

| 用途 | 数量建议 | 允许使用的信息 | 使用次数/限制 |
| --- | ---: | --- | --- |
| 旧集复现 | 旧 1,000 episode | 预测、旧指标 | 只用于兼容性；不算新证据 |
| smoke | 64–128 episode | 正确性、耗时、有限性 | 不据此决定正式科学结论 |
| calibration | 2,048 episode | 固定温度/增益网格的 NLL | 每个阶段选一次，选择后锁定 |
| dev | 1,000 episode | 架构开发与 checkpoint 选择 | 记录每一次查看与选择 |
| confirmation | 2,000 episode | 最终预测和注册指标 | 仅对锁定方案开封 |
| unseen-name 诊断 | 2,000 episode | 独立名字池上的表现 | 单独报，不替代主确认 |

各用途可在独立 purpose seed 下使用 index `[0,N)`。如必须沿用数值窗口，先检查最新 registry；例如旧 RFX 已使用过 `val` start=200000、seed4242，不能把这一区间再称为全新验证集。[R07][R20]

S1A、S1B、S2 各自使用独立阶段命名空间。若根据已开封的 S1 测试结果修改 S2，S2 必须使用新的确认集。测试集保护逻辑应新增研究登记，不能删掉原有保护或伪装成 validation 来绕过。

### 5.2 信息访问协议

第一轮保持 memory-only：支持事实用于构建所选记忆，问答阶段只能访问该记忆和问题本身。所有 arm 的系统提示、分词、support/query 边界、支持 facts 顺序、答案评分规则相同。

支持文本的 Transformer KV cache、raw support、完整隐藏状态和另一种记忆表示可以在诊断过程中临时存在，但不得被普通问答前向读取；正式部署计时与空间统计需将这些保留项明确计入或释放。

正确 fact index、桥实体、true-hop 和答案只可进入训练 loss 或离线指标函数，不得进入非 oracle 的 `forward/read/stop` 接口。当前问题的金标答案不能写回状态。每个问题从同一 episode 的相同只读快照出发，不能因先前问题的答案而改变记忆。

### 5.3 指标定义

| 类别 | 必报指标 | 解释 |
| --- | --- | --- |
| 最终任务 | 完整词表 top-1 EM、NLL、每 hop EM、UNKNOWN 指标 | candidate-filtered EM 仅作诊断 |
| 寻址 | key top-1、正确 slot 权重、最高/次高间隔、熵 | 多跳按当前模型状态对应的实际查询记录 |
| 读出 | 每样本 \(\|m-v_*\|^2/(\|v_*\|^2+\epsilon)\) 后取平均、余弦、norm ratio | 主文必须标明是该定义还是旧报告的原定义；不能混算比值与均值 |
| 表征 | clean/read value-head 准确率 | 只用匹配且冻结的审计头，标明它不参与推理 |
| 循环 | 下一正确地址率、桥实体准确率、每次实际 read、终态漂移 | 只看最终 EM 不足以解释机制 |
| 资源 | 持久状态字节、存储/训练参数、准备峰值显存、训练步时、端到端问答延迟 | 分开报告，不用一个 memory 数字替代全部成本 |

对 early position 不存在明确单事实监督的情况，不给它强行分配最终答案 fact。S0 的 read-error 主诊断限最终有效问题位置；all-token 的最终 EM 仍正常计算。

多跳读误差中的目标 value 分两种：正确路径预期的下一事实，以及模型实际状态已明确解码后对应的合法下一事实。分别标记为 path-target / state-target；状态无法可靠解码时不填后者，不能把两者混为同一准确率。

### 5.4 统计与预先锁定的判定

新计划建议以下阈值，**这些是待登记的研究决策标准，不是已有结果**：

- S1A 只有一个主对比：SoftSlot-L1 − Matrix-L1。目标为 EM 提高至少 3 pp，成对 95% 区间下界 >0，三个续训 seed 的点估计方向一致。
- S1B 有两个主对比：SoftSlot-L4 − Matrix-L4，以及 SoftSlot-L4 − SoftSlot-L1，均在 hops 2–3 上报告。使用 Bonferroni 的 97.5% 区间；循环主张要求第二个对比至少 +3 pp、区间下界 >0，同时 hop1 的下降不超过 2 pp。第一个对比不成立时，不能声称 slots 优于矩阵。
- 长跳 h4–8、same-checkpoint depth sweep、读误差与成本是预登记次要指标；不能在主指标失败后换一个成功子集作为主结果。

使用 10,000 次**成对、整 episode** bootstrap。同一个 episode 的多个问题必须一起重采样；多个 seed 的同一 episode 也保持配对。主区间可在先对三个 seed 求平均后对 episode 重采样，明确它**条件于这些训练 seed 和同一个 parent writer**。另外逐 seed 报数，不把三个续训 seed 写成三个独立 writer 重训。

如果种子稳健性是论文主张的一部分，后续需要独立 writer parent 或新数据布局复现。当前三个续训 seed 不能替代这个证据。正式开封 test 前锁定指标脚本、contrast 列表、阈值、失败处理和 bootstrap seed。

---

## 6. S0：冻结 checkpoint 的最小读替换实验

### 6.1 固定项

主 checkpoint 为 S-RE，Qwen3-0.6B-Base，原注入层、r128、DPFP ν3、T1/E16/四关系、L1、memory-only。保留全部原参数、归一化 buffers、事实写入位置、问题格式和所有有效 question-token 的读取位置。[R03][R22]

S-RE 只按 L1 训练。对它直接运行 L4/L8 可以另存为未训练深度诊断，但不纳入 S0 主结论，也不据此判定循环研究的成败。

### 6.2 条件表

| 条件 | 读算子 | 参数更新 | 主要用途 |
| --- | --- | --- | --- |
| Native | 完全原始 S-RE forward | 无 | 复现与 hook 一致性 |
| Matrix | \(m=Mq\)，M 和 q 与原模型相同 | 无 | 新接口的原算子对照 |
| SoftSlot | \(m=V^T\operatorname{softmax}(Kq/\tau)\) | 无 | 主要新读算子 |
| HardSlot | \(m=v_{\arg\max_i k_i^Tq}\) | 无 | 查询能否选中可解码事实的诊断 |
| Clean-final | 最终问题位置注入正确事实的 value | 无；使用标签 | 明确标为 oracle 诊断，不是可部署 arm |

主替换 scope 为 `all_valid_question`。可额外提供 `final_only` 定位表，但每张比较表内保持相同 scope。Clean-final 仅在最终位置使用金标 value；不能把最终答案事实广播到较早的问题位置，制造不存在的因果信息。

`final_only` 诊断需显式持有原 MatrixState、替换 SlotState 与 scope mask；非最终位置继续读取原 MatrixState。这个双状态上下文不能用于 slot 部署内存/延迟主张，正式后端不得保留隐藏 fallback。

SoftSlot 同时改变了混合、归一化和可能的遗忘/recency 机制。因此原始对照检验“整个读取/存储算子的替换”，不能把全部收益精确归因为消除了线性干扰。如需缩小归因，补充一个预登记的 recency-matched SoftSlot：将原 writer 的固定 recency 权重以 `log(weight)` 加入 slot logits，单独报告，禁止在测试后挑更好的版本。

### 6.3 温度和尺度

先报告未校准的 \(g=1,\tau=1\)。校准版使用固定网格：

```text
temperature = [0.03, 0.1, 0.3, 1.0]  # 仅 SoftSlot
gain = [0.5, 1.0, 2.0]              # Matrix / SoftSlot / HardSlot 均允许
selection = calibration NLL minimum
tie_break = gain nearest 1, then larger temperature
```

校准只调整一个全局温度和一个全局正增益，不允许逐样本用正确答案选择它们。记录所有候选、校准耗时和选中值；dev/test 不再调参。新实现应将 τ 和 g 的取值计入导出配置；如果以后改为可学习参数，另立训练条件。

### 6.4 步骤与验收

1. 解析 S-RE metadata，核对原运行 config 与实际 loader。
2. 在旧验证集回放 Native，仅用于兼容性，期待原正确数 723/1,000；不同软件/数值路径导致差异时先定位，不直接继续科学比较。
3. 插入 Matrix 接口，在同 batch 上比较 logits。沿用同一数学运算路径时要求完全一致；若变更了 kernel/dtype，必须在看新性能前登记数值容差并解释原因。
4. 从 `FixBatch.keys/values` 捕获实际写入表征，校验 shape、特征映射、归一化和事实顺序。不要把 `base_keys` 当 DPFP keys。
5. 跑未校准与校准的同内容对照；所有 arm 使用相同 episode ID 与问题顺序。
6. 按阶段 5 的粒度输出最终 EM 和读出诊断；保存样本级预测与 slot 权重摘要。
7. 形成 `decision_s0.md`，选定 S1 的主后端与参数。D128-3 只做一次主要方向的辅助复查。

### 6.5 结果如何决定下一步

| S0 模式 | 合理解释 | 下一步 |
| --- | --- | --- |
| Soft/Hard 明显改善，读误差下降 | 同一表示还有可恢复的信息 | 进入 S1A 的匹配训练 |
| Hard 好、Soft 差 | soft mixture 的选择性/温度/尺度仍有问题 | 锁定校准规则后训练 soft；hard 暂作诊断 |
| Clean-final 好、Hard 差 | 寻址、角色编码或 query 是主要瓶颈之一 | 仍可做有限 S1A；随后优先角色/转移研究 |
| final-only 改善、all-token 不改善 | 非最终位置或上层接口存在分布问题 | 保持结果分开；训练新读取位置协议再比较 |
| 全部替换不改善 | 即插即用替换没有解决问题 | 允许一次预先限额的 S1A 适配；不无限调参 |
| Native 无法复现、owner/mask 错误 | 结果无法科学解释 | 修复接口，暂停性能比较 |

可将 dev 提升 ≥5 pp 且读误差下降作为“优先投入”的探索信号，但不能当统计确认。S0 不改善只能否定当前冻结替换的效果，不能证明训练后的新读算子必然无效。

---

## 7. S1A：单跳、同起点、同训练预算

### 7.1 主实验仅两臂

- **Matrix-L1：** 从 S-RE 出发，继续训练读取器，保留原 ridge 存储。
- **SoftSlot-L1：** 从完全相同的 S-RE 状态出发，替换为显式 KV/softmax 读取。

HardSlot 的 argmax 会切断 query 的选择梯度，故不作为主要训练臂。若以后使用 STE、Gumbel 或稀疏路由，需另行定义优化对照。

### 7.2 正确冻结 writer

首先冻结 `A, K, V, a_lambda, a_beta`，以及存在时的 `w_lambda, w_beta`；冻结 `input_mean/input_std` 等预处理 buffers、DPFP ν、ridge eta 和归一化规则。仅训练两臂共同具有的 `Q, R, U, B`。α、ρ 与增益/温度在第一轮固定。

**A 同时用于 writer 与 query 输入。** 冻结 writer 就需要冻结这一个共享 A。若想另学 query 的输入投影，应复制为 `A_write/A_query` 并对两臂同样处理，额外增加的 \(r d_{in}\) 存储参数必须报告；这属于后续架构消融，不是本轮默认。[R17][R25]

每个 run 记录实际 trainable/frozen 名字、参数数量和 buffer 摘要。几个 smoke 更新后，在同一个固定 support 批次、相同 precision 下验证 writer 表征不变。不同 episode 的记忆本来可以不同。不能仅通过模块命名判断冻结是否成功。

### 7.3 训练约定

| 项目 | 首轮建议 |
| --- | --- |
| 训练任务 | T1，E16，四关系，沿用原 fact renderer 与 memory-only 信息协议 |
| 输入与读位置 | 两臂相同；全部有效 question-token |
| 深度 | L1 |
| parent | 同一个 S-RE；三个续训 seed 为 17/29/43 |
| 可训练集合 | Q/R/U/B；无额外训练 head |
| 主损失 | 完整词表答案 CE/NLL；先不增加 coefficient/bridge loss |
| 优化器 | 两臂均新建 AdamW，weight decay=0.01，clip=1.0 |
| 学习率 | 小型 pilot 候选 3e-4 / 1e-3 / 3e-3；两臂候选数相同，主运行前锁定 |
| batch | 默认沿用 32 episode、每 episode 4 question；实际有效样本数完整记录 |
| 步数 | 上限 6,000；由耗时 pilot 确定所有 arm 相同的正式步数，不能跑到“满意”为止 |
| 模型选择 | 相同预定 checkpoint 频率；dev NLL 最低，平局取较早 checkpoint |

S1A 的 g/τ 继承 S0 在独立 calibration 上选定的各 arm 全局值，训练和 checkpoint 选择时不重调。学习率 pilot 可采用各候选 200 步，使用同一小型开发分片；pilot 权重不作为正式 run 起点。首轮可对各 arm 选择自己的最佳学习率，但需同等搜索预算，并报告“调参后公平比较”；若要判断同一优化器轨迹下的纯算子差异，增加一个共同固定 LR 的描述性对照。

因为 writer 已冻结，原先只作用于 writer 的辅助损失无需继续计算。读出监督若作为第二轮加入，必须在两臂同时加入并同样调权；否则不能归因于 memory backend。

### 7.4 判定与有限重试

与本轮 Matrix-L1 对比，不以历史 72.3% 作为唯一 baseline。按第 5.4 节做确认。若 dev 显示训练稳定但无收益，可用预先保留的一次接口消融预算检验 `A_query` 独立或重新初始化读取器；两臂同样修改。该次仍无收益，则停止扩大 slot 单跳投入，记录负结果，转向 query/任务接口定位。

不能因为结果不理想再不断更换 LR、loss、验证窗口，并把最终最优一次当预注册结果。

---

## 8. S1B：将“存储效果”与“循环效果”分开

### 8.1 最小 2×2

| 后端 | 独立训练 L1 | 独立训练 L4 |
| --- | --- | --- |
| Matrix | M1 | M4 |
| SoftSlot | S1 | S4 |

四臂均从同一 S-RE 的 writer/reader 初始状态出发；writer 冻结，Q/R/U/B 训练，数据流、位置规则、总更新数一致。不要让 slot 从一个另行训练得更好的 reader 开始，而 matrix 从旧 reader 开始。S1B 在其独立 calibration 集上按 S0 的同一网格、用 L1 parent 选一次 g/τ；同后端的 L1/L4 共用这些值，所有正式 checkpoint 不再重调。

初始任务使用明确登记的 T2 chain-end。为减少生成器变化，可复用旧 T2 的 E12、两链布局、关系恒 0 设置，训练问题 hops1–3，测试 hops1–8。**这与 S1A 的 E16/T1 不同，因此只在本节四臂内部作主比较。** 多关系组合是 S2 的新任务，而不是把关系恒 0 的 T2 成功解释成关系组合能力。

如果选择 E16 或多关系 T2 作为替代，必须在运行前锁定新 task/layout digest，四臂完全相同；不得运行后选择更有利的任务布局。

### 8.2 深度和计算控制

首先独立训练 L1、L4，二者均使用最终答案损失；这一步衡量部署任务表现，不要求 L1 学会逐边算法。训练步数相同只代表样本/更新预算相同，**不代表 FLOPs 或时间相同**。记录全部训练/推理成本。

对 all-token 协议，区分 `loop_depth=L`、`logical_reads=有效读取位置数×L`、批量 backend 调用次数与实际 kernel 时间。不能把每题所有 token 的成本写成仅 L 次标量读取。后文 h+1 的计数适用于 S2 单个问题控制器。

对已训练的 S4/M4 再运行同 checkpoint 的 L=1/2/4/8 扫描，保持 memory 和输入相同。这是深度敏感性诊断，不能替代独立 L1 baseline。

如出现循环收益，再增加两个最必要的机制对照：

1. **固定 query 的 L4：** 每轮使用同一初始 query 的读出，以区分反复非线性变换与重新寻址。
2. **warm-start untied L4：** 将共享参数复制到每一步，先验证 logits 一致，再同预算训练；用来检验共享带来的参数效率或归纳偏置，不重复不匹配初始化的旧 B8。

它们是第二批对照，不能在主要 2×2 尚无可靠结果时一次性铺开。

### 8.3 继续条件

只有 S4 对独立 S1 的多跳增益和同 checkpoint 轨迹共同支持循环，才进入 S2 的语义控制研究。若 slots 只提高 L1，却不提高 L4 的组合收益，应将其结果定位为读取改进。

若 M4 同样有效而 slots 没有额外优势，S2 可以以矩阵读取器为主；研究目标是验证有效机制，不预设某一种存储必须获胜。

---

## 9. S2：角色对齐、逐转移与终态，逐项实施

### 9.1 新增因素阶梯

| 条件 | 新增内容 | 必须匹配的对照 |
| --- | --- | --- |
| Base-old | S1B 的原读取器结构，在本节新任务上重新训练 | 避免把数据变化当架构收益 |
| Arch | 新 p/a 状态与相同 mention 输入，只有最终答案损失 | 后续所有新增 loss 的直接基线 |
| Arch+Role | head/tail/query 的实体角色对齐 | matrix/slots 接收同样结构提示和 loss |
| +Transition | 模型实际 rollout 的下一地址/桥实体监督 | 不使用该监督的同结构模型 |
| +Null | 可见图上的 NULL、状态保持与 UNKNOWN | 固定深度、无 NULL 的同结构模型 |

先固定深度训练，之后才试混合深度和动态停止。每次只加入一项，检查增益和负面影响。新状态结构和 role loss 分两步检验；Arch 与其后各变体使用相同新参数初始化和最终答案损失。S2 使用新的确认数据，不反复查看 S1 的 test。

S2 第一版统一采用 `last_valid_question`：从本阶段训练开始，仅在最后有效问题 token 注入 memory branch；其他位置只保留冻结基础路径。所有 arm，包括 Base-old，都按此规则重新训练和评估。旧 all-token 模型仅作历史参考；不能用一次推理消融替代这个位置匹配的基线。

### 9.2 角色表示与参数预算

第一版尽量保持动态状态总维度 128。例如 p 使用 96 维、a 使用 32 维，拼接后仍为 128 维；c 复用原有固定 u，不额外复制一个完整问题状态。输出为 \(Wx_q+\alpha B[p_L;a_L]\)。维度比例是待验证默认值，不是理论最优。

S2 首轮继续冻结 S-RE writer，支持集的 K/V 不变。新的文本 mention 编码器只服务于 reader 的状态初始化和训练目标，不静默替换支持集的 key/value。`f_entity` 输入为实体 mention 的文本/冻结 backbone 特征，输出 96 维；它不能以实体 ID 查表。训练时可使用 renderer 提供的 span 和同实体对应关系；这些是结构监督，必须对所有主对照开放并明确记录。推理时不能查 oracle ID/codebook。

建议维度契约为：c∈R128，p∈R96，a∈R32，m∈R128；F_addr 将 [p,c] 的 224 维映射到原始 query 的 128 维，再经原 DPFP/归一化得到 768 维 q；F_step 将 [p,m,c] 的 352 维映射到 96 维；F_answer 将 [a,m,c] 的 288 维映射到 32 维。矩阵和 slots 使用相同映射。训练目标中的 f_entity 不作为额外事实记忆存入推理状态。

首版 Arch 固定以下实现，不开启额外架构网格：c 为原冻结标准化路径的 u；p0 为问题起始实体 span 的冻结隐藏特征均值，经共享单层线性 mention encoder 后 L2 归一化；a0 为 32 维零向量。F_addr 为一层线性映射后接原 query_code；F_step 为一层线性映射后 L2 归一化；F_answer 为一层线性映射后 tanh。各映射跨循环步共享，线性层 Xavier 初始化、bias=0，归一化 epsilon=1e-6。新坐标下不直接继承旧 B 的语义：S2 各 arm 的输出 B 均零初始化，并记录完整初始状态。相关 span 属于本节明示的结构输入；模板以外的自动 span 识别是另一个实验。

如果冻结 writer 限制了角色对齐，后续可新建 S2-W：从文本 head/relation/tail 重训角色化 writer。但那会同时改变写入表征，必须对两种 backend 共同重训，并单列结果；不能与本轮的冻结 writer 对照混合。

如果新模块的映射参数超过旧 Q/R/U/B，先完整报告增量，再增加参数预算匹配的对照；不能只说“rank 还是 128”就声称参数相同。

### 9.3 真正的逐转移目标

令训练路径为 \(e_0\to e_1\to\cdots\to e_h\)，采用模型自身读出得到的 \(p_{s+1}\)：

\[
\mathcal L_{step}
=\sum_{s<h}d\bigl(p_{s+1},\operatorname{sg}(f_{entity}(e_{s+1}))\bigr).
\]

配合有负样本的对比/分类约束，避免所有实体 code 坍缩到同一点。再加 head/tail/query 对同一实体的跨角色一致性损失。冻结或 stop-gradient 的目标分支、负样本来源、温度和 loss 权重都要写入配置。

主训练使用自由 rollout；如加入 teacher forcing，单独登记比例和退火曲线，评估时比例为 0。训练前 h 步的桥实体监督，与 TSM 的“到达终点后反复预测最终答案”是不同实验。

对于实际读取不足 h 次的样本，只计算已执行转移的桥实体损失；不能一面强制逐边转移，一面要求它在不足深度时直接输出终点。固定答案任务损失如何适用于短预算必须预先定义。

### 9.4 任务与因果约束

S2 首轮引入至少两种关系，**每道 chain-end 问题明确指定一个要反复跟随的关系**，支持图同时包含其他关系的干扰边。训练 hops1–3，外推 h4–8，控制总事实数及干扰边，避免 hop 与支持长度完全绑定。随机化实体名和事实顺序，生成器禁止利用答案 token 排序等捷径。

变长关系序列是后续独立任务：成功条件是完成问题中给定的关系序列；序列尚未完成时遇到 NULL 应为 UNKNOWN，不能按 chain-end 规则当成功终止。

`q_hop_true` 只允许出现在标签/诊断文件。chain-end 的推理输入不能获得真实 hop。若 exact-walk 问题在文本中明确给定步数，则该文本信息可用，但不得把 metadata 的隐藏 true-hop 作为控制器输入。

### 9.5 NULL 与停止规则

主对照给两种 backend 增加同结构、同输入范围的 `NullHead(p_s,m_s,c)`，预测当前关系是否没有有效出边；默认阈值 0.5，训练与校准规则在测试前锁定。matrix 的 NullHead 不能额外查看原始 K/V 或 max slot score，否则它就保存了另一份记忆，必须另计资源且失去纯 matrix 对照。

NULL 标签按可见图的实际实体与关系定义。第一版可在从可见图抽取的已知实体 code 上单独训练该 head，再评估自由 rollout 的迁移。若实际 rollout 偏离真路径，不能仅按 `s < true_hop` 给当前状态标 continue；没有明确实体对齐对象的 rollout 状态不计算逐点 NULL 监督。路径意图标签与实际查询标签必须分开。普通推理仅依赖预测 head。

slot attention 中加入始终有效的 NULL 项可作为后续特有消融；此时才研究 null logit 与匹配分数的绝对尺度。不能让 slot 使用这种额外机制，却声称与无 NULL 的矩阵进行了单因素公平比较。

第一版限定：闭世界、完整支持图、无事实覆盖、E 不超过容量。没有指定关系的有效出边时，NULL 表示可以停在当前实体。后续 missing-fact/eviction 任务中，同样的检索失败输出 UNKNOWN；不将其记为成功到达终点。

**读取计数：** 若每次非 NULL 读取推进一条边，到达终点后再查一次 NULL，则 h 条边需要 h+1 次读取。测试 h8 时最大允许读取至少 9 次；所有 baseline 的预算比较也按真实调用次数计。达到预算但尚未确认 NULL 时设置 `budget_exhausted=true`，不能伪装成已完成。

推理动作顺序固定为：由 p_s 生成 query → 读 m_s → 判断 NULL → 若为 NULL，则保持 p_s/a_s 并进入吸收态；否则才执行 p/a 更新。吸收态不再产生新的逻辑 read。预算耗尽的严格任务输出为 UNKNOWN，同时保存未强制停止的答案分布供诊断，不能用其最有利答案替换主指标。若批处理实现仍对吸收行计算完整 kernel，应如实报告执行成本，不声称这些行已节省实际算力。

训练先采用固定展开预算，NULL 的辅助标签按前述可见图样本产生；默认用预测 continue 概率软混合“状态保持”与“候选更新”，不使用金标 hop mask 决定更新。具体令存活概率 w0=1、t_s=1−P(NULL)，则 p_{s+1}=p_s+w_s t_s(p_candidate−p_s)，a 同式，w_{s+1}=w_s t_s。NullHead 为输入 [p,m,c] 的单层线性二分类头，初始权重为零、bias=−log(99)，使初始 null 概率为 0.01。推理使用硬阈值 0.5；软训练与硬停止的差异列为必报消融。不得由执行者临时选择不同停止动作。

S0/S1 的无 NULL slot 在全空状态下返回零向量，避免 NaN；S2 的主版本由共同 NullHead 判断，slot 特有的 attention-NULL 版本则在全空状态下选择 NULL。不同版本需要显式配置，不能静默改变语义。

### 9.6 机制验收

至少报告：每步正确地址率、value 到下一 query 的转换准确率、完成后额外读取的答案稳定性、NULL 的 false-stop/missed-stop、实际读次数和 budget-exhausted 比例。

对中间状态做一次明确的因果干预：在固定支持图中，把当前桥实体状态替换为另一可识别实体的匹配状态，检查后续答案是否按新分支变化。干预条件只用于机制分析，不作为普通测试预测。最好同时给出无关状态扰动和同实体重编码控制。

若仅最终 EM 提高、轨迹与干预不支持逐跳语义，应将结果表述为低维迭代计算的收益，避免声称已学会可解释的关系遍历。

---

## 10. S3：容量、状态字节与实用性

### 10.1 当前 E16 下的存储计算

以下按 FP32、每 episode、单注入层计算，先只列内容 payload：

\[
B_M=4d_vd_k,\qquad B_{KV}=4E(d_k+d_v).
\]

| 表征 | 矩阵内容状态 | E16 槽位内容状态 | payload 相等时的 E |
| --- | ---: | ---: | ---: |
| 原始 128 维 key/value | 65,536 B = 64 KiB | 16,384 B = 16 KiB | 64 |
| DPFP ν3：key768/value128 | 393,216 B = 384 KiB | 57,344 B = 56 KiB | 109.714… |

该计算表明：在当前很小的事实数下，显式保留 KV 的内容状态可以更小。它没有说明任意长上下文下都更省，也没有计入全部 runtime 元数据、控制器、参数与临时空间。

若 slot 状态含 `valid: bool[S]` 和 `count: int64[1]`，逻辑持久张量字节为：

\[
B_{slot}=4S(d_k+d_v)+S+8.
\]

由此，393,216 B 的**记忆状态上限**允许 DPFP 最大 S=109，实际 390,773 B；65,536 B 上限允许普通 key/value 最大 S=63。不能把忽略 metadata 的 S=110 或 S=64 写成满足这些完整记忆预算。

上述上限只约束 memory state；两边共同的 query/controller 状态另报。若新增每槽 age、relation 标记、压缩参数或额外 control state，按实际结构重新计算容量。批处理常驻内存按 batch/episode 复制情况计，不只报单样本理论值。

### 10.2 两种不同对照

**同内容实验：** E=16，slot S=16；两种后端接收相同 facts 和实际 K/V。报告内容相同、slot 状态较小。不要称“容量完全相等”。

**固定预算实验：** 给出同一个 bytes cap，slot 从一开始分配可用最大容量，两边接收同一批 E 个 facts。分别报告 capacity、实际占用的事实数与总已分配字节。未使用的槽位是明确的后续容量，不能额外填无用数组来制造“内存相等”。

### 10.3 负载与溢出

建议依次测 E=16/32/64/96，再考虑 E=128/256。S-RE 在 E16 训练，直接改 E 的结果先标为负载分布外压力测试；正式负载结论需匹配训练负载和数据复杂度。

在 S=109 时，E128 已经溢出。第一版直接报错，不静默截断。只有明确登记 FIFO、更新覆盖或检索选择等策略后才进入超容量实验。两边看相同原始输入，并同时报告：支持事实保留率、正确路径的可访问率、条件于证据仍存在的准确率、总体 EM/UNKNOWN。

矩阵可以不断写入，并不意味着它能精确保留无限事实；slot 的淘汰也不能被隐藏在总体指标里。固定状态研究应展示随负载变化的曲线，避免只选择某个最有利 E。

### 10.4 部署测量

S0 的 `FixBatch` 会同时保存矩阵、K/V、raw support 等诊断材料，不能直接据此报告 slot 的部署内存。[R25] 正式 S3 应只保留该 arm 的 backend state；释放支持历史和另一种表示。分别测量：

1. 支持文本编码与记忆准备时间；
2. 持久 memory/control state 的逻辑字节与真实已分配显存；
3. 单次 read 和完整问题 forward 的延迟；
4. 训练峰值、准备峰值和问答峰值；
5. 实际上层 LLM 前向次数、批处理大小与 padding 比例。

在同硬件、相同 dtype、相同 batch 上预热，计时前后同步 GPU；记录中位数、P90 和测量次数。动态停止若每步调用上层 LM，必须把这些调用完整计入。可以另报低维 read 微基准，但不能拿它代替端到端延迟。

### 10.5 实际任务扩展的进入条件

只有同内容、独立 L1 对照和成本测量均可靠后，才选择一个真实任务。优先选择有限证据、多事实组合、可以审计证据链的任务；分别设置 memory-only 和 context-retained 条件，并报告支持抽取/实体识别的实际错误。

若成功主要依赖模板 span、固定名字池或金标路径监督，保留这些边界。真实文本结果必须单独建立，不能由合成关系遍历结果自动推出。

---

## 11. 实现拆解与接口契约

### 11.1 只修改必要范围

以下路径是**建议新增**，当前基线没有这些模块。最终实现文件名可以调整，但接口和行为验收应保留。

| 新文件/修改点 | 职责 | 第一批是否必需 |
| --- | --- | --- |
| `src/looplora/core/memory_backend.py` | MatrixState/SlotState、显式 owner、只读 read、bytes 核算 | 是 |
| `src/looplora/llm/slot_read.py` | 实际 query_code 与 backend 对接、读取 scope、轨迹捕获 | 是 |
| `src/looplora/llm/slot_experiment.py` | S-RE/DPFP 显式导入、参数冻结、task/manifest 接入 | 是 |
| `src/looplora/llm/slot_run.py` | preflight / s0 / train / evaluate 的统一目标 CLI | 是 |
| `scripts/prepare_slot_manifest.py` | 标准库元数据校验、数据划分登记、配置展开 | 是 |
| `scripts/analyze_slot_study.py` | 从样本级记录生成对比、区间、资源表与 verdict | 是 |
| `configs/slot_loop_v1/` | 冻结的阶段配置和 release 文件 | 是 |
| `docs/prereg/SLOT-LOOP-v1.md` | 新假设、数据和预算锁定，引用旧研究 verdict | 是 |
| `src/looplora/llm/semantic_reader.py` | p/a 角色状态、转移与 NULL；S2 新增 | 后续 |
| `tests/test_slot_backend.py`、`tests/test_slot_experiment.py` | 语义和兼容性检查 | 是 |

现有 `MemoryContext.state`、`SupportSnapshot.states`、`AdapterBank` 和 adapter 读取路径主要假定 Tensor 状态，不能简单用一个全局 closure 偷偷塞入 KV 表。应显式更新类型与快照操作，并保留旧 matrix 路径。[R16][R26][R27][R28]

还需迁移 `src/looplora/llm/batching.py` 的 `repeat_snapshot`：目前它直接对 Tensor 调用 repeat_interleave。首版约定 owner 始终引用原始 B 个 episode 行，问题展开只展开 owner，不复制整份 state；如必须物理复制，则创建独立映射并做等价测试，不能混用原行号与复制后的行号。[R29]

### 11.2 最小状态接口

下面是**待实现的类型契约**，不是可直接导入的现有代码：

```python
from dataclasses import dataclass
from typing import Protocol
from torch import Tensor

@dataclass(frozen=True)
class MatrixState:
    matrix: Tensor       # [B, dv, dk]

@dataclass(frozen=True)
class SlotState:
    keys: Tensor         # [B, S, dk], actual written features
    values: Tensor       # [B, S, dv]
    valid: Tensor        # [B, S], bool
    count: Tensor        # [B], int64

MemoryState = MatrixState | SlotState

class MemoryBackend(Protocol):
    def read(self, state: MemoryState, query: Tensor,
             owner: Tensor) -> Tensor:
        # query [N, dk], owner [N] -> readout [N, dv]
        ...

    def select_rows(self, state: MemoryState,
                    rows: Tensor) -> MemoryState: ...

    def clone_state(self, state: MemoryState) -> MemoryState: ...

    def reset_rows(self, state: MemoryState,
                   rows: Tensor) -> MemoryState: ...

    def persistent_bytes(self, state: MemoryState) -> int: ...
```

`owner` 明确指出展平后的每个 query 来自哪个 episode。B×Q×T 的展开、padding 筛选、batch 重排和 question fork 都必须同步处理 owner，不能依赖隐含 reshape 猜测。

第一版只支持 **build-final + read-only memory-QA**。Matrix builder 复用原 delta/ridge 实现及其 write mask、reset、实际 gates/recency 规则；不能只凭 K/V 重新猜一个写入过程。Slots builder 保存同一条原路径捕获的实际 K/V，超过 S 直接报错。

`query_code` 由原 adapter 负责；backend 不再次做 DPFP 或归一化。矩阵 read 为按 owner 索引 M 后计算 Mq；slot read 为按 owner 索引 K/V，mask 后 softmax 并求值加权和。全空行显式返回零，不能对全 −∞ logits 直接 softmax。

未实现的 `online_write` 入口必须报“不支持”；不得静默不写、覆盖最早事实、跨问题继续累积或保留额外历史。今后实现在线版本时另行登记写入时机、因果性和容量策略。

### 11.3 导出格式

新导出至少包含：格式版本、parent checkpoint 摘要、source commit、task/layout digest、backend 类型、key/value 维度、feature map、normalization epsilon、τ/g、capacity、读取 scope、loop steps、参数/预处理 buffers、冻结集合和训练配置。

模型权重导出与 episode state 导出分开。模型导出不能意外捆绑最后一个训练 episode 的 slots。重载后必须复现预测；只恢复 Q/R/U/B 而漏掉 input_mean/std、DPFP 或 scope 不算有效重载。

---

## 12. 必要验证与错误即停条件

| 检查 | 方法 | 不通过时的处理 |
| --- | --- | --- |
| 原路径一致性 | 新 Matrix backend 对原 S-RE 同批 logits/预测 | 阻塞全部科学比较 |
| owner 与 mask | 两个 episode 放入可区分 sentinel，重排/展开/padding 后核对 | 修复索引，不看性能结论 |
| feature 一致性 | 实际 keys/query 维度、DPFP、norm 与原路径一致 | 禁止用 raw/base key 代替 |
| slot 参考读取 | 小矩阵手算、空行、padding slot 对结果无影响 | 修复后再跑 |
| 只读隔离 | 多个问题与不同 depth 前后状态内容不变 | 阻塞 memory-only 结论 |
| writer 冻结 | 固定 support、同 precision，优化前后 K/V/M 与 buffers 一致 | 修复共享 A/参数别名 |
| 梯度可达 | reader 有有限梯度，冻结 backbone 不更新但梯度链贯通 | 禁止整个训练路径 no_grad |
| 容量边界 | E=S 正常，E>S 报错 | 禁止静默截断 |
| 字节统计 | `numel()*element_size()` 合计含 mask/count | 修正预算与 Smax |
| 数据隔离 | episode IDs 跨用途无重复，评估 IDs 跨 arm/seed 一致 | 重新锁 manifest |
| 导出/重载 | 同输入预测一致，scope/backend/normalization 恢复 | 不发布该 checkpoint |
| NULL 标签 | 仅改变指定关系出边，标签按可见图变化 | 修复语义，不用 true-hop 控制推理 |

这些是有明确风险对应的行为检查。第一轮不要求重跑大量与改动无关的测试；若项目已有必须满足的回归门槛，则执行该门槛。所有导入 torch、训练、推理和 GPU 测试通过仓库的 compute-node 作业入口运行。

工程正确性失败时立即停止性能比较。科学假设失败时保留结果，按预先定义的下一步走，不通过更改测试协议把失败消掉。

---

## 13. 配置模板与目标 CLI

### 13.1 命令状态说明

**已经存在：** `scripts/submit.py`、原 S-RE/DPFP loader、当前报告和数据生成器。

**需要先实现：** 本节的 `slot_run`、manifest 准备脚本、分析脚本、测试文件和新配置。下面是实现者应提供的命令契约，**不能在审查基线 commit 上直接执行成功**。示例没有在本轮提交作业。

代码完成后，先提交包含实现、测试和阶段配置的新 commit，再把其完整 40 位值填入 `IMPL_COMMIT`。每次阶段 release 的配置也要进入可追溯 commit，不能使用未提交文件进行科学运行。

**输出目录必须持久化。** 当前 PBS 模板把 `--commit` 源码展开到计算节点临时目录，并在退出时删除；因此 compute 命令的 `--out` 一律使用 `/work/xg24i002/x10041/LoopLora/...` 下的绝对路径。相对 `runs/...` 会写入临时快照，不能依赖它保留结果。[R30] 本节路径来自原实验环境；迁移机器时应统一修改显式 persistent root。

### 13.2 S0 配置示例

以下 YAML 是建议的 schema。所有 `null` 待填字段必须在相应阶段 release 前解析；正式 runner 遇到未解析字段应 fail closed，而不是自动猜值。

```yaml
schema: slot_loop_v1
study_id: slot-loop-v1
phase: S0
status: proposed

provenance:
  evidence_commit: a255e814740ff85ab3b4ec4836c88f0a8f802b2d
  implementation_commit: from_submission
  manifest_path: reports/slot-loop-v1/manifest.lock.json
  digest_algorithm: blake2b-128

parent:
  kind: sq2
  arm: S-RE
  inference_path: /work/xg24i002/x10041/LoopLora/runs/p1l/sq2/arms/S-RE/writer.pt
  checkpoint_path: /work/xg24i002/x10041/LoopLora/runs/p1l/sq2/arms/S-RE/checkpoint.pt
  inference_blake2b128: null
  config_hash: 1bfb72b7d87bc0aff743c1e018e8680e
  loader: looplora.llm.sq2.load_inference

protocol:
  backbone: Qwen3-0.6B-Base
  information_access: memory_only
  task_schema: relations_t1
  facts: 16
  relations: 4
  head_sharing: 0.3
  loop_steps: 1
  read_scope: all_valid_question
  answer_metric: full_vocab_top1
  support_cache_visible_to_query: false
  question_write: false
  online_write: unsupported

memory:
  key_dim: 768
  value_dim: 128
  feature_map: native_dpfp_nu3
  query_transform: native_query_code
  state_dtype: float32
  slot_capacity: 16
  overflow: error
  empty_read: zero
  owner_indexing: original_episode_row
  arms: [native, matrix, soft_slot, hard_slot, clean_final_diagnostic]

calibration:
  temperature_grid: [0.03, 0.1, 0.3, 1.0]
  gain_grid: [0.5, 1.0, 2.0]
  metric: nll
  tie_break: [gain_nearest_one, temperature_larger]
  fit_per_example: false
  report_uncalibrated: true

data:
  root_seed: 20261029
  train_seed_fields: [study_id, train, phase, run_seed]
  eval_seed_fields: [study_id, purpose, phase]
  eval_shared_across_arms_and_run_seeds: true
  generator_version: null
  layout_digest: null
  resolved_manifests: null
  smoke_episodes: 64
  calibration_episodes: 2048
  dev_episodes: 1000
  confirmation_episodes: 2000
  confirmation_access: closed

execution:
  nodes: 1
  max_walltime_seconds: 1200
  backend_only_resource_claim: false
  allow_overwrite: false
```

`parent.kind` 应决定显式 loader；不要把它转换成普通 reader 的 `writer_checkpoint`。`protocol.task_schema` 是新 runner 的字段，不要直接映射为旧 `stage='writer'` 后再期望任务名生效。

`implementation_commit: from_submission` 在运行记录中解析为 submit 提供的不可变源码 commit；不要把“包含本配置的新 commit 哈希”硬编码回同一 commit 的配置内，形成自引用。manifest/config 的摘要写入运行 sidecar；若文档内部也带摘要，需定义排除自身摘要字段的规范化规则。

### 13.3 S1A/S1B 的最小覆盖项

| 字段 | S1A | S1B |
| --- | --- | --- |
| phase | S1A | S1B |
| task_schema | relations_t1 | 明确登记的 relations_t2_chain_end |
| facts/relations | 16/4 | 原 T2 E12、关系恒0布局 |
| arms | matrix_L1、soft_slot_L1 | matrix_L1/L4、soft_slot_L1/L4 |
| parent | 均为同一 S-RE | 均为同一 S-RE |
| trainable | Q/R/U/B | Q/R/U/B |
| run_seeds | 17/29/43 | 17/29/43 |
| loss | answer_ce | answer_ce |
| read_scope | all_valid_question | all_valid_question |
| gain/temperature | 引用 S0 锁值 | 本阶段 calibration 锁值；L1/L4共用 |
| optimizer | fresh AdamW | fresh AdamW |
| max_steps | release 前锁定，≤6000 | release 前锁定，所有四臂相同 |
| checkpoint_rule | 相同频率，dev NLL最低 | 相同频率，dev NLL最低 |
| confirmation | 本阶段独立 manifest | 本阶段独立 manifest |

S2 另建配置，不能仅修改 `loop_steps`：它还改变读取位置、状态结构、任务语义和监督。第一版可将 Role/Step/NULL 损失权重分别固定为 0.1/1.0/0.1；各损失先按有效样本/步骤平均，Role 使用 batch 内不同实体作为负例、温度0.1。Step 可采用归一化 code 的余弦距离，并由 Role 对比损失防止坍缩。这些是起始设计值，需在 S2 登记中锁定；若做梯度尺度 pilot，调整后的规则应在查看 S2 主 dev/test 前明确。

### 13.4 实现后的运行入口

**元数据准备：** 仅执行标准库配置/路径/摘要处理，不能导入 torch、加载模型或运行实验。

```bash
python3 scripts/prepare_slot_manifest.py \
  --config configs/slot_loop_v1/s0.yaml \
  --out reports/slot-loop-v1/manifest.pending.json
```

把 pending manifest、阶段配置和实现一并提交；确认目标 commit 包含这些文件，再更新 IMPL_COMMIT。只有存在于该不可变提交内的相对输入路径，才会被 preflight 的源码快照读取。

**开发测试：** 使用工作树仅做测试；下列新测试文件需要先实现。

```bash
python3 scripts/submit.py \
  --name slot-tests \
  --queue debug-g \
  --nodes 1 \
  --walltime 00:15:00 \
  --worktree . \
  --wait \
  -- -m pytest -q tests/test_slot_backend.py tests/test_slot_experiment.py
```

**计算节点 preflight：** 校验模型、数据生成和读取路径。`NEW_IMPLEMENTATION_COMMIT` 是占位符，必须替换为新实现的完整 commit。

```bash
IMPL_COMMIT=NEW_IMPLEMENTATION_COMMIT

python3 scripts/submit.py \
  --name slot-preflight \
  --queue debug-g \
  --nodes 1 \
  --walltime 00:15:00 \
  --commit "$IMPL_COMMIT" \
  -- -m looplora.llm.slot_run \
     --mode preflight \
     --config configs/slot_loop_v1/s0.yaml \
     --manifest reports/slot-loop-v1/manifest.pending.json \
     --out /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/preflight
```

preflight 产出 `preflight.json`（身份、episode IDs、已通过的检查）与 `identity.json`（旧路径核对详情）。随后由轻量 manifest 脚本读取这份持久化结果生成锁定文件：

```bash
python3 scripts/prepare_slot_manifest.py \
  --config configs/slot_loop_v1/s0.yaml \
  --preflight-report /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/preflight/preflight.json \
  --out reports/slot-loop-v1/manifest.lock.json
```

将 lock 文件及解析后的配置再次提交，更新 release commit。calibration 的选择结果进入下一份 release。未完成这些步骤时，`--mode train/evaluate` 必须拒绝运行。

**S0：** preflight 通过，且新配置/锁定 manifest 已提交后执行。

```bash
IMPL_COMMIT=NEW_RELEASE_COMMIT

python3 scripts/submit.py \
  --name slot-s0 \
  --queue debug-g \
  --nodes 1 \
  --walltime 00:20:00 \
  --commit "$IMPL_COMMIT" \
  -- -m looplora.llm.slot_run \
     --mode s0 \
     --config configs/slot_loop_v1/s0.yaml \
     --manifest reports/slot-loop-v1/manifest.lock.json \
     --out /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s0
```

**单个训练 arm：** 本例是目标 CLI，先由 release 写定 steps、学习率和 calibration 值；其余 arm 采用相同结构的命令。新增 release commit 后同步更新 IMPL_COMMIT。

```bash
python3 scripts/submit.py \
  --name slot-s1a-s17 \
  --queue regular-g \
  --nodes 1 \
  --walltime 00:40:00 \
  --commit "$IMPL_COMMIT" \
  -- -m looplora.llm.slot_run \
     --mode train \
     --config configs/slot_loop_v1/s1a.yaml \
     --manifest reports/slot-loop-v1/s1a-manifest.lock.json \
     --release reports/slot-loop-v1/s1a-release.json \
     --arm soft_slot_L1 \
     --seed 17 \
     --out /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a/soft_slot_L1/s17
```

**锁定后的评估与汇总：** `evaluate` 和包含 bootstrap 的正式汇总均通过 submit.py 在 compute node 运行；登录节点只处理小型元数据和已生成表格的格式化。评估 CLI 应显式要求 `--split confirmation` 与对应 release，验证 checkpoint 摘要和已锁定分析版本。

```bash
python3 scripts/submit.py \
  --name slot-analysis \
  --queue debug-g \
  --nodes 1 \
  --walltime 00:15:00 \
  --commit "$IMPL_COMMIT" \
  -- scripts/analyze_slot_study.py \
     --stage S1A \
     --manifest reports/slot-loop-v1/s1a-manifest.lock.json \
     --records /work/xg24i002/x10041/LoopLora/runs/slot-loop-v1/s1a \
     --out /work/xg24i002/x10041/LoopLora/reports/slot-loop-v1/s1a-summary
```

分析脚本必须生成 source-index，指向实际 run records/预测文件并记录摘要；需要进入 git 的小型 records/指标应显式归档到 reports。不能假设相对 reports 路径自动镜像了 runs 中的输出。

当前 submit.py 的 `--` 后面接 Python 参数，如 `-m module`，不是再嵌一层 `python3`；不假设它有 `--dry-run`。遵循项目的单节点、walltime 和 scale-up 规则。不要复制本节命令就自动提交全部实验。

---

## 14. 预算、时间与职责安排

### 14.1 初始预算是上限，不是耗时预测

建议先为 S0–S1B 设置 **18 node·h 的分阶段总上限**，所有作业单节点。这是本计划的预算提案，不能视为已经获得的实际运行时间或必须花完的配额。S2/S3 不包含在内，要由前序结果决定是否另立预算。

| 批次 | 预算上限建议 | 放行条件 |
| --- | ---: | --- |
| 接口测试、preflight、S0 | 1.0 node·h | 先通过语义检查，再比较性能 |
| S1A pilot + 2臂×3seed | 4.5 node·h | 锁定训练规则、初始参数与数据 |
| S1B pilot + 4臂×3seed | 11.5 node·h | 单跳读取问题得到足够定位，形成独立新研究登记 |
| 评估、统计与必要补测 | 1.0 node·h | 只解决明确的剩余风险 |
| 合计 | 18.0 node·h | 不自动延长，不包含S2/S3 |

S1A 的 6,000 步上限不是无条件承诺。先测加载开销、100 个训练步和一批完整评估，按最慢 arm 的步时推算各臂相同的训练步数，预留保存/评估余量。若时间不足以达到事先定义的有意义步数，应缩小实验或采用登记的分段恢复；不让每臂在 walltime 结束时各跑不同步数，却声称样本预算相同。

现有 submit 脚本通常限制单作业 1 小时；有已有结果支持且提供 scale-up 文档时才可使用其扩展规则。本计划默认 15–55 分钟单节点作业，不自行开启多节点或突破项目规则。[R23]

### 14.2 工作分工

| 角色 | 负责内容 | 交付与复核 |
| --- | --- | --- |
| 研究负责人 | 假设、数据、主对比、预算和 gate | 锁定 prereg/release；保留旧 verdict |
| 实现者 | backend、loader、owner、mask、冻结与导出 | 提交代码和针对性测试结果 |
| 实验执行者 | 资产核查、单节点作业、异常记录 | run record、样本预测、checkpoint、成本 |
| 分析复核者 | 重新生成表格、检查配对、核对结论 | 独立生成 summary 与 decision |

小团队可由同一个人兼任，但建议在“开封确认集”和“写最终结论”之前做一次独立复核。本文已经做过计划层面的交叉审查；这不等同于实现或实验已经通过验证。

### 14.3 每阶段的结束条件

| 情况 | 处理 |
| --- | --- |
| 旧路径/数据/因果性无法验证 | 停止性能比较，先修复工程问题 |
| S0 无提升但接口正确 | 允许一次限额 S1A；不据此断言理论失败 |
| S1A 同预算训练无收益，有限接口消融也无收益 | 停止继续扩大此 slot 方案；提交负结果与定位结论 |
| S1B 只有存储收益、没有独立循环收益 | 将论文/项目主张收缩为读取或记忆适配 |
| S1B 有循环收益，但短跳损失或成本过大 | 优先 S2 的转移与终态，不先扩大模型/任务矩阵 |
| S2 只有额外监督带来收益 | 明确归因为监督和结构条件，不宣称无监督涌现推理 |
| S3 成本优势不存在 | 保留科学结论，放弃实用加速/省内存的过度表述 |

---

## 15. 交付目录、记录格式与可复现要求

以下是**目标交付树**；本文没有在原仓库创建这些文件。

| 路径 | 内容 |
| --- | --- |
| `docs/prereg/SLOT-LOOP-v1.md` | 本计划落实后的正式研究定义与差异说明 |
| `configs/slot_loop_v1/` | 每阶段/arm配置，含所有默认值 |
| `reports/slot-loop-v1/manifest.lock.json` | 代码、checkpoint、生成器、数据ID与分析版本 |
| `reports/slot-loop-v1/checkpoint_inventory.json` | 权重来源、格式、摘要、loader、复现状态 |
| `reports/slot-loop-v1/preflight/` | identity、mask、状态隔离、字节和冻结验证 |
| `reports/slot-loop-v1/s0/` | 全部校准候选、dev预测、诊断结果和决定 |
| `reports/slot-loop-v1/s1a-summary/`、`s1b-summary/` | 每arm/seed记录索引、归档指标、汇总和区间 |
| `runs/slot-loop-v1/` | 原环境中的checkpoint与完整训练日志，按仓库现行存储策略管理 |

### 15.1 每次 run 的最小记录

```json
{
  "study_id": "slot-loop-v1",
  "stage": "S1A",
  "arm": "soft_slot_L1",
  "run_seed": 17,
  "parent_writer_count": 1,
  "source_commit": "REQUIRED",
  "manifest_blake2b128": "REQUIRED",
  "parent_checkpoint_blake2b128": "REQUIRED",
  "status": "planned",
  "planned_steps": null,
  "actual_steps": null,
  "read_scope": "all_valid_question",
  "loop_depth": 1,
  "key_dim": 768,
  "value_dim": 128,
  "slot_capacity": 16,
  "trainable_parameter_names": ["Q", "R", "U", "B"],
  "frozen_buffers_verified": false,
  "confirmation_accessed": false,
  "metrics_path": null,
  "samples_path": null,
  "cost_path": null,
  "deviations": []
}
```

`planned` 不能被分析脚本当作完成记录。退出失败、抢占、中断和重启必须保留 attempt 信息；成功续跑也保留累计步数、optimizer恢复情况和数据游标。默认拒绝覆盖已有输出。

### 15.2 样本级与资源记录

每个样本至少记录 episode ID、question ID、split、arm/seed/checkpoint、目标/预测、NLL、hop标签（仅指标）、实际逻辑读取数、是否早停/预算耗尽，以及可选的轨迹摘要。未经必要限制的完整支持内容不必重复存入每个样本行。

资源表至少包含 trainable/stored 参数、持久 memory bytes、控制状态 bytes、准备/训练/推理峰值、prefill/问答时间、硬件、dtype、batch与padding统计。明确是否保留另一种 memory 或历史 KV。

### 15.3 最终报告必须同时回答

1. 新方案相对哪个同起点、同信息访问的 baseline 改善了多少？
2. 改善来自读取算子、跨步 query、额外监督还是参数/计算变化？
3. 哪些负结果和已注册门槛没有通过？
4. 独立 L1、同 checkpoint depth sweep 和中间态干预是否一致？
5. 结论条件于几个 writer parent、哪些名字池、哪些任务和预算？
6. 实际状态和端到端耗时是否支持实用主张？

---

## 16. 执行前勾选表

- [ ] 已核对当前仓库 HEAD 与本文审查 commit 的差异，补读新增相关结果。
- [ ] S-RE/D128-3 的真实 checkpoint、摘要、配置和 loader 已验证。
- [ ] 新 runner 没有直接误用 `rfx.FinalRead` 或普通 writer checkpoint 格式。
- [ ] S0 保持 L1、all-valid-question，gold value 仅用于明确的 final-only 诊断。
- [ ] 数据 manifest 锁定，eval episodes 跨 arm/seed 相同，train/eval 不交叉。
- [ ] 主比较采用完整词表 EM，calibration/dev/confirmation 的用途分离。
- [ ] Matrix/Slot 的 state、owner、padding、空行和容量边界均通过检查。
- [ ] 冻结 writer 包括共享 A 与预处理 buffers，reader 梯度仍可通过冻结 backbone。
- [ ] τ/g 的校准和继承规则明确，checkpoint 评估时不重选。
- [ ] 独立训练 L1/L4 与同 checkpoint depth sweep 分开。
- [ ] S2 的架构、位置、监督和 NULL 逐项控制；推理不接收 gold path/true-hop。
- [ ] read 深度、位置数、实际调用与终止检查的额外读取分别计数。
- [ ] 同内容与固定字节预算实验分开，溢出策略不会静默改变。
- [ ] 所有含 torch 的工作在计算节点执行，预算与源码 commit 已登记。
- [ ] 新阈值没有覆盖旧 SQ2/LSQT/P1L 的历史判定。
- [ ] 报告明确区分已测量结果、数学推导、设计假设和后续计划。

**最终交付标准：** 另一位执行者仅依赖已提交的配置、manifest、真实 checkpoint 和本研究代码，就能重跑表格并得到相同判定；无需依靠口头约定补全读取位置、数据种子、停止语义或比较对象。

---

## 17. 来源与核查入口

R 类链接固定到本文审查 commit。报告说明“观察到了什么”；代码说明“实际执行了什么”；本计划的新阈值、API和预算是建议，不应反向当作这些来源的结论。

| 编号 | 来源 | 用途 |
| --- | --- | --- |
| [R01] | P1L 报告 | oracle reader、r32/r128和上下文协议 |
| [R02] | P5L 报告 | 1.7B 复现与每hop表现 |
| [R03] | SQ2 报告 | DPFP ridge/coef、主要数值、旧gate |
| [R04] | DPFP 报告 | D128-3与负载/表示结果 |
| [R05] | B8W 报告 | 同函数初始化的共享/非共享对照 |
| [R06] | LSQT 报告 | 从头LSQ与注册的中间区间 |
| [R07] | RFX 报告 | 冻结checkpoint的store/query干预 |
| [R08] | DPFP-X 报告 | 更大rank/seed的后续边界 |
| [R09] | LSR 报告 | 停止规则、短长跳权衡与额外前向 |
| [R10] | TSM 报告 | 已有终态监督与gate结果 |
| [R11] | P1L-C 报告 | 读取位置消融与因果干预边界 |
| [R12] | CTC 报告 | 解析器/角色结构的特权信息 |
| [R13] | WP-L6 报告 | 已尝试的后续writer路线 |
| [R14] | 2026-10-29 followup | 当天研究状态和下一步约束 |
| [R15] | notation | 仓库术语/记号 |
| [R16] | core/adapters.py | 实际循环、query_code与读取实现 |
| [R17] | llm/train.py | writer冻结集合、任务构造与checkpoint格式 |
| [R18] | SQ2-state | S-RE实际运行目录与资产 |
| [R19] | sq2_release.py | D128-3父checkpoint元数据与摘要 |
| [R20] | rfx.py | 旧FinalRead支持范围 |
| [R21] | read_fix_run.py | 旧训练/验证入口、checkpoint导出 |
| [R22] | sq2.py | S-RE loader |
| [R23] | scripts/submit.py | 当前作业提交参数与限制 |
| [R24] | data/relations.py | episode身份与生成方式 |
| [R25] | writer_fix.py | 标准化buffers和实际写入K/V |
| [R26] | core/wrapper.py | MemoryContext与基础分支 |
| [R27] | llm/protocols.py | SupportSnapshot和信息访问 |
| [R28] | llm/inject.py | AdapterBank状态操作 |
| [R29] | llm/batching.py | 问题展开和repeat_snapshot |
| [R30] | scripts/pbs/job.pbs | 不可变源码快照、临时目录与输出持久化 |
| [L01] | End-To-End Memory Networks，2015 | 多跳显式记忆的先例 |
| [L02] | Key-Value Memory Networks，2016 | 分离key/value的记忆先例 |
| [L03] | Differentiable Learning of Logical Rules for Knowledge Base Reasoning，2017 | 关系组合与可微规则学习先例 |

[R01]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/P1L.md
[R02]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/P5L.md
[R03]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/SQ2.md
[R04]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/DPFP.md
[R05]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/B8W.md
[R06]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/LSQT.md
[R07]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/RFX.md
[R08]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/DPFP-X.md
[R09]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/LSR.md
[R10]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/TSM.md
[R11]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/P1L-C.md
[R12]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/CTC.md
[R13]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/WP-L6.md
[R14]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/plan/followup_2026-10-29.md
[R15]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/notation.md
[R16]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/core/adapters.py
[R17]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/train.py
[R18]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/docs/reports/SQ2-state.md
[R19]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/sq2_release.py
[R20]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/rfx.py
[R21]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/read_fix_run.py
[R22]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/sq2.py
[R23]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/scripts/submit.py
[R24]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/data/relations.py
[R25]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/writer_fix.py
[R26]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/core/wrapper.py
[R27]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/protocols.py
[R28]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/inject.py
[R29]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/src/looplora/llm/batching.py
[R30]: https://github.com/UnbearableFate/LoopLora/blob/a255e814740ff85ab3b4ec4836c88f0a8f802b2d/scripts/pbs/job.pbs
[L01]: https://arxiv.org/abs/1503.08895
[L02]: https://arxiv.org/abs/1606.03126
[L03]: https://arxiv.org/abs/1702.08367

