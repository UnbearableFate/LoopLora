# 循环低秩适配与上下文记忆：完整研究计划

**本文档为事前制作的参考文档,并非权威, 实际执行中应当根据实际情况调整计划并对本文档进行更新**

## 文档信息

- 版本：v1.1（2026-10-26 执行中更新；v1.0 为事前版本）
- 编写日期：2026-10-26
- 文档性质：研究提案与实验执行协议；本文中的候选模型尚未完成训练或性能验证。
- 工作题目：固定记忆预算下的低维迭代查询与参数高效适配
- 英文工作题目：Iterative Low-Rank Adaptation under a Fixed Context-Memory Budget
- 研究简称：循环低秩适配。暂不确定正式方法名称，避免与已有 RecLoRA 混淆。
- 范围：冻结预训练主干，只训练低秩侧分支及小型记忆读写模块；主干的每个原有算子每次前向只执行一次。
- 证据基线：用户提供的 RecLoRA 11 页原文，以及文末列出的第一手论文资料。文献边界以本次核查版本为准，不构成“全球首次”的保证。
- 执行顺序：机制验证 → 预算对照 → 端到端适配 → 停止机制与工程优化。初期不同时引入多个未验证组件。

## 阅读导航

- 研究目标与已有工作边界：第 0–3 节。
- 模型、训练与理论分析：第 4–6 节。
- 数据、基线、统计与工程：第 7–10 节。
- 阶段门槛、风险和执行任务：第 11–15 节。
- 参考伪代码、起始配置、主张登记与文献：附录 A–D。
- 实际执行状态与相对原计划的调整：下方“执行记录与计划调整”（持续更新，优先于正文中的同名设定）。

---

## 执行记录与计划调整（持续更新）

### E.1 约束与环境（2026-10-26）

- 资源上限（用户指定，当前阶段）：单个实验的训练资源 ≤ 8 节点 × 1 小时；由 `scripts/submit.py` 在提交时检查，实际 walltime 按实测 ×1.5 申请。2026-10-26 用户补充：当已获得的实验结论支持扩大实验时，单次实验预算可扩至 8 节点 × 2 小时（`submit.py --scale-up <决策记录或预注册>`，理由写入该文档）。
- 平台：Miyabi-G（GH200，aarch64），组 xg24i002；torch 2.14.0+cu132、transformers 4.57.6，uv 管理（`pyproject.toml`/`uv.lock`）。所有测试与实验经 PBS 在计算节点运行。
- 系统计划停机：2026-10-28 09:00 JST；之前提交的作业须在此前结束。
- 执行组织：Claude Code 作为 Orchestrator；codex worker（2026-10-26 19:36 前启动的为 GPT-6-Astra xhigh，此后新开 session 为 GPT-6.1-sol xhigh）经 herdr 在独立 tab 与 git worktree（`wp/*` 分支）中实现子任务；worker 为固定角色、长生命周期（`docs/workers.md`），按需清理并重新注入上下文；交流以 `docs/tasks/`（任务书）与 `docs/reports/`（报告）为准。
- 代码约定：不使用 SHA-256，内容哈希统一用 blake2b；随机种子由 `derive_seed(base, label…)` 派生。

### E.2 相对原计划的调整及理由

| 项目 | 原计划 | 调整 | 理由 |
| --- | --- | --- | --- |
| 首轮 LLM | Qwen/Qwen3-0.6B | Qwen/Qwen3-0.6B-Base，revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | 模板化关系问答不需要 chat template；Base 模型已在共享缓存中，可固定 revision。确认规模候选（P5 再定）：Qwen3-1.7B-Base（已缓存）或 Qwen3-4B-Base |
| P1 结构 | 单阶段 | P1-S（纯符号、单模块、小型 reader）→ P1-L（Qwen3-0.6B-Base 单模块、模板文本、memory-only 协议） | 按 §7.5 的难度递增；符号阶段成本极低，可先隔离 reader 机制并给出多 seed 结论 |
| T2 查询形式 | “起点+关系序列”或“沿某关系走 h 步” | 主任务 `t2_chain_end`（沿关系走到无出边的终点，h 不写入查询）；第二设置 `t2_walk`（查询写明 h）；干扰设置 `t2_typed` | 初版 reader 没有 step embedding，无法在 L>h 时自行计数停止；走到终点的任务对 L 单调、可测训练外深度。`t2_walk` 用作 Gate 1 的第二独立设置并检验计数需求 |
| 机制阶段 writer | 单跳预训练后冻结 | 增加 oracle writer（随机/正交编码）与联合训练对照，三种条件分别报告 | 线性 writer 从同一 u 产生 key 与 value，u 必须把 head 与 tail 信息放在不同子空间；单跳训练没有让 value 子空间对齐 key 子空间的压力（§5.2 已预见），需要 oracle 条件区分“reader 无效”与“writer 不适合组合” |
| 符号阶段读出 | 未定义 | 主指标在本 episode 候选实体集合上取 argmax；全词表准确率为副指标；符号基础输出 W=0 | 避免 r 维瓶颈对 1024 类全词表分类的投影损失混入读机制结论 |
| 主读取器的激活 | V2 使用 ReLU（附录 A/B） | P1-S 中 M1（ReLU）与 M1-lin（恒等激活）并列为主要读取器，各配同激活的固定 query 对照（B5 / B5-lin），通过判定用 Bonferroni 97.5% 区间（P1S.md 修订 A1，在读取任何 pilot 结果前作出） | `docs/theory.md` 命题 6：无偏置、宽度 r 的 ReLU 无法把正负号混合的编码经 Qφ(U·) 原样传给下一次查询；只测 ReLU 会把激活缺陷与读取机制混为一谈 |
| P1-L 的启动时机 | P1-S 通过 Gate 1 后才开始 P1-L | P1-L Pilot-1（LLM writer 可行性，仅验证集）与 P1-S 并行；Pilot-2 与正式运行仍以 P1-S 结论为条件 | Pilot-1 不依赖 P1-S 结果；系统停机（10-28 09:00）前的时间有限，并行可节省一个排队周期 |

### E.3 任务状态

| 任务 | 状态 |
| --- | --- |
| T-01 符号表 | 完成（2026-10-26）：`docs/notation.md` |
| T-02 参考实现与 Gate 0 测试 | 完成（WP-A，`docs/reports/WP-A.md`）：7 种变体、函数式 delta 记忆、写前快照 scan、Appendix-A oracle、混合精度 wrapper、记账与收缩界；304 项测试在计算节点通过 |
| T-03 关系数据生成器 | 完成（WP-B，`docs/reports/WP-B.md`）：t1/t2_chain_end/t2_walk/t2_typed/t3_update/t3_missing 与三类反事实；105 项测试通过；≥ 20k episodes/s；补充反事实 `swap_successors`（保持终点集合、改变正确终点，区分“沿链走”与“猜终点”；WP-B2，合并 bbe2a24，120 项测试通过） |
| T-07 LLM 注入与吞吐 pilot | 完成（WP-E，`docs/reports/WP-E.md`）：Qwen3-0.6B-Base 注入、memory-only/context-retained 协议、teacher forcing 与增量解码一致、1024 个单 token 名字；54 项测试通过；吞吐 pilot 见报告 |
| T-04/T-05 符号机制流水线与首轮实验 | 流水线完成（WP-D，c86d72b）；验证集 pilot 完成（WP-D2，633f37c，`docs/reports/P1S-pilot.md`）：T1 learned writer 单跳 val 最高 84.7%（r=32,E=12），按规则 E*=12、链 (8,4) 并标记；oracle_orth 下 M1-lin 在 hops 1–3 达 94%、M1 68%，单次读取变体恰为 33.3%（只答对 hop 1）；learned writer 条件下所有变体（含 B4）在各 hop 约 41–50%，呈“在两个链终点间猜测”的捷径特征。预注册定稿 v1（修订 A2 步数 20000、A3 walk-specificity 判据、A4 主矩阵）；正式运行进行中（WP-H）；同时预注册后续研究 F1（`docs/prereg/P1S-F1.md`，r=r_m∈{64,128}，20000 步）与 F2（`docs/prereg/P1S-F2.md`，分解式 learned writer，r=32，15000 步；修订 A6 记录对“仅一个后续研究”规则的偏离），均在任何确认集结果之前登记。F1 完成（合并 7155dc1，`docs/reports/P1S-F1.md`，4.18 node·h）：joint writer 在 r=64、128 下 M1 与 M1-lin 均通过 G1/G2（如 r=64 M1：Δ_reach +30.5 pp [97.5% CI 29.2, 31.9]，walk specificity +20.9 pp，hop-1 +9.0 pp，对 B5 +24.0 pp）；swap_successors 跟随率 59% 对未交换 36%；oracle_random 在 r=64、128 下 reach +65–85 pp，但 L=4 时 hop-1 下降 17–33 pp（越过终点后继续读取，未停止），因 hop-1 判据未通过 G1。WP-H 正式运行按预注册预算规则改为 15000 步（预计 7.3 node·h），F2 使用相同步数。**P1-S 正式结果（WP-H，合并 758c06a；决策记录 `reports/decisions/P1S.md`）**：r=32、15000 步，M1 在 oracle_orth（Δ_reach +92.9 pp）与 joint（+8.3 pp [97.5% CI 7.4, 9.3]，walk specificity +2.5 pp，对 B5 +7.8 pp）下通过 G1/G2；M1-lin 在四种条件下均未通过 G1（oracle 条件为越过终点导致 hop-1 下降；joint 为猜终点）；t1_frozen（单跳预训练后冻结，原计划协议）未通过（typed 复现的 walk specificity 不显著）；固定 query 与单次读取只能答对 hop 1，无记忆循环为随机水平。结论：继续 P1-L，携带 M1 与 joint writer；t1_frozen 不再携带；r 由 P1-L 验证 pilot 在 {32, 64} 中确定（r=32 的 joint 效应较小且随 seed 变化，r=64 稳健）。WP-H 4.44 node·h，P1-S 合计 8.48 node·h。**F2 完成**（合并 4bafd71，1.73 node·h，`docs/reports/P1S-F2.md`）：分解式 writer（key 只见 head/relation，value 只见 tail）在 r=32 下去除了“猜终点”捷径并逐跳行走（joint_fact M1：L=1 时 hop 1–4 为 86/2/1/1%，L=4 时 50/32/16/5%；Δ_reach +23.0 pp，walk specificity +20.2 pp，三个 seed 一致；swap_successors 跟随率 48% 对未交换 4%），但与 oracle_random 一样越过链尾，hop-1 下降 28–37 pp，四行均仅因 hop-1 判据未通过 G1。F2 预注册的“不启动 P1-L”规则以“learned 码不能组合”为前提，与 WP-H joint/M1 通过及 F2 自身结果矛盾；按决策记录 §7 的偏离说明，P1-L 是否启动仍按 P1-S 决策，F2 只决定 writer 形式（分解式不作为默认，保留为 LLM writer 修复候选）。对越过链尾的 walk 型编码，停止机制（P4）成为主要剩余问题 |
| T-04/T-05 LLM 单模块流水线 | 流水线完成（WP-G，合并 2bf5d6d；`docs/reports/WP-G.md`）：writer/冻结 reader/联合训练、B0/B1/B2 基线、全词表 EM、反事实、确定性 GPU 执行；611 项测试与 14 个 smoke 运行通过；实测 280–550 s/千步（batch 8）。P1-L 预注册草稿 `docs/prereg/P1L.md`，待 P1-S 结论后定稿。Pilot-1（WP-L1，合并 9eb0e10，`docs/reports/P1L-pilot1.md`，1.09 node·h）：layer-14 `o_proj` 单模块、r=32、memory-only 的 T1 writer 在 1500 步×batch 32 内未学会单跳检索（val 全词表 EM ≤ 0.3%，候选准确率约为随机水平，loss 约 200 步后停在 7.3–7.5）；按预注册规则，P1-L 的 T2 运行暂停，先做 writer 诊断（WP-L2：输入位置的信息探针、输出通路 oracle、训练预算、输入位置替代）。WP-L2 完成（合并 44b27cf，2.07 node·h，`docs/reports/P1L-diagnosis.md`）：(a) 信息存在——rank-32 探针在 tail token 处可解码 tail 97.6%、问题末位置可解码起点 99.4%、关系 100%；(b) 输出通路可用——直接给出答案编码时 lr 3e-3 下 1500 步 val EM 98.7%（lr 3e-4 仅 20%）；(c) 预算不是原因——6400 步（Pilot-1 的 4.3 倍）、两种输入位置（o_proj 输入与 layer-14 残差流）均无学习（loss 停在名字先验 ≈7.2–7.5，候选准确率为随机水平）；(d) key–query 匹配从未形成（source-fact top-1 6–7.5%，随机 6.25%）。相比之下符号 T1 writer 约 150 步即离开平台期。下一步 WP-L3：在同一 T1 设定下并行筛选 writer 修复（lr 3e-3、输入标准化、key 取 tail 前位置/value 取 tail 位置的分解式 writer、辅助寻址损失、辅助 value 损失及组合），并以“oracle 寻址”臂区分失败环节是寻址还是 value 通路；规则预先写定：最简单且 val EM ≥ 0.80 的方案进入 P1-L，全部 < 0.30 则 P1-L 停止并报告负结果。同时 WP-C5 实现精确等价的快速写入扫描（当前每步约 0.5 s 且与 batch 无关，为开销主导）。WP-C5 完成（合并 7f68a3c，0.16 node·h，`docs/reports/WP-C5.md`）：只保留最终状态的打包写入扫描，loss 与参考实现一致；T1 训练吞吐提升 1.16–2.06 倍（batch 128 单进程 231 → 373 episodes/s）；剩余开销中 CPU 端的样本生成与渲染约占 154/343 ms，由 WP-B4 改为后台预取（不改变数据）。WP-B4 完成（0.44 node·h，`docs/reports/WP-B4.md`）：有序后台预取，训练轨迹与原实现逐位一致；batch 128 单进程 379 → 673 episodes/s（相对 Pilot-1 的 93.5 episodes/s 约 7 倍） |
| §6 理论分析（命题、证明、反例、预测） | 完成（WP-T，`docs/theory.md`，`docs/reports/WP-T.md`）：9 个命题；要点：线性 reader 的精确链终点构造需 ργ>1（ρ=0.25 时 γ>4），不是欧氏收缩；单次读取在对齐首跳读取下 h≥2 的条件上界 1/(N−2)，但无约束 Q 无普适上界；无偏置 ReLU 的 Qφ(Uv) 像位于 r 生成元锥内，不能对正张成全空间的编码做恒等传递（宽度 2r 可以，单独隐藏偏置不行）；双 BA 在 B=0 一阶梯度为零，skip 形式可启动。36 项确定性检查在计算节点通过 |
| T-08 真实任务处理 | 完成（WP-B3，合并 ba2b2e8，`docs/reports/WP-B3.md`）：MuSiQue-Ans v1.0（HF `dgslibisey/MuSiQue` @ c8f4f8c9，条目数与论文一致，文件 blake2b 已记录）；划分 train 18,263 / val 965 / test 736 / confirm 1,492（上下文上限 640 token，丢弃数按 hop 类型记录，与模型输出无关）；仅给定支持段落且随机顺序；官方答案 EM/F1；`answer_substitution` 与 `remove_final_support` 反事实；保留能力子集（ARC-E、PIQA、HellaSwag、WinoGrande、MMLU 各 500，ARC-C 299，均为验证集）。P3 是否运行仍由门槛决定 |
| §6 补充：learned writer 的终点集合捷径 | 完成（WP-T2，合并 0c63cdc，`docs/theory.md` 命题 10）：共享关系分量的 key 使 delta 存储近似为 value 的 EMA；λ≈1、β 小时可近似望远镜求和 Σ(终点−起点)，构造的 r=32 例子与深度无关（实测 β≈0.565 不在该构造的区间内）；理想“猜终点”预测各 hop 平坦的 50%/50%/33.3%（三种布局），swap_successors 配对提升为 0；value 仅含 tail 可去除直接的 head 相减，但不能普遍排除终点推断；A3 能拒绝已列出的通用过滤策略，但 A3 与 swap 均不能识别所有内部算法。LLM 对应设计：key 取 tail 之前的前缀位置，value 取 tail 的原始 token embedding（仅为预测） |
| T-09 停止机制（P4-S pilot） | 已开始（WP-C3，`docs/reports/P4S-stopping.md`，仅 screening、无显著性主张，0.18 node·h）：在 24 个 F1 checkpoint 上评估无标签停止规则（阈值在 val 上选定）。相对状态变化停止在 oracle_random r=128 上部分恢复越过终点造成的短 hop 损失（M1-lin hop-1：L=4 81.5% → 96.6%），同时提高 hop 4–8（44.9% → 64.4%），但不一致（r=64 M1-lin 变差）；参考实现中自适应停止不比固定 L=16 快；§6.3 收缩证书在 96000 个 episode 中无一满足 c<1（界 4.6–37.1，命题 7 的预测）；真实 hop 停止不是上界（学到的更新步数不等于图 hop 数）。WP-C4（合并 ecf397f，0.02 node·h，`docs/reports/P4S-stopping-r32.md`）：同样的三种无标签规则用于 r=32 learned 码（F2 分解式与 WP-H joint）：分解式 walk 型编码的 hop-1 仍损失 25–42 pp（reach 与 walk specificity 保持为正），无一满足“hop-1 不降超过 1 pp”；未分解 joint/M1 实际停在 L≈16，没有停止收益。结论：现有无标签规则不能解决 learned 码越过链尾的问题，不做正式 P4 检验；后续若继续 P4，需要训练得到的停止机制（§4.7 learned router），排在 P1-L 之后 |
| T-06 预算匹配（P2） | 符号阶段 P2-S 已预注册并开始（`docs/prereg/P2S.md`，WP-P2S）：参数匹配已由 P1-S 的 M1 对 B4 覆盖（读取深度不增加参数）；P2-S 补充单次读取获得更多 rank（B6，r=64/128）或更多状态（B7，r_mem=64/128）时能否达到 M1（r=32，L=4）的多跳准确率，以及不共享参数的展开 reader（B8），并给出准确率—延迟表；使用新的确认集（eval seed 888001）。LLM 规模的 P2 取决于 P1-L |
| T-10 | 未开始；在 P1-L 结论后整理。文献核查（T-10 前置）已完成：`docs/reports/lit_check.md` |

---

## 0. 执行摘要

### 0.1 核心问题

在相同的预训练模型、长期可训练参数、上下文记忆容量和写入预算下，增加当前查询的低维迭代计算，能否改善上下文中的关系组合能力？这种收益能否在同一 checkpoint 上通过推理时增加迭代步数获得，并优于将相同实际计算预算用于增大单步适配器？

本研究不把“LoRA 参数共享并循环使用”本身作为创新。RecLoRA 已覆盖共享 LoRA、token 级动态深度、路由和 step embedding [R1]；固定快权重上的多次读取已有早期先例 [R5]；上下文生成适配器和在线低秩记忆也已分别被 Generative Adapter 与 δ-mem 研究 [R6–R7]。

拟验证的贡献，是在明确的 PEFT 与状态预算约束下，区分并量化“表示容量、记忆容量、查询深度”三者的作用，提出一个可执行、可分析的低维迭代查询模块，并确认它的能力及成本边界。

### 0.2 主模型

采用列向量约定：

\[
u_t=Ax_t,\qquad z_t^{(0)}=u_t,
\]

\[
q_t^{(s)}=Qz_t^{(s)},\qquad m_t^{(s)}=M_{t-1}q_t^{(s)},
\]

\[
z_t^{(s+1)}
=u_t+\rho\phi(Rz_t^{(s)}+Um_t^{(s)}),
\]

\[
y_t=Wx_t+\alpha Bz_t^{(L_t)}.
\]

当前输入的整个内循环固定同一份历史记忆 \(M_{t-1}\)。迭代结束后，若该 token 被允许写入，则按一个既定 delta rule 更新记忆一次。额外循环增加读取计算，不增加主干调用次数或该 token 的写入次数。

### 0.3 第一批必须得到的证据

1. 单模块、同一 checkpoint、完全相同的输入与记忆，增加读取深度确有收益。
2. 收益不能被无记忆循环或固定 query 的循环完全解释。
3. 收益不能仅由增大单步 rank 或状态容量替代。
4. 至少存在一个实际延迟或固定状态预算下的有利点。
5. 训练内预算调节与训练外深度外推分别报告。

### 0.4 当前不作出的承诺

- 不保证循环越深准确率越高。
- 不保证低维循环优于所有同预算单步模型。
- 不宣称循环自动提高输出更新的 rank。
- 不宣称收敛意味着答案正确。
- 不宣称新增固定大小 memory 自动替代完整模型的 KV cache。
- 不宣称所有模块组合或局部理论推导具有原创性。
- 不把研究计划中的预设门槛、预算和示例数值写成实验结果。

---

## 1. 研究背景与定位

### 1.1 从静态 LoRA 到动态上下文适配

标准 LoRA 对一个线性投影使用：

\[
y=Wx+\alpha BAx.
\]

给定训练后的 A、B，该层的更新是固定线性变换；可将 BA 合并进 W [R2]。

如果适配分支引入非线性或输入相关状态，则更一般地写成：

\[
y=Wx+\alpha B f_\theta(Ax,M).
\]

这里的“低秩”首先表示输入和输出瓶颈，以及残差输出被限制在 B 的列空间。对非线性模型，不应不加限定地把整个分支称为一个固定低秩权重矩阵。

### 1.2 三种循环必须区分

| 循环 | 状态含义 | 是否跨 token | 是否重复主干 |
| --- | --- | --- | --- |
| 整个 Transformer 的深度循环 | 高维 hidden state | 取决于架构 | 通常会重复循环块 |
| 当前 token 的适配器循环 | 低维工作状态 \(z_t^{(s)}\) | 否 | 本计划规定不重复 |
| 关联记忆的时间递推 | 上下文状态 \(M_t\) | 是 | 每个 token 仍只走一次原主干 |

本文研究后两者的组合。权重共享不等于状态共享；若将整个因果线性注意力层按深度重复，每个虚拟深度一般有自己的历史状态。该模型不应与“单份记忆、当前 token 多次读取”混同。

### 1.3 原始候选公式的位置

用户最初提出：

\[
y=Wx+\alpha BA\operatorname{ReLU}(BAx).
\]

该式要求重复使用的 BA 具有兼容的方形维度。它是一个共享两次投影的非线性侧分支，可作为本研究基线，但不作为最终创新主张。

它有两个重要性质：

- 无激活时，重复固定低秩变换仍可合并为静态更新：
  \[
  (BA)^L=B(AB)^{L-1}A,\qquad L\ge1.
  \]
- 若采用 A 非零、B=0 的常见初始化，双重 BA 分支在零点的一阶梯度为零。仅替换 ReLU 为其他满足 \(\phi(0)=0\) 的激活，不能自动解决这一结构性问题。

因此，原始公式的实验版本必须明确初始化策略。例如加入直接 LoRA 项：

\[
y=Wx+\alpha[BAx+\eta BA\operatorname{ReLU}(BAx)],
\]

或使内部因子非零、仅用一个可学习的外部零门初始化。不得同时把外部缩放和输出路径都设成导致梯度无法启动的零值。

---

## 2. 相关研究与创新边界

### 2.1 RecLoRA 原文核查

依据用户提供稿件第 4–6 页，RecLoRA 明确包含 [R1]：

- 共享模块参与递归处理；
- 线性 router 与二值 gate，训练时使用 STE；
- token 级动态深度；
- 活跃集合逐步缩小，退出后不重新进入；
- 每步独立 learned step embedding；
- 冻结预训练主干；
- 域专属模块与跨域权重初始化。

原文第 5 页称循环处理使用“LoRA-augmented Transformer block”，公式 (3) 给出块内投影 \(W_0x+BAx\)。Figure 1 与 Algorithm 1 又将更新封装为共享 LoRA block / UPDATE，未完整定义 attention 范围与所有冻结算子的重复方式。因此：

- 不可断言 RecLoRA 只循环 BA；
- 不可断言 RecLoRA 缺少非线性或 token mixing；
- 不可仅依据示意图估算其准确 FLOPs；
- 在代码确认前，自行实现的基线应称为“RecLoRA 风格实现”。

原文第 6 页的 key-query 是域专家选择，不能等同于本文的跨 token 关联记忆。

2026-10-26 核查补充（`docs/reports/lit_check.md`）：OpenReview 页面受机器人验证阻挡未能打开；检索摘要将该条目标为 “ICLR 2026 Conference Withdrawn Submission”（未打开原页，属未核实信息）。引用时应写明为撤回/未录用的投稿稿件，页码级主张仍以用户上传稿为准。另：[R5] 附录称增加 inner loop 次数“性能相近”，即在其设置中未见更多内循环读取的收益，这是 H1 的直接先验反例，需在讨论中对照。

该版本没有展示同一 checkpoint 的训练外深度外推，也没有报告充分的实际延迟或平均 token 循环次数。第 9 页 Table 2 与 Figure 2 的 \(K_{\max}=1\) 分别为 61.14 和 66.34，未说明差异；Figure 2 的五个深度结果为 66.34、64.20、68.50、66.05、62.45。这些信息仅用于限定证据范围，不用于推断作者动机或否定全部结果。

### 2.2 近邻工作矩阵

| 工作 | 已覆盖内容 | 本计划仍需证明的区别 |
| --- | --- | --- |
| LoRA [R2] | 冻结主干与低秩线性适配 | 非线性迭代是否值得额外计算 |
| Linear Attention / Fast Weight Programmers [R3–R4] | 因果矩阵状态、外积写入、delta 型更新 | 不以记忆递推本身认领创新 |
| Fast Weights 2016 [R5] | 固定快权重下多次 inner-loop 读取 | PEFT 约束下的设计、预算曲线与组合能力证据 |
| Generative Adapter [R6] | 从上下文统计生成低秩适配器 | 额外查询计算的作用，而非上下文决定权重 |
| δ-mem [R7] | 冻结 LLM 上在线记忆与低秩 attention 修正 | 固定 writer/state 后，演化查询是否提供独立收益 |
| RecLoRA [R1] | 共享递归、token 路由、step embedding | 单遍主干、明确低维核心、记忆查询深度与预算泛化 |
| DEQ / ELM [R8–R9] | 固定点求解；ELM 还结合 LoRA 训练 | 不把固定点、残差停止或 DEQ+LoRA 单独当创新 |
| LT2 [R10] | 循环 Transformer 与线性/稀疏 mixer | 不循环完整块，控制单份记忆的读写预算 |
| ELR [R11] | 可调循环预算、循环条件化低秩调制 | 不以混合深度训练或 loop-conditioned LoRA 单独立项 |
| OperLoop [R12] | 优化视角、delta correction、步长与衰减 | 不以“循环是一种优化”单独立项 |
| LLM Sleep [R13] | 反复处理上下文以形成 fast weights | 增加查询阶段计算，与增加记忆形成计算分开比较 |
| End-to-End Memory Networks [R19] | 每个输出符号多次 hop 读取记忆，查询逐 hop 更新（bAbI 组合推理） | 本计划为固定大小 delta 记忆、冻结主干与 PEFT 约束；“多次读取+查询更新”本身不认领 |
| MemReasoner [R20] | 矩阵记忆上的迭代读取与查询更新，直至读出收敛 | 其记忆按 episode 伪逆写入、端到端微调；本计划为逐 token delta 写入、冻结主干、单份写前快照 |
| Retrofitting Recurrent Depth [R21] | 冻结基座 + 跨循环共享 rank-16 LoRA，深度外推约 1.5× | 其循环整个隐藏状态并重跑主干层，无记忆；本计划不重复主干 |

### 2.3 预期贡献的证据等级

**可以现在陈述的设计事实：**主干不重复、循环在 r 维、每次内循环记忆固定、写入次数可独立控制。

**有待证明的机制假设：**演化查询比固定查询更适合关系组合；更深读取与任务组合深度有系统关系。

**有待测量的系统假设：**该设计能在真实延迟或严格状态限制下取得优势。

**必须待实验后再写的结论：**提高真实 LLM 推理能力、具有训练外深度泛化、优于已有方法、形成新的最优性能—成本边界。

---

## 3. 研究问题与可证伪假设

| 编号 | 问题或假设 | 主要证据 | 主要反证 |
| --- | --- | --- | --- |
| H1 | 相同输入、记忆和 checkpoint 下，更深 reader 有独立收益 | \(L=1,2,4,8\) 的配对评估 | 固定 memory 后收益消失 |
| H2 | 查询随迭代演化是收益来源 | 演化 query 优于固定 query | 固定 query 已同样有效 |
| H3 | 收益涉及上下文关系组合 | 按 hop 分层、关键关系反事实 | 无记忆循环解释全部收益 |
| H4 | 深度与 rank/state 容量具有不同作用 | 深度—rank—负载的受控扫描 | 增大单步 rank 在所有预算点均更优 |
| H5 | 同一模型可在推理时调节预算 | 同 checkpoint 范围内深度曲线 | 只有分别重训的深层模型有效 |
| H6 | 训练外深度也能产生有效计算 | 训练 \(\{1,2,4\}\)，测试 8 | 测试 8 崩溃或仅增加错误 |
| H7 | 自适应停止优于固定深度分配 | 匹配实际平均及尾部延迟 | 节省步数却无实际速度或质量优势 |

H1–H5 为主线。H6 是增强目标，不作为项目早期必须同时实现的条件。H7 只在固定深度循环已经有效后研究。

---

## 4. 数学定义与候选模型

### 4.1 符号与维度

全篇采用列向量。省略 batch 维时：

| 符号 | 维度 | 含义 |
| --- | --- | --- |
| \(x_t\) | \(d_{\rm in}\) | 插入位置的当前 token 输入 |
| \(W\) | \(d_{\rm out}\times d_{\rm in}\) | 冻结基础投影 |
| \(A\) | \(r\times d_{\rm in}\) | 输入瓶颈 |
| \(B\) | \(d_{\rm out}\times r\) | 输出瓶颈 |
| \(R,U,Q,K,V\) | 各 \(r\times r\) | 循环、记忆读写投影 |
| \(u_t,z_t^{(s)},q_t^{(s)},m_t^{(s)}\) | 各 \(r\) | 低维输入、工作状态、query、readout |
| \(M_t\) | \(r\times r\) | 上下文关联记忆，key 映射到 value |
| \(L_t\) | 非负整数 | 当前输入的 reader 更新次数 |
| \(\alpha,\rho\) | 标量 | 输出缩放与内循环强度 |
| \(\lambda_t,\beta_t\) | 标量 | 初版的遗忘与写入系数 |

本计划的 L 表示更新次数：\(L=0\) 时 \(z=u\)，输出分支退化为 LoRA；\(L=1\) 表示一次迭代读取。论文与代码统一此约定，避免与“总层数”混用。

### 4.2 模型 V0：标准 LoRA

\[
y_t=Wx_t+\alpha BAx_t.
\]

部署效率比较时必须包含可合并版本；训练机制比较可使用未合并实现。不要以未合并 LoRA 的额外 kernel 开销夸大动态方法优势。

### 4.3 模型 V1：无记忆的低维循环

\[
u_t=Ax_t,\quad z_t^{(0)}=u_t,
\]

\[
z_t^{(s+1)}
=u_t+\rho\phi(Rz_t^{(s)}),
\]

\[
y_t=Wx_t+\alpha Bz_t^{(L_t)}.
\]

作用：验证低维共享非线性深度；它是 V2 的必要对照，不是上下文记忆模型。

初版不使用 learned step embedding。持续注入 u，避免迭代仅依赖初始状态并逐渐丢失输入。该模型是新的参数化，不能声称与 \(BA\operatorname{ReLU}(BAx)\) 等价。

### 4.4 模型 V2：固定记忆上的迭代查询，主候选

\[
q_t^{(s)}=Qz_t^{(s)},\quad
m_t^{(s)}=M_{t-1}q_t^{(s)},
\]

\[
z_t^{(s+1)}
=u_t+\rho\phi(Rz_t^{(s)}+Um_t^{(s)}).
\]

输出仍为：

\[
y_t=Wx_t+\alpha Bz_t^{(L_t)}.
\]

整个内循环使用同一份 \(M_{t-1}\)。不在每个 s 重新写入，不把未来 token 的状态用于当前读取。

实现时顺序计算 Qz、Mq、Um、Rz；不显式构造 \(UMQ\)，以避免不必要的 \(O(r^3)\) 乘法。

### 4.5 写入规则及明确的因果顺序

初版采用 decay-first delta 更新，作为既有更新家族中的基线 [R4,R14]：

\[
k_t=\frac{Ku_t}{\max(\|Ku_t\|_2,\epsilon)},\quad v_t=Vu_t,
\]

\[
\bar M_t=\lambda_tM_{t-1},
\]

\[
M_t=\bar M_t+\beta_t(v_t-\bar M_tk_t)k_t^\top.
\]

初版选用每模块的可学习标量 \(\lambda=\sigma(a_\lambda)\)、\(\beta=\sigma(a_\beta)\)，随后才比较依赖 u 的门控。该规则的误差项使用衰减后的 \(\bar M_t\)，不能把它与其他论文“先在未衰减状态上求误差”的规则混写。

每个 token 的顺序为：

1. 取得当前 x 和旧状态 M；
2. 进行 L 次读取，得到输出；
3. 若 write mask 为 1，则写入一次；
4. 若 write mask 为 0，保持 M 不变，包括不额外衰减；
5. 遇到新 episode，先重置状态再处理 token；
6. padding token 不读取有效状态、不触发写入。

两种任务协议分别定义：

- 在线协议：有效上下文 token 按因果顺序读后写。
- 上下文记忆问答协议：只在支持上下文写入；问题与答案生成阶段保持记忆不变，每个问题从同一上下文快照独立分叉。

不得把 teacher-forced 的正确答案写入记忆后，再用该记忆评估同一个答案。

### 4.6 模型 V2-FQ：固定 query 的必要消融

先算：

\[
m_t=M_{t-1}Qu_t,
\]

再循环：

\[
z_t^{(s+1)}=u_t+\rho\phi(Rz_t^{(s)}+Um_t).
\]

保留相同非线性核心，只取消 query 演化。m 和 Um 可以缓存；不为匹配 FLOPs 故意重复无效计算。其成本与 V2 不同，应据实报告。

### 4.7 模型 V3：可选的稳定性与停止机制

在 V2 的固定深度收益成立后，才研究：

- reader 对记忆的受控归一化；
- 循环核心的范数约束；
- 根据残差或输出变化停止；
- learned router 与理论残差指标的比较。

V3 不能在最初所有实验中默认启用，否则难以识别收益来自哪个组件。

---

## 5. 初始化、训练目标与梯度

### 5.1 默认初始化建议

以下为候选起点，需通过训练集/验证集 pilot 确认：

- A：常规非零随机初始化。
- B：全零，保持初始适配输出为零。
- R、U、Q、K、V：小尺度非零初始化；不将 ReLU 前所有循环路径都设为零。
- \(\alpha=1\)：固定非零，作为初版缩放约定。
- \(\rho=0.25\)：初版固定；验证候选为 0.1、0.25、0.5。
- 初版不使用 \(\alpha/r\) 的隐含缩放；若使用，必须在配置中明确并对所有基线统一。
- \(\lambda\) 初值 0.99、\(\beta\) 初值 0.1，仅为 pilot 起点。

B=0 时，首个反向传播可能只有 B 获得有效梯度，内部模块要在 B 更新后才能开始学习。这是可接受的延迟启动。必须验证第二次或后续更新时内部梯度能到达。若同时设 \(\alpha=0\) 且 B=0，分支可能无法启动。

### 5.2 两阶段训练

**机制阶段：固定编码器与 writer。**

先用单跳关联任务学习 writer 及必要输入投影，随后冻结 A、K、V、写入门控。缓存相同的 x/u/M，训练 reader 的 R、U、Q、B，并比较读取深度。

这是一项保守诊断实验：固定的 writer 不专门为多步 reader 优化。若结果为负，要区分“reader 无效”与“writer 的 value/key 空间不适合关系组合”。训练时冻结 writer 不是证明读取深度效应的必要条件；可以追加一个预算受限的联合训练对照，再在评估时锁定相同输入与记忆。关键控制是评估时固定 x/u/M，而不是要求 writer 从未与 reader 联合训练。

**端到端阶段：联合训练。**

允许 A、writer、reader 联合训练，评估模型上限。机制阶段与端到端阶段结果分别呈现，不能把联合训练收益全部归因于读次数。

如需解耦输入投影 \(A_{\rm write}\) 和 \(A_{\rm read}\)，将其作为新增参数的变体，单独记账。

### 5.3 深度采样与目标

第一版使用固定深度 L=1 与 L=4 验证是否可训练，然后训练混合深度的单个 checkpoint：

\[
L\sim p(L),\qquad L\in\{1,2,4\},
\]

\[
\mathcal L_{\rm main}
=
\mathbb E_{(C,q,a),L}
\left[-\log p_\theta(a\mid q,M(C);L)\right].
\]

初版 p(L) 均匀采样；若调整，应在验证集选择并记录。不同 L 的同一模型推理扫描才用于证明预算调节。

一开始不叠加多项辅助损失。需要时再单独评估：

- 单跳写入/读取辅助监督；
- 低深度保真项；
- 稳定性正则；
- 自适应停止的计算惩罚。

这些均为已有训练手段或候选设计，不因叠加而自动构成创新。

### 5.4 冻结参数不等于关闭整个计算图

设置主干参数不更新，但若梯度需要穿过冻结层到达早期 adapter，不能将这些层整体放入 no_grad 或 detach。

固定编码器的单模块机制实验可以缓存其输出并停止梯度；多层联合适配时必须保留必要的激活反向路径。

记忆写入默认参与训练图。若使用截断 BPTT 或在 chunk 边界 detach M，必须标明截断长度，且比较方法使用一致协议。逐 token 无条件 detach 会显著改变 writer 的学习问题，不能作为隐蔽的省显存操作。

---

## 6. 理论分析任务及可陈述结论

### 6.1 固定输出子空间与输入信息瓶颈

对所有 L：

\[
\delta y=\alpha Bz^{(L)}\in\operatorname{col}(B).
\]

固定 M 时，若 \(Ax_1=Ax_2\)，则适配分支输出相同。因此：

- 迭代增加子空间内部的映射复杂度；
- 迭代不自动扩大输出子空间；
- 更深计算无法恢复 A 已经丢失的信息。

固定历史状态、在可微区域内：

\[
\frac{\partial \delta y}{\partial x}
=
\alpha B J_f A,
\quad
\operatorname{rank}\left(\frac{\partial \delta y}{\partial x}\right)\le r.
\]

上述 rank 界在历史状态固定时成立；对整个序列求导时，必须显式计入跨 token 的记忆路径，不能直接套用。

### 6.2 线性内循环与记忆组合阶数

令 \(\phi\) 为恒等映射，\(T=\rho(R+UMQ)\)：

\[
z^{(L)}=\sum_{j=0}^{L}T^j u.
\]

特别地，R=0、U=Q=I 时：

\[
z^{(L)}=\sum_{j=0}^{L}\rho^jM^j u.
\]

这说明迭代可增加关于动态记忆的组合阶数。但：

- 固定 M、L 时，可预计算该多项式；
- 线性版本对当前 x 仍是线性映射；
- 不能声称它必须逐查询执行全部循环；
- 真实 learned key/value 未必自然满足关系组合所需的空间对齐。

实验应将线性 reader 作为机制对照，并报告预计算策略在多查询场景下的成本。

### 6.3 收敛与停止误差

固定上下文，定义：

\[
F(z)=u+\rho\phi(Hz),\qquad H=R+UMQ.
\]

若 \(\phi\) 的 Lipschitz 常数为 \(L_\phi\)，且：

\[
c=|\rho|L_\phi\|H\|_2<1,
\]

则有唯一不动点 \(z^\star\)。令 \(e_s=\|F(z_s)-z_s\|_2\)，则：

\[
\|z_s-z^\star\|_2\le \frac{e_s}{1-c},
\]

\[
\|\alpha Bz_s-\alpha Bz^\star\|_2
\le
\frac{|\alpha|\|B\|_2}{1-c}e_s.
\]

证明来自收缩不等式与三角不等式，是标准固定点分析的应用 [R8]，不作为全新定理认领。

使用该界时必须保证 c 是有效上界；一次不充分的谱范数估计不能支撑严格保证。若避免构造 H，可以采用：

\[
c\le |\rho|L_\phi
\left(\|R\|_2+\|U\|_2\|M\|_2\|Q\|_2\right).
\]

可以在读侧使用 Frobenius 范数约束的记忆缩放以给出保守上界，并将它作为单独消融。ReLU 的 \(L_\phi=1\)；更换激活或加入归一化后要重新检查条件。

停止界只保证接近适配器自己的固定点，不保证下游答案正确或任务损失单调下降。固定点输出也可能比某个有限中间步更差。

### 6.4 低维循环的精确消元条件

若某一模块满足：

\[
h_{s+1}=h_s+g_sB\phi(Ah_s),\quad h_0=x,
\]

则 \(h_s=x+Bz_s\)，且：

\[
z_{s+1}=z_s+g_s\phi(Ax+(AB)z_s).
\]

线性 router 也可预投影。此类 \(O(r^2)\) 递推是等价实现，不自动构成新函数类。含完整 attention、FFN 或不兼容高维非线性的循环不能直接套用。

对原始 \(BA\operatorname{ReLU}(BAx)\)，一般：

\[
A\operatorname{ReLU}(Bz)\ne\operatorname{ReLU}(ABz).
\]

应以明确的结构条件区分模型创新与计算优化。

### 6.5 记忆误差与深度的相互作用

拟研究问题：当 \(M=\widetilde M+E\) 含写入干扰时，迭代是帮助组合有效关系，还是放大 E？

可先使用矩阵幂的标准扰动界解释线性 reader：

\[
\|M^L-\widetilde M^L\|_2
\le
\|E\|_2
\sum_{j=0}^{L-1}
\|M\|_2^{L-1-j}\|\widetilde M\|_2^j.
\]

它不是新的通用扰动理论。研究价值在于将干扰负载、范数控制、读取深度与实际任务误差对应起来，解释收益何时饱和或反转。

初期理论交付物：三项命题的完整证明、假设清单、反例，以及与实验变量一致的预测。暂不追求对整个 LLM 的全局正确性保证。


---

## 7. 数据、任务与模型选择

### 7.1 任务 T1：单跳关联检索

目的：先验证 writer 能保存可用信息，建立容量与干扰基线。

- 上下文给出随机 key-value 对。
- 查询要求取回一个 value。
- 改变事实数量、key 相似度、无关条目数量及 r。
- 评估读出正确率、key 的相关性、状态范数与遗忘。
- 单跳检索尚未可靠时，不直接解释多跳失败为循环无效。

对显式符号实验，可先用可控的正交或随机编码研究理想记忆，再用学习到的 writer 替换。理想编码的成功不等于语言模型成功。

### 7.2 任务 T2：关系组合，核心机制任务

构造答案唯一的函数图或有类型的关系：

\[
a\xrightarrow{r_1}b,\quad b\xrightarrow{r_2}c,\quad c\xrightarrow{r_3}d.
\]

查询给出起点与关系序列，或明确要求沿某关系走 h 步，答案为唯一终点。

变量独立控制：

- hop 数 \(h\)；
- reader 深度 \(L\)；
- 记忆维度 r；
- 总边数与无关边数；
- token 长度；
- key 干扰与写入误差。

不同 h 尽量保持总边数和文本长度相近，避免将更长上下文误解释为更深组合。

初始训练 hop 为 1–3，测试 hop 为 1–8；训练深度为 \(\{1,2,4\}\)，测试深度为 \(\{1,2,4,8\}\)。两种外推分开报告。

### 7.3 任务 T3：更新、干扰与信息缺失

在 T2 之后加入：

- 同一事实的后续更新，要求使用指定时刻的有效关系；
- 与目标无关但相似的 key；
- 删除必要事实后的“信息不足”样本；
- 增加干扰而保持推理路径不变。

事实更新任务保留时间语义，不能随意打乱更新顺序。静态图任务则可以随机排序关系表。

删除必要边时，应修改标签为“信息不足”，或只作为敏感性检查。不能删除信息后仍要求唯一原答案并把合理拒答判错。

### 7.4 防止捷径

- 实体随机重命名，训练/验证/测试使用独立生成种子。
- 查询不直接包含答案或完整推理路径。
- 对关键关系做反事实替换，检查答案是否随事实改变。
- 控制训练与测试图结构重复，按图/episode 划分，而不是仅按问题划分。
- 固定答案形式和最大生成 token 数，避免显式推理文本长度成为额外计算来源。
- 真值 hop 只用于分层与 oracle 对照；除非问题本身明确提供 hop，不将测试标签用于选择 L。
- 同一上下文的多个问题从相同状态快照分叉，不能相互写入或累积正确答案。

### 7.5 从符号任务过渡到语言任务

按难度递增：

1. 显式符号 key/value 与小型 reader。
2. 模板化自然语言关系，使用冻结 LLM 特征。
3. 未见过的语言模板、实体和关系表达。
4. 真实语料多跳问答。

真实任务首选 MuSiQue，它以单跳问题组合构造多跳问答 [R15]；补充任务可选带 supporting facts 的 HotpotQA [R16]。先限制为给定支持段落的设置，隔离检索器误差。

采用两种明确协议：

- **上下文保留：**原文仍在基础模型可访问的 attention/KV 中，测量 memory 分支的额外帮助。
- **记忆问答：**先编码支持文档，随后清空该文档对应的 backbone KV，仅保留声明的 memory；用问题和新建的生成 KV 回答。

后者是改造后的记忆评测协议，不应把其分数冒充原数据集完整官方协议的直接可比结果。所有对照具有相同信息访问条件；数据预处理、答案规范化与 official script 的差异需记录。

真实数据上的 pretraining knowledge 可能绕过上下文，应结合支持事实反事实或合成任务，不能只靠真实问答准确率证明记忆机制。

### 7.6 预训练模型与插入位置

为降低试错成本，建议默认：

- 首轮 LLM：Qwen/Qwen3-0.6B [R17]；
- 确认阶段：Qwen/Qwen3-4B [R18]，仅在小模型通过阶段门槛后使用；
- 机制插入位置：单个中间层的 attention 输出投影；
- 所有局部基线使用相同插入位置；
- 多层阶段：先固定少量均匀间隔插入位置，再研究位置敏感性。

这些是可替换的计划选择，不是对这些模型的最优性结论。运行前锁定模型和 tokenizer 的精确 revision、chat template、attention 实现、dtype 与软件依赖。更换模型家族时，不沿用未验证的层索引和投影形状假设。

---

## 8. 基线与实验矩阵

### 8.1 最小矩阵

| ID | 方法 | 核心用途 | 初期优先级 |
| --- | --- | --- | --- |
| B0 | 冻结基础模型 | 未适配参照 | 必需 |
| B1 | 同位置 LoRA | 静态 PEFT 基线 | 必需 |
| B2 | 单步非线性瓶颈 adapter | 排除只有非线性的收益 | 必需 |
| B3 | V1，无记忆低维循环 | 排除只有额外深度的收益 | 必需 |
| B4 | V2 的 L=1，共享同一 writer | 单次读取直接对照 | 必需 |
| B5 | V2-FQ，固定 query 循环 | 隔离 query 演化 | 必需 |
| M1 | V2，演化 query，L>1 | 主候选 | 必需 |
| B6 | 更大 rank 的单步模型 | 参数/实际延迟对照 | 通过首轮后 |
| B7 | 更大 state 的单次读取 | 记忆容量对照 | 通过首轮后 |
| B8 | 不共享参数的展开 reader | 共享是否必要 | 确认阶段 |
| B9 | 原始双 BA 非线性公式及带 skip 版本 | 对照最初提案 | 补充 |
| B10 | RecLoRA 风格循环及 router | 递归适配近邻 | 确认阶段 |
| B11 | 原版 δ-mem | 外部方法对照 | 核心效应确认后 |
| B12 | Generative Adapter | 上下文参数化近邻 | 数据/训练预算可匹配时 |

B4 的插入位置、写入次序、门控与读出不必与原版 δ-mem 相同。因此应称为“匹配的单次读取基线”，不能直接标为原版 δ-mem。

外部方法应先核对作者实现和训练配方。无法严格复现时注明修改项，并将结果与机制隔离基线分组报告。

### 8.2 三组预算对照分别进行

**参数相同：**相同 trainable 参数数量或明确匹配误差，比较不同深度。

**状态相同：**相同动态状态总字节数，包含模块数、heads/slots 和 dtype；不能只比较单个 r。

**时间相同：**在目标硬件上匹配实际端到端 latency，再比较质量。

通常不能同时精确匹配参数、状态、训练 FLOPs、推理 FLOPs 和延迟。不要以一句“公平比较”掩盖未匹配维度。

### 8.3 训练预算

至少区分：

1. 相同训练数据量与更新次数；
2. 匹配训练 FLOPs 或 GPU 时间；
3. 同一 checkpoint 的推理深度扫描。

混合深度训练的平均 L 应报告。不得用更多训练计算得到的循环模型，仅与较少训练计算的单步模型比较后归因于架构。

初期只取 r=16、32；若存在机制收益，再扩展到 8、64。不要在未确认信号前展开 rank × depth × writer × 插入层 × 模型规模的全网格。

### 8.4 单模块机制实验的冻结规则

为避免早期循环改变后续层输入，第一轮只插一个模块，固定上游特征和 writer：

- 在同一 checkpoint 的深度扫描中，缓存完全相同的 x、u、M；
- 在不同 reader 的比较中，使用相同的固定 A 与 writer，并记录 reader 参数差异；
- 每个问题、每个 L 从同一快照复制状态；
- 不能只固定 writer 参数或 M 的形状，就宣称状态相同。

端到端多层实验可以允许跨层反馈，但必须独立命名和报告。

缓存相同 x/u/M 的机制实验使用固定给定输入、单步分类，或明确标注的 teacher-forced 指标。自由生成的 exact match 必须由模型自己的预测前缀逐 token 产生，不能复用由正确答案前缀生成的缓存隐藏状态。不同 L 的自由生成可以从同一支持上下文记忆快照开始，但其后生成的 token 与对应 x/u 可能自然不同；这类结果不再声称全程输入相同。

---

## 9. 评测、统计与结果记录

### 9.1 主指标

- 主任务：关系组合 exact match，按 hop 分层，另报告预先定义的多跳平均。
- 记忆基础：单跳正确率与容量/干扰退化。
- 机制敏感性：关键事实改变后，预测是否按预期改变。
- 泛化：训练内计算预算、训练外计算深度、未见任务 hop 分开。
- 实际性能：prefill、decode、总请求时间、吞吐、峰值显存。
- 停止策略：平均、P50、P95 循环次数和实际延迟。

语言问答采用答案 EM/F1 与规范化规则；模型生成的长解释不能代替答案正确率。普通能力保留可使用预先锁定的小型验证集或持出语言建模损失，作为副指标。

### 9.2 样本与随机性

建议最终确认使用至少 3 个训练 seed，每个主要难度层级约 1,000 个独立测试 episode。该数量为起点，应根据验证集 pilot 的方差和预设最小效应调整。

同一图生成的多个问题不是独立样本。置信区间优先按图/episode 进行配对或聚类 bootstrap，并单独报告 seed 间变化。

主指标、比较方法、预算范围和停止阈值在正式测试前锁定。测试集不用于选择 rank、rho、最大 L、插入位置或最有利的任务子集。

### 9.3 必需图表

1. 同一 checkpoint 的质量—L 曲线，按 hop 分层。
2. 质量—实际延迟曲线，标出更大 rank 的单步对照。
3. hop × L 的准确率热图。
4. 记忆负载 × L 的退化图。
5. 固定 query、无记忆、完整模型的配对差值。
6. 单模块机制结果与端到端结果的对应关系。
7. 若研究停止：质量—平均延迟及尾部延迟曲线。

### 9.4 每个结果必须保留的最小字段

实验 ID、代码 commit、配置哈希、数据生成种子/版本、模型 revision、训练 seed、checkpoint、插入位置、参数量、state 字节数、rank、L、训练 tokens、训练时间、评测样本数、答案/得分、延迟、峰值显存、失败原因。

主结论应能由原始样本级结果重算，不能只保存汇总表或最佳截图。

---

## 10. 计算复杂度与工程实现

### 10.1 参数与算术成本

对 V2，忽略偏置与少量标量：

\[
P=r(d_{\rm in}+d_{\rm out})+5r^2.
\]

若加入两个依赖 u 的标量写入门控，另计约 \(2r+2\) 参数。

不计原始 Wx，主要运算为：

- A、B 投影：\(r(d_{\rm in}+d_{\rm out})\) MAC；
- 每次 reader：Qz、Mq、Um、Rz，约 \(4r^2\) MAC；
- writer：Ku、Vu、\(\bar M k\) 与 outer update，约 \(4r^2\) 的主要乘加/矩阵元素更新，另有衰减、归一化等操作。

因此可用如下主导项估算，而不是作为精确硬件 FLOPs：

\[
C_t\approx r(d_{\rm in}+d_{\rm out})+(4L_t+4)r^2
+O(L_t r+r^2).
\]

write mask=0 时不支付 writer 成本。MAC 与 FLOP 的换算约定须在报告中明确。

### 10.2 状态与训练激活

每个模块、每个独立序列的推理 memory 为 r² 个元素：

\[
S_{\rm memory}=N_{\rm modules}\times B_{\rm batch}\times r^2
\times \text{bytes-per-element}.
\]

另有 O(r) 的工作状态。共享权重不消除 batch 内各序列的独立 memory。

朴素训练或并行读取实现可能保存所有 prefix state 与迭代激活，新增开销达到：

\[
O(Tr^2+TLr),
\]

还未计 backbone 激活。不得将推理 O(r²) 状态写成训练内存也与序列长度无关。

### 10.3 参考实现与优化顺序

1. 逐 token、功能式更新的 FP32 memory 参考实现。
2. 完整序列与 incremental decode 对齐。
3. 固定 L 的向量化/分块 prefill。
4. BF16 memory 的误差与速度评估。
5. 融合 r 维矩阵运算，减少 kernel launch。
6. 最后加入动态 active set 或停止逻辑。

先验证确定性正确性，不在第一版同时采用多个难以审计的自定义 kernel。

### 10.4 Prefill 与 scan 的条件

当某层输入 \(u_{1:T}\) 已知、writer 不依赖当前 reader 输出时：

\[
M_t=M_{t-1}F_t+G_t,
\]

\[
F_t=\lambda_t(I-\beta_tk_tk_t^\top),
\quad
G_t=\beta_tv_tk_t^\top.
\]

这为 scan 或专门的 delta-rule 分块算法提供条件 [R14]。但：

- 通用 dense affine scan 可能引入 r³ 中间运算；
- 显式保存全部 prefix states 会增加 Tr² 存储；
- 通用线性注意力 kernel 未必暴露完整状态供多次 query 使用；
- 默认 kernel 可能返回写后读取，而本模型要求写前状态；
- 若 v 改为依赖最终 z，标准线性 scan 的适用性需重新检查。

算法逻辑中的“写一次”不等于 checkpointing 绝不重算 writer。任何重算都计入实际成本。

### 10.5 正确性测试，必须通过后再训练主实验

| 测试 | 具体风险 |
| --- | --- |
| B=0 输出与梯度启动 | 零输出不应导致整个分支永久无梯度 |
| 梯度穿过冻结层 | 冻结 W 不应意外切断早期 adapter |
| 修改未来 token | 过去输出和 memory 不得变化 |
| 逐 token / 分块 / 整段一致 | scan 与写前读取的时间对齐 |
| padding、packing 与 reset | 无效 token 不得遗忘或写入其他样本状态 |
| 固定输入改变 L | 主干调用数和逻辑写入次数不变 |
| memory snapshot 分叉 | 多问题、多深度之间无相互污染 |
| functional update | 原地写入不得破坏 autograd 保存值 |
| 低精度与优化 kernel | 与 FP32 参考的误差界限清楚 |
| 已知收缩实例 | 停止误差界只验证数值结论，不冒充任务验证 |

若部署 beam search，还要同步复制和重排 memory；第一版评测使用固定 greedy decoding，减少额外变量。

### 10.6 性能测量协议

报告硬件、精度、软件版本、batch size、输入/输出长度、warmup 次数、同步方式、计时范围与重复次数。分别计时：

- 上下文编码与写入；
- 查询/生成 reader；
- 完整请求；
- 同一 memory 被多个问题复用时的摊销成本。

动态路由可能引入 gather/scatter、padding 和图编译开销；平均 L 降低不等于延迟降低。静态 LoRA 的合并版本应进入部署成本对照。

---

## 11. 阶段计划、资源与继续／停止标准

以下时间与门槛为规划建议，不是已完成结果或承诺工期。先以实际吞吐 pilot 估算 GPU 小时，再锁定预算。当前计划不启动训练任务。

### 11.1 阶段安排

| 阶段 | 建议工作窗口 | 工作内容 | 交付物 |
| --- | --- | --- | --- |
| P0 | 2–3 个工作日 | 原文/公式归档、参考实现、测试、单个吞吐 pilot | 可重复配置、正确性报告、资源估计 |
| P1 | 3–5 个工作日 | T1/T2、单模块、固定 writer/M、B3/B4/B5/M1 | 深度机制曲线、首轮判定 |
| P2 | 约 1 周 | rank/state/训练计算对照，至少 3 seeds | 预算匹配结果、机制结论 |
| P3 | 1–2 周 | 小型 LLM、真实多跳 QA、少量多层模块 | 端到端结果、普通能力保留 |
| P4 | 约 1 周 | 停止机制、范数约束、分块或 kernel 优化 | 精度—延迟曲线、停止机制消融 |
| P5 | 约 1 周 | 第二规模确认、证据整理、论文草稿 | 可复现实验包、主张—证据表 |

总日历约 6–8 周作为安排参考，受排队、数据准备和问题定位影响。若 P1 无清晰信号，应提前停止后续大模型扩展。

### 11.2 资源控制

- 首轮筛选建议设一个小预算，例如不超过 24 GPU 小时；这只是预算上限草案，不是运行时间预测。
- 在吞吐 pilot 后估算每个配置的训练、评测与 checkpoint 成本。
- 采用逐步筛选，不同时跑完整参数网格。
- 总预算可按 P0/P1/P2/P3/P4/P5 分配为约 5%/15%/25%/30%/15%/10%，根据实际瓶颈调整。
- 若使用现有 GH200/PBS 环境，先验证 aarch64 依赖、checkpoint 路径与作业恢复；不假定 x86 的预编译扩展可直接复用。
- 最终扩展到 7B 或更多 seeds 前，应基于阶段结果更新预算表，而非因已有投入继续放大实验。

### 11.3 Gate 0：实现成立

全部关键正确性测试通过，且：

- 初始输出保持基础映射；
- 梯度能够启动；
- 内循环不重跑主干；
- 不读取未来状态；
- state/reset/mask 明确；
- prefill 与 decode 语义一致。

未通过时只修复实现，不解释任务性能。

### 11.4 Gate 1：存在读取深度效应

建议预注册的确认门槛：

- 同 checkpoint、相同 x/u/M，较深 reader 在预定义多跳主集合上比 L=1 提高至少约 3 个百分点；
- 配对差值置信区间为正；
- 第二个独立任务设置或干扰条件中复现；
- 单跳任务退化不超过约 1 个百分点，或给出预先允许的明确权衡。

这些门槛可在验证集 pilot 后、正式测试前调整并冻结。若存在天花板效应，应事先改变难度，不能事后挑选有利测试子集。

### 11.5 Gate 2：演化查询具有独立作用

完整模型需要优于无记忆循环和固定 query 循环。

- 若无记忆模型已解释全部收益：转为非线性低维适配研究。
- 若固定 query 同样有效：停止宣称反馈式多次关联读取是主要机制。
- 若收益仅出现在不同 L 同时改变编码或记忆的比较中，而在相同 x/u/M 的评估中消失：将主要收益归因于 writer-reader 协同或编码变化。若联合训练后，对同一快照仍存在深度收益，则保留固定记忆读取深度的主张，并注明适合组合读取的编码需要联合训练。

### 11.6 Gate 3：存在实际预算价值

至少存在一个可靠的精度—实际延迟有利点，或在严格状态限制下具有清晰能力优势。

如果更大 rank 的单步模型在所有匹配时间点都更好，应收窄为能力边界或理论分析，不再主打系统效率。

### 11.7 Gate 4：端到端推广

在真实语言任务或更大模型上保留主要效应，并且：

- 信息访问协议一致；
- 结果跨 seeds 稳定；
- 普通能力退化处于预设范围；
- 新增状态和完整 KV 分开记账。

只在合成任务成立时，可形成机制研究，但不能声称通用 LLM 推理提升。

---

## 12. 风险、负结果与转向

| 观测 | 可能解释 | 下一步 |
| --- | --- | --- |
| 固定 M 后收益消失 | writer 或编码变化是主因 | 改为协同训练问题，撤回固定记忆 claim |
| 固定 query 同样有效 | 非线性迭代而非查询反馈 | 简化模型，研究低维深度 |
| 大 rank 单步始终更优 | 容量是主要瓶颈 | 分析 rank-depth 边界，停止盲目加循环 |
| 深度 8 失败但 1/2/4 有效 | 无训练外深度泛化 | 保留范围内预算调节 claim |
| 干扰越多，深度越差 | 多次读放大记忆误差 | 分析噪声、规范化与停止条件 |
| 收缩后任务明显变差 | 稳定性约束压制有用动态 | 区分有限深度与固定点目标 |
| 理论 FLOPs 低但实际慢 | 串行依赖或小 kernel 开销 | 固定 L、融合计算、实测后决定 |
| 仅上下文仍在 KV 中有效 | 模型绕过或仅辅助显式上下文 | 不宣称压缩记忆替代能力 |
| writer 单跳都学不会 | 编码或读写训练问题 | 先用理想 memory 定位，再修 writer |
| 只有某一 seed 有效 | 效果弱或优化不稳定 | 不扩模型，确认方差和超参敏感性 |

不将每个负结果都用新增模块掩盖。每次增加组件应对应一个已观察到的具体失败原因，并有独立消融。

---

## 13. 可复现交付物与实现任务

### 13.1 建议仓库结构

~~~text
configs/
  base.yaml
  mechanism.yaml
  end_to_end.yaml
models/
  lora_baselines.py
  recurrent_reader.py
  delta_writer.py
  state_manager.py
data/
  generate_relations.py
  preprocess_qa.py
train/
  train_writer.py
  train_reader.py
  train_joint.py
eval/
  evaluate_depth.py
  evaluate_counterfactual.py
  benchmark_latency.py
tests/
  test_causality.py
  test_state_lifecycle.py
  test_gradient_flow.py
  test_incremental_equivalence.py
reports/
  experiment_registry.csv
  decisions/
  figures/
~~~

这些是待实现的规划路径，不是本次已创建的程序。

### 13.2 可直接分配的任务包

| 任务 | 输出 | 验收条件 |
| --- | --- | --- |
| T-01 | 数学定义与符号表 | 维度、L 定义、写前读、mask 无歧义 |
| T-02 | 参考 reader/writer | Gate 0 测试通过 |
| T-03 | 关系数据生成器 | 唯一答案、独立 splits、反事实标签正确 |
| T-04 | 固定 writer 训练与缓存 | 相同输入/state 可重用且哈希一致 |
| T-05 | 首轮机制实验 | 同 checkpoint 的 L 扫描与固定 query 对照 |
| T-06 | 参数/状态/时间匹配 | 每类预算的匹配误差明示 |
| T-07 | LLM 注入与生成状态 | teacher forcing / decode / reset 一致 |
| T-08 | 真实任务处理 | 支持文档、答案与信息访问协议可复现 |
| T-09 | 停止机制 | 数值界限定清楚，实际 latency 有测量 |
| T-10 | 证据归档与论文 | 每项主张有实验或证明，保留负结果 |

### 13.3 每阶段决策记录

每个阶段结束输出一个短决策文件，记录：

- 原始假设；
- 实际配置与资源消耗；
- 主要结果和不确定性；
- 支持或否定了哪些主张；
- 继续、缩小范围或停止的理由；
- 下一阶段唯一最重要的问题。

---

## 14. 论文叙事与最终主张

### 14.1 最强但尚待验证的主张

在单遍冻结主干和固定上下文状态预算下，低维查询迭代能够为相同训练后模型提供可调的上下文计算；其收益依赖关系组合深度与记忆质量，并在部分实际预算范围内优于单次读取或增加 rank。

该表述只有在相应实验全部成立后才能用作结果结论。

### 14.2 可接受的较窄结果

- 证明若干递归低秩模块可精确降维，并实现显著实际速度收益。
- 系统刻画固定 rank 下，循环深度与记忆干扰的能力边界。
- 发现循环只在特定信息访问或任务组合条件下有效。
- 提供可复现的循环适配器对照和负结果，明确为何某些“动态 LoRA”只是重参数化。

### 14.3 不采用的叙事

- “首次把循环引入 LoRA。”
- “首次将 linear attention 解释为动态权重。”
- “只需增加循环次数即可获得推理能力。”
- “状态收敛，所以答案更正确。”
- “r² 状态意味着能无损保存 r² 条任意事实。”
- “更深仍输出在 col(B)，所以深度完全没有作用。”
- “只增加少量参数，所以计算也几乎免费。”

### 14.4 建议的最终论文结构

1. 问题：固定 PEFT 与上下文状态预算下，如何增加有效查询计算。
2. 近邻与边界：RecLoRA、δ-mem、fast weights、looped models。
3. 方法：单遍主干、低维 reader、固定内循环记忆、独立 writer。
4. 分析：输出子空间、组合阶数、噪声与停止条件。
5. 机制实验：相同 x/u/M 与反事实检验。
6. 预算实验：参数、状态、训练成本与实际延迟分别匹配。
7. 端到端任务与失败模式。
8. 局限、复现与结论。

---

## 15. 优先执行清单

第一批实验只回答一个问题：

> 在单个模块、固定输入投影和 writer、同一记忆快照下，演化查询的多步读取是否比单步读取、固定 query 和无记忆循环更好？

建议顺序：

1. 完成 FP32 参考实现与因果/状态/梯度测试。
2. 完成 T1 单跳检索，验证 writer 可用。
3. 固定 writer，运行 T2 的 B3、B4、B5、M1。
4. 用同一 checkpoint 扫描 L；不要为每个测试 L 单独重训后混淆结论。
5. 出现可靠信号后补齐 LoRA、非线性单步和更大 rank 对照。
6. 先测实际延迟，再决定是否研究动态停止。
7. 通过 Gate 1–3 后，再扩展多层与真实语言任务。

研究价值由证据决定。若核心机制未成立，优先收窄问题和报告负结果，而不是给结构换名字。


---

## 附录 A：参考伪代码与状态语义

以下为单样本、单 token、FP32 的语义参考，不是本次已实现或验证的训练程序。矩阵均按第 4 节形状提供；参数 p 的创建、批处理和模型注入由后续实现完成。

~~~python
import torch

def recurrent_adapter_step(
    x, M_prev, p, loop_steps, *,
    valid=True, reset=False, write_enabled=True
):
    # Reference semantics: deterministic, fixed loop count, no in-place writes.
    M_old = torch.zeros_like(M_prev) if reset else M_prev
    base = p.W @ x  # Frozen parameter; keep input gradients when needed.
    if p.bias is not None:
        base = base + p.bias

    if not valid:
        # Padding never decays or writes the state.
        return base, M_old, {"reads": 0, "writes": 0}

    u = p.A @ x
    z = u

    for _ in range(loop_steps):
        q = p.Q @ z
        m = M_old @ q  # Always read the same pre-write memory.
        z = u + p.rho * torch.relu(p.R @ z + p.U @ m)

    y = base + p.alpha * (p.B @ z)

    if not write_enabled:
        return y, M_old, {"reads": loop_steps, "writes": 0}

    k_raw = p.K @ u
    k = k_raw / torch.linalg.vector_norm(k_raw).clamp_min(p.eps)
    v = p.V @ u

    M_bar = p.lam * M_old
    error = v - M_bar @ k
    M_next = M_bar + p.beta * torch.outer(error, k)

    return y, M_next, {"reads": loop_steps, "writes": 1}
~~~

实现备注：

1. p.W 冻结，但 base 的输入梯度是否保留取决于训练阶段，不能默认全局 no_grad。
2. p.alpha 非零，p.B 初始为零；参数 p.lam、p.beta 使用约定的合法参数化。
3. 混合精度版本需显式处理 q、M、m 的 dtype；cast 不应意外 detach 梯度。
4. L=0 时 reader 不访问 memory，但若 write_enabled=True，仍执行一次规定写入。
5. 在原有线性投影位置用该 wrapper 替代线性层；不额外添加一份未经定义的 x 残差。
6. 若选择整个 Transformer 残差位置插入，应单独定义为另一个模型，所有基线保持同位置。
7. 回答阶段关闭写入，多个问题分别从支持上下文的同一 memory snapshot 出发。
8. 优化实现不得改变 valid、reset、write_enabled 与写前读取的语义。

---

## 附录 B：首轮配置草案

以下 YAML 是拟定配置，不是已存在程序的可直接执行接口。实现后需保持字段与代码一致。所有超参数只允许在训练集/验证集 pilot 调整。

~~~yaml
project:
  name: recurrent_lora_memory
  plan_version: "1.0"
  stage: mechanism
  status: proposed

model:
  backbone: Qwen/Qwen/Qwen3-0.6B
  revision_policy: resolve_and_pin_before_first_run
  freeze_backbone_parameters: true
  preserve_required_input_gradients: true
  insertion:
    target: attention_output_projection
    placement: single_middle_layer
  rank: 32
  activation: relu
  output_scale_alpha: 1.0
  loop_strength_rho: 0.25
  init_output_B: zeros
  init_other_projections: nonzero_scaled
  step_embeddings: false

memory:
  orientation: key_to_value
  update: decay_first_delta
  read_order: before_write
  loop_reads_share_state: true
  writes_per_enabled_token: 1
  key_normalization: l2_clamp
  epsilon: 0.000001
  gate_type: learned_scalar_per_module
  lambda_initial: 0.99
  beta_initial: 0.1
  state_dtype_reference: float32
  reset_at_episode_boundary: true
  padding_keeps_state: true
  query_and_answer_write: false

training:
  mechanism_writer: pretrained_then_frozen
  freeze_A_in_fixed_writer_experiment: true
  joint_training_control: separate_experiment
  fixed_depth_smoke_tests: [1, 4]
  mixed_depth_values: [1, 2, 4]
  mixed_depth_probabilities: [0.333333, 0.333334, 0.333333]
  optimizer: adamw
  learning_rate_initial: 0.0002
  learning_rate_validation_candidates: [0.0001, 0.0002]
  weight_decay_initial: 0.01
  gradient_clip_norm: 1.0
  micro_batch_size_initial: 1
  effective_batch_size_target: 16
  max_tokens_per_sequence_initial: 1024
  max_train_tokens_pilot: 1000000
  memory_detach_policy: none_in_reference
  checkpoint_policy: record_config_and_data_state
  seed_screening: [17]
  seeds_confirmation: [17, 29, 43]

synthetic_data:
  task: unique_answer_relation_composition
  train_hops: [1, 2, 3]
  eval_hops: [1, 2, 3, 4, 5, 6, 7, 8]
  rank_screening: [16, 32]
  control_total_edges_across_hops: true
  control_context_length_across_hops: true
  random_entity_names: true
  split_unit: graph_or_episode
  counterfactuals: [replace_key_fact, remove_required_fact, swap_memory]

evaluation:
  depths: [0, 1, 2, 4, 8]
  checkpoint_policy: same_checkpoint_depth_sweep
  mechanism_inputs: identical_x_u_M
  generation: greedy
  free_generation_uses_gold_prefix: false
  answer_token_budget: fixed_per_task
  memory_access_protocols: [context_retained, memory_only_support]
  metrics: [exact_match, hop_stratified_accuracy, latency, peak_memory]
  report_prefill_and_decode_separately: true
  report_memory_and_backbone_kv_separately: true
  bootstrap_unit: graph_or_episode

gates:
  thresholds_are_proposed_not_results: true
  depth_gain_pp_initial: 3.0
  maximum_one_hop_drop_pp_initial: 1.0
  confidence_interval_required_positive: true
  verify_actual_latency_before_efficiency_claim: true
~~~

配置说明：

- 训练前检查生成任务的实际 token 长度；超长样本不能静默截断关键事实。
- 提高序列长度、batch size 或增加模块前，先测实际内存与吞吐。
- 模型规模、rank 与每层位置是实验条件，不是默认为最优的结论。
- L=0 不读 memory，适合作为同 checkpoint 的结构消融，不等于单独训练到最优的 LoRA 基线。
- 配置中的 1,000,000 个训练 token 是首轮上限草案；达到上限仍未学会单跳时先诊断，不把欠训练当作架构定论。

---

## 附录 C：主张—证据登记表

| 主张 | 当前状态 | 所需证据或限定 |
| --- | --- | --- |
| 每次额外 reader 为 O(r²) | 由定义可得 | 实现不构造 UMQ；报告实际常数 |
| 主干算子只执行一次 | 设计要求 | forward 调用计数和 profiler |
| 新增推理 memory 为每模块每序列 r² | 由定义可得 | 乘模块数、batch、dtype；不含 KV |
| 循环不增加固定 B 的输出子空间 | 代数结论 | 历史状态与模型定义的限定清楚 |
| 线性循环生成记忆多项式 | 代数结论 | 固定 M 时可预计算；不夸大必须循环 |
| B=0 可以启动训练 | 条件性结论 | alpha 非零，首步 B 梯度与后续内部梯度测试 |
| 收缩条件下具有停止误差界 | 标准理论应用 | 有效范数上界；不保证任务正确 |
| 多次演化查询改善组合检索 | 未验证 | 固定 x/u/M 的配对实验 |
| 同 checkpoint 具有可调预算收益 | 未验证 | 训练内深度扫描 |
| 训练外深度泛化 | 未验证 | 测试超过训练范围的 L |
| 优于增大 rank 的单步模型 | 未验证 | 匹配实际时间或资源的对照 |
| 提高真实 LLM 任务表现 | 未验证 | 多 seed、明确访问协议的真实任务 |
| 方法具有充分研究创新性 | 待整体证据评估 | 近邻核查、具体机制与实验共同支持 |

---

## 附录 D：参考文献与核查入口

引用编号对应正文。对预印本固定本次阅读版本，后续实验前再次核对版本变化。以下链接用于追溯来源，不表示所有工作都必须完整复现。

### [R1] RecLoRA

Learning to Re-think: Gated Recurrence with LoRA for Efficient and Effective Domain Incremental Learning.

- 核查依据：用户上传的 20567_Learning_to_Re_think_Gat(1).pdf，11 页。
- 上传稿标注为 ICLR 2026 under review；本文不据此推定最终录用状态。
- [OpenReview 页面](https://openreview.net/forum?id=O2n7oXjEau)
- [论文链接](https://openreview.net/pdf?id=O2n7oXjEau)
- 重点位置：第 4–6 页，§3.2–3.3、Figure 1、Algorithm 1；第 9 页 Table 2 与 Figure 2。
- 用途：共享递归 LoRA、router、step embeddings、域增量适配及实现边界。

### [R2] LoRA

Edward J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models.

- [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- 用途：静态低秩适配与合并基线。

### [R3] 因果线性注意力

Angelos Katharopoulos et al. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.

- [官方论文页](https://proceedings.mlr.press/v119/katharopoulos20a.html)
- 用途：因果线性注意力的递归状态形式。

### [R4] Fast Weight Programmers 与 delta 更新

Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber. Linear Transformers Are Secretly Fast Weight Programmers. ICML 2021.

- [官方论文页](https://proceedings.mlr.press/v139/schlag21a.html)
- 用途：线性注意力与快权重的联系、有限记忆与 delta 型更新。

### [R5] 固定记忆上的多次读取

Jimmy Ba et al. Using Fast Weights to Attend to the Recent Past. 2016.

- [作者托管论文](https://www.cs.toronto.edu/~hinton/absps/FastWeights.pdf)
- [arXiv:1610.06258](https://arxiv.org/abs/1610.06258)
- 重点位置：§3，内循环与 layer-normalized fast weights。
- 用途：固定快权重、输入持续注入、多次 inner-loop 读取已有先例。

### [R6] Generative Adapter

Generative Adapter: Contextualizing Language Models in Parameters with A Single Forward Pass.

- [核查版本：arXiv:2411.05877v1](https://arxiv.org/html/2411.05877v1)
- 重点位置：§2.2，双线性生成器与流式状态更新。
- 用途：上下文统计生成低秩适配权重的直接近邻。

### [R7] δ-mem

δ-mem: Efficient Online Memory For Large Language Models.

- [核查版本：arXiv:2605.12357v2](https://arxiv.org/html/2605.12357v2)
- 重点位置：§3.1–3.5，投影、旧状态读取、低秩修正与写入粒度。
- 用途：冻结 LLM 的在线关联记忆与 attention 适配。严格复现时核对其具体门控与误差计算次序。

### [R8] Deep Equilibrium Models

Shaojie Bai, J. Zico Kolter, Vladlen Koltun. Deep Equilibrium Models. 2019.

- [arXiv:1909.01377](https://arxiv.org/abs/1909.01377)
- 用途：固定点网络与隐式深度的理论背景；不把标准不动点分析重新认领为原创。

### [R9] Equilibrium Language Models

Yikun Jiang et al. Equilibrium Language Models. ICLR 2026.

- [官方论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/6041ba59d44d7d1c150334b8bb02497c-Paper-Conference.pdf)
- 重点位置：§3.1、§3.4，固定点模块、LoRA 训练与求解停止。
- 用途：固定点语言模型与 LoRA 的近邻边界。

### [R10] LT2

LT2: Linear-Time Looped Transformers.

- [核查版本：arXiv:2605.20670v1](https://arxiv.org/html/2605.20670v1)
- 重点位置：§2、§6。
- 用途：循环 Transformer 中的线性/稀疏 mixer；不能以“循环线性注意力”单独主张首创。

### [R11] Elastic Latent Recurrence

One Model to Loop Them All: Elastic Latent Recurrence for Language Models.

- [核查版本：arXiv:2610.12582v1](https://arxiv.org/html/2610.12582v1)
- 重点位置：§3，混合深度与循环条件化低秩调制。
- 用途：可调计算与低秩循环条件化的近邻。
- 名词注意：该文 one-loop rank overrides 中的 rank 指 data-parallel 进程，不是 LoRA 的低秩维数。

### [R12] OperLoop

Looped Transformers as Optimizers.

- [核查版本：arXiv:2609.37379v1](https://arxiv.org/html/2609.37379v1)
- 重点位置：§3–4。
- 用途：从投影、目标和优化器构造循环转移；优化视角与 delta correction 已有直接研究。

### [R13] LLM Sleep

Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference.

- [核查版本：arXiv:2605.26099v2](https://arxiv.org/html/2605.26099v2)
- 重点位置：§5、Algorithm 1。
- 用途：把额外计算用于记忆形成；与固定记忆上的查询计算分开比较。

### [R14] Gated Delta Networks

Songlin Yang, Jan Kautz, Ali Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule.

- [核查版本：arXiv:2412.06464v1](https://arxiv.org/html/2412.06464v1)
- 用途：遗忘与 delta 写入，以及专门的分块并行训练思路。不能默认其 kernel 原封不动支持本文的多次写前读取。

### [R15] MuSiQue

MuSiQue: Multihop Questions via Single-hop Question Composition. TACL 2022.

- [官方论文页](https://aclanthology.org/2022.tacl-1.31/)
- [作者代码与数据仓库](https://github.com/stonybrooknlp/musique)
- 用途：真实语料多跳问答候选；本文的 memory-only 协议需要单独标注。

### [R16] HotpotQA

HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018.

- [官方论文页](https://aclanthology.org/D18-1259/)
- [项目主页](https://hotpotqa.github.io/index.html)
- 用途：支持事实与多文档组合问答的补充评测。

### [R17] 小模型候选

Qwen/Qwen3-0.6B-Base（执行中由 Qwen/Qwen3-0.6B 调整，见 E.2）。

- [官方模型页](https://huggingface.co/Qwen/Qwen3-0.6B-Base)，固定 revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
- 用途：降低首轮 LLM 试错成本；运行前固定模型与 tokenizer revision。

### [R18] 确认规模候选

Qwen/Qwen3-1.7B-Base（HEAD `ea980cb`，已缓存）或 Qwen/Qwen3-4B-Base（HEAD `906bfd4`），P5 前选定。

- [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base)，[Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base)
- 用途：核心效应确认后的规模验证；不在机制未成立时直接扩大模型。

### [R19] End-to-End Memory Networks

Sainbayar Sukhbaatar et al. End-To-End Memory Networks. NeurIPS 2015.

- [arXiv:1503.08895](https://arxiv.org/abs/1503.08895)
- 用途：多 hop 记忆读取与查询更新的经典先例（2026-10-26 补充）。

### [R20] MemReasoner

- [arXiv:2503.07903](https://arxiv.org/abs/2503.07903)
- 用途：矩阵记忆上的迭代读取与查询更新（§3.3），端到端微调（2026-10-26 补充）。

### [R21] Retrofitting Recurrent Depth into a Pretrained LM

- [arXiv:2608.11233](https://arxiv.org/abs/2608.11233)
- 用途：冻结基座 + 跨循环共享 LoRA 的循环深度近邻，无记忆（2026-10-26 补充）。

版本核查（2026-10-26）：[R10] 已至 v2（仅许可证变化），[R13] 已至 v3（仅引用修订），[R14] 已至 v3（ICLR 2025 定稿，结构调整，门控 delta 公式不变）；其余预印本与所列版本一致。详见 `docs/reports/lit_check.md`。

---

## 文档完成状态

本文件完成了研究问题、候选模型、公式与维度、创新边界、训练与状态协议、基线与消融、数据与统计、工程风险、阶段判定、伪代码和参考文献的整理。

本文没有声称已完成新方法训练、真实模型性能测量或正式的全球新颖性证明。下一项实际工作是执行 P0 与首轮单模块机制实验，并用结果决定是否继续扩大研究范围。



