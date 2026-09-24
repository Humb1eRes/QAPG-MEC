# Yang GUPA 向共同仿真环境迁移的独立审查

## Material Passport

- 审查日期：2026-09-18。
- 用户目标：保留 Yang 原文中可辨识的效用与更新机制，加入共同环境对照；不以另一个势函数冒充原方法。
- 来源：Yang et al., *QoE-Aware User Allocation in NOMA-Enabled MEC Systems: A Distributed Game-Theoretical Approach*, IEEE TVT 75(2), 3251–3264 (2026), DOI 10.1109/TVT.2025.3605977。
- 来源文件：`../../literature/baseline-audit/yang-tvt/Yang_2026_GUPA_TVT.pdf`，用户提供的完整 14 页论文。已有 `READ_PREFLIGHT.json` 为 PASS；本审查使用逐页提取文本，并视觉核对 PDF 第 5、7、9 页，即印刷页 3255、3257、3259。
- 代码审查：`common_simulator.py`、`eedo.py`、`PROTOCOL.md` 及 `../reference/edgesport_sim.py`。
- 本文是实现前的源公式/适配审查，不是已执行实验的结果证明。未修改共享代码或稿件。

## 结论

可实现并清楚命名为 **GUPA-O: an orthogonal-link adaptation inspired by GUPA**。它保留原传输时延的负对数效用、初始全部未分配、逐用户枚举候选、以系统总效用评估改进、一次提交一个竞争获胜者、无改进即停止这些可辨识机制。在本文独立正交上行和独立设备功率上限下重新求解功率，得到每个已分配用户使用最大可行功率；这一变化及其原因必须写明。

该适配必然退化为最强信道/最大功率选择。因此，实施、验证后可以替换或合并原 strongest-channel 的公开标签，**不能把两条完全相同的轨迹计为两个独立算法证据**。这满足“原机制迁移到共同环境”的有限目标，但不等于复现 Yang 的 NOMA 实验，也不能支持“本文算法优于 Yang 原完整 GUPA”的结论。

## 原文到底优化什么、怎样更新

| 项目 | 可核实的原机制 | 定位 |
|---|---|---|
| 传输方向 | 基站/边缘服务器向用户下行发送；不是设备向服务器卸载的上行链路 | PDF 2–4，II-A、II-B，图 1、Eq. (3)–(6) |
| 功率所有者 | 服务器给各用户分配发射功率；所有信道总功率受服务器预算约束 | PDF 2 Def. 2；PDF 3 Eq. (4) |
| 无线耦合 | 同信道用户的 NOMA 干扰、其他基站的同信道干扰、SIC 解码顺序 | PDF 4，Eq. (6)–(12) |
| 用户动作 | 服务器、信道、分配功率三元组；未分配为 (0,0,0) | PDF 3 Def. 3；PDF 5 Eq. (18) 后说明 |
| 时延 | `T_i = D_i / r_i`，只计传输时延，没有计算时延、排队时延或能耗 | PDF 4 Eq. (17) |
| 个体效用 | `E_i = -alpha log(T_i) + beta`，声明 alpha > 0；未分配效用 0 | PDF 5 Eq. (18)–(23) |
| 问题目标 | 最大化所有用户的总 QoE | PDF 5 Eq. (24) |
| 博弈定义 | 个体 QoE 最优反应和 Nash 条件 | PDF 5 III-A，Eq. (25) |
| 实际算法候选评价 | 每个用户选择使 **系统总 QoE** 最大的候选；仅总 QoE 严格改进者竞争更新 | PDF 7 Algorithm 1，第 4、11、13–15 行 |
| 原候选功率 | 先核对信道剩余功率至少为 `p_max`，候选试算采用 `p_min` | PDF 7 Algorithm 1，第 8–11 行；PDF 4 Eq. (14)、(15) |
| 原提交功率 | 获胜者提交新动作，并把 `p_min` 升为 `p_max` | PDF 7 Algorithm 1，第 16–18 行 |
| 竞争规则 | 描述了竞争与单一获胜更新，但未指定“最大改进获胜”、随机抽样概率或固定索引优先 | PDF 7 Algorithm 1，第 15、16 行与 IV-A |
| 停止 | 所有用户均无更新意愿；文中解释为到达 Nash equilibrium | PDF 7 Algorithm 1，第 19 行及 IV-A |

关键区分：理论定义用个体 QoE，Algorithm 1 的可执行比较明确写系统总 QoE。实现应对应 Algorithm 1，不能把自己的边际队列收益当作原个体 QoE，也不能将原算法宣称为“每次全局最大增益用户获胜”；最大增益/固定索引只能是适配时补充的可复现竞争规则。

## 原文中的实施歧义与处理

1. **Eq. (22) 的符号冲突。** 印刷公式为 `(gamma-delta)/(log Tmin-log Tmax)>0`，但在 `gamma>delta` 且 `Tmin<Tmax` 时该商为负。Eq. (18)、(20)、(21) 的负对数关系要求正斜率系数；若按端点重建，应使用分母 `log Tmax-log Tmin`。不能照抄负 alpha，使更长时延反而有更高效用。GUPA-O 可直接使用 alpha=1，并明示这是适配归一化。
2. **候选/提交功率不同。** Algorithm 1 第 10 行确实用 `p_min`，第 18 行用 `p_max`，不是文本抽取错误。不能说采用 `p_max` 试算的实现与原伪代码逐行相同。
3. **原功率及速率参数没有迁移数值。** Table II（PDF 9）列有服务器 46 dBm 总预算、20 MHz 带宽、噪声、任务大小等，但没有适配所需的统一 `r_min`、`r_max`、alpha、beta 数值。将服务器 46 dBm 当成每个设备发射功率会改变共同硬件环境。
4. **原潜在函数不是总时延效用的随意替代物。** Eq. (30) 使用共享信道功率乘积和未分配惩罚。独立正交链路去掉干扰后，原相关项退化；原证明、Nash/PoA/迭代上界不能直接移植到本文。

## 为什么不把原 `p_min`/`p_max` 规则机械套进来

若在独立上行中仍以统一最低速率反解 `p_min(n,m) = (2^(r_min/W)-1) noise/g_nm`，所有可行服务器的候选速率都恰好为 `r_min`。同一用户的候选 `D_i/r_min` 因而相同，服务器选择主要由遍历顺序决定。提交后再反解到共同 `r_max` 也不能恢复原 NOMA 外部性。

这不是已有队列调度博弈，也不是 strongest-channel 的自然推导；它是原混合试算/提交逻辑在新物理模型下的另一个退化。为使比较对象真正优化声明的传输效用，推荐明确重解共同环境的功率子问题，并让候选试算和提交使用相同的实际功率。不要隐去这一改动。

## 推荐实现合同：GUPA-O

### 保持共同 CPU 控制和记账

使用原 strongest-channel 的固定 V 本地 CPU 与边缘 CPU 规则，以便区别仅来自被审查的无线选择：

- `f = original.local_cpu(Q, cfg)`；本地完成量 `min(Q, tau*f/c)`；候选任务量 `D = max(Q-local,0)`。
- `F = original.edge_cpu(H,cfg)`；边缘完成量 `min(H,tau*F/c_edge)`，按已有来源账本分配。
- `D` 是当前槽待上传的剩余连续比特批次，**这是任务大小的动态适配，不是原静态 D_i 数据集**。
- 上传量 `min(D_n,tau*r_nm)`；新上传下一槽才能在服务器执行；所有策略使用共同全时隙无线能耗与立方 CPU 能耗。
- 本地 CPU、边缘 CPU 和来源账本不是原 GUPA 提出的方法，只是本文共同执行器。

### 具体函数与参数

建议独立实现 `gupa_orthogonal_decision(Q, H, B, gains, cfg)`，返回现有 decision 数据字段。

1. **可行链路**：`D_n>0`、`g_nm>0`、`p_max_w>0`；本环境各服务器均为可选，只有一个专用正交上行选项，因此不增加信道编号。
2. **功率**：`p_nm=p_max_w`。由 `d[-log(D/r)]/dp >0`、独立设备功率约束得到，无原共享预算。候选和提交用同一值；无水填充、无队列势函数。
3. **速率**：`r_nm=W log2(1+p_max_w*g_nm/noise_w)`。
4. **效用**：在每槽开始搜索之前，固定 `T_max = max_{n:D_n>0} D_n / min_{(n,m):feasible} r_nm`；采用 `q_nm=1+log(T_max/(D_n/r_nm))`，未分配 `q_n(null)=0`。等价于 alpha=1、beta=1+log(T_max) 的原负 log 传输时延形式。实际实现宜用 log 差避免不必要的比值溢出。
5. **归一化边界**：没有正工作量或没有正速率时直接返回空卸载；不要对零任务取对数。T_max 在候选枚举期间固定，不能每换一个候选重新归一化。
6. **初始化**：每槽开始所有用户未分配，所有无线功率/上传量为零。
7. **每轮建议**：每个用户枚举服务器，选使系统总效用最大的新动作。由于无外部性，总效用变化正好是该用户新旧 q 的差。可以据此优化运算，但保留可审计的总效用轨迹。
8. **竞争提交**：在严格正改进者中选一个获胜者；可用最大改进、相同改进时最小用户索引。注明这是原文未指定部分的确定性补充，不能归于 Yang 的原始赢家规则。
9. **停止**：没有正改进者即停。每个用户至多从 null 更新一次到自己的最高速率服务器，故最多活跃用户数次提交，无需人为截断到 12 轮；不能误用 QAPG 的 12 sweeps 作为此处 12 次单用户提交。

这里给每个可行分配正效用，是**明确的可行用户接纳约定**。由于 null 仍为零，不能笼统称“对整个博弈做保持策略不变的正仿射变换”：只给 assigned 动作加偏置会影响接纳相对于 null 的比较。它不声称复现原 QoE 标度或 MOS 分数。

### 退化等价的推导

对给定活跃用户 n，D_n 和该槽 T_max 不依赖服务器。q_nm 严格随 r_nm 增加，r_nm 严格随 g_nm 增加，故 `argmax_m q_nm = argmax_m g_nm`。每个可行分配 q>=1，大于未分配的零；用户接入后不会因其他用户更新而改变 q，因为没有干扰、共享功率、共享上行带宽或服务器效用项。因此所有活跃用户依次选择其最强信道并用 p_max，提交顺序不影响结果。

若 CPU、队列更新和上传裁剪与旧 strongest-channel 相同，则所有非计时状态和能耗轨迹也相同。这里是对**本文明确适配**的简单代数结论，不是原 GUPA 的 NOMA 最优性结论。

## 必须验证的项目

- 零队列、零功率、零增益输入不产生 NaN 或负上传；这些输入与旧强信道代码的空动作标签可能不同，但物理零服务结果应一致。
- 固定候选时，功率与速率/效用单调；分配的 QoE 为正；归一化在整轮内不随当前动作变化。
- 每次只提交一个用户，总效用严格增大；停止后无未执行的正增益建议。
- 所有更新次数不超过活跃用户数；候选并列时有确定性规则。不要把浮点近似并列擅自扩大到使非最强信道获选。
- 小系统与穷举全组合的独立效用最大值一致；用户顺序变化不改变非并列的最终关联。
- 与旧 strongest-channel 在生产输入逐槽核对关联、功率、上传、CPU、完成量、所有队列与全部能耗分量。允许明确记录仅源自不同数学库的舍入容差，不用百分比容差掩盖真实策略差异。
- 按既定 3 负载 × 5 种子补足 15 组，不另调参数；记录资源限制与流量守恒。旧标签的 3 种子/短窗口也应由同一轨迹复核，不只比较最后均值。

## 稿件与图例的准确说法

推荐首次出现：

> GUPA-O is an orthogonal-link adaptation of the delay-utility and single-winner update framework of Yang et al.'s GUPA. We retain its negative-log transmission-delay utility and total-utility improvement rule, and re-solve power allocation under the independent device power constraints of our system. Candidate evaluation and commitment use the same optimized power. With orthogonal links and no shared transmission budget, the adapted policy reduces to strongest-channel selection at maximum power.

在方法名较短的图中可写 `GUPA-O`，图注/方法节说明原 strongest-channel 被这个经验证的特例替代。不要同时把 `Strongest-channel` 和 `GUPA-O` 当作两个不同基线计数，也不要说“we reproduce Yang et al.'s GUPA”或承接原论文的 Nash、PoA、分布式通信开销结论。

本文比较的是该正交适配在共同工作负载下产生的能耗、积压和完成量；原 GUPA 的效用不惩罚能耗或排队，因此这些指标不应表述为原论文自身目标的复现。
