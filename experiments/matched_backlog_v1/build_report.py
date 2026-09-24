"""Build the Chinese result report from complete validated comparison tables."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];RESULTS=ROOT/'experiments/results/matched_backlog_v1';OUT=RESULTS/'analysis'
def read(p):
    with p.open() as h:return list(csv.DictReader(h))
def fmt(v,n=3):return f'{float(v):.{n}f}'
def main():
    a=json.loads((OUT/'ANALYSIS.json').read_text());s=a['all_target_summaries'];rows=read(OUT/'HOLDOUT_COMPARISON.csv');pairs=read(OUT/'HOLDOUT_PAIRED_DIFFERENCES.csv')
    q=[r for r in rows if r['method']=='QAPG-R'];qp=[r for r in q if r['status']=='PASS'];failed=[r for r in rows if r['status']=='TEST_BACKLOG_FAIL'];nf=[r for r in rows if r['status']=='NO_FEASIBLE_GRID_POINT']
    comparable=[r for r in s if r['energy_saving_vs_best_percent']!=''];savings=[r['energy_saving_vs_best_percent'] for r in comparable]
    lines=['# QAPG-R：相同积压要求下的能耗比较','',
      f'修正后的 QAPG-R 在 {len(qp)}/12 个预先设定的“负载 × 积压门限”条件下，通过了全部五组独立测试种子的积压要求。在 {a["qapg_lowest_energy_cells"]}/{len(comparable)} 个具备合格对照的达标条件下，其平均能耗低于所有达标对比方法。',
      '',f'与各条件下平均能耗最低的达标对照相比，能耗降低 {min(savings):.2f}%–{max(savings):.2f}%。这是本轮公共参数网格及仿真条件下的观察结果。','',
      '## 算法修正','',
      '原算法先选择本地处理和候选上传量，再调整服务器分配；服务器负载代价未能充分参与候选上传量的优化。本轮将本地处理量、上传量、发射功率和服务器选择放入同一设备更新中，并交替优化服务器处理量。每次更新都计算下一时隙队列与能耗的共同目标。','',
      '新控制器使用固定能耗权重 V，移除了原算法的自适应 V 和几个启发式压力系数。边缘队列权重固定为 M/N=0.1，用于将汇聚队列按名义来源数量归一化；该值在轨迹实验前确定。所有方法都统一将 CPU 频率限制到实际选定处理量所需的频率，保留相同的三次方 CPU 能耗和整时隙发射能耗模型。','',
      '内部开发名称为 QAPG-R，用来区分这次实质性算法修改与旧版 QAPG。完整目标函数、约束、更新方程及停止条件见代码目录中的 CONTROLLER_DESIGN.md。旧版 QAPG 和去掉二次负载项的 NoQuad 均保留为对照。','',
      '## 公平比较规则','',
      '| 项目 | 固定规则 |','|---|---|',
      '| 环境 | 100 个设备、10 个服务器；全部方法共享硬件约束、到达与信道轨迹、队列更新顺序和能耗口径 |',
      '| 方法 | QAPG-R、原 QAPG、NoQuad、GUPA-O、EEDO、BP-Greedy；所有方法统一 CPU 修正 |',
      '| 负载 | β=1、2、3，对应期望到达量 4.8β Mbit/slot |',
      '| 积压要求 | 平均总积压不超过 B*=4.8βρ Mbit；ρ=1.5、2、3、5 |',
      '| 调参预算 | 每方法、每负载使用相同九档 V 比例：1/16 至 16 |',
      '| 选参 | 两组种子，每组 1500 槽，去掉前 300 槽；两组都达标才可入选，再取平均能耗最低者 |',
      '| 独立测试 | 五组新种子，每组 3000 槽，去掉前 300 槽；锁定参数后运行，不因测试失败换参数 |',
      '| 达标与统计 | 五组种子的各自平均积压均不超过门限才算达标；报告种子间均值与样本标准差 |','',
      '“相同积压要求”是相同上限约束，不要求不同方法恰好产生一样的实测积压。上传到服务器的任务不算完成，只有本地或边缘真正处理的工作量才计入完成量。ρ 是工作量归一化的实验门限，不是逐任务时延保证。','',
      f'正式选参完成 324 次运行，锁定 45 个不同配置，随后完成 {a["test_runs"]} 次独立测试。共有 72 个方法—负载—门限组合：{a["cell_status_counts"]["PASS"]} 个测试达标，{len(nf)} 个在公共选参网格内无可行点，{len(failed)} 个在测试中未达标。以下全部保留。','',
      '## 全部 12 个条件','',
      '能耗单位为 J/slot。表中对照是该门限下通过测试的方法中平均能耗最低者。QAPG-R 未达标时不计算节能比例。','',
      '| β | ρ | 上限 Mbit | QAPG-R 能耗 | QAPG-R 积压均值 / 最差种子 | 最低能耗达标对照 | 对照能耗 | 节能 |','|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in s:
        status='' if r['qapg_status']=='PASS' else ' **未达标**'
        energy=f'{r["qapg_energy_mean"]:.3f} ± {r["qapg_energy_sd"]:.4f}' if r['qapg_energy_mean']!='' else '无可行点'
        backlog=f'{r["qapg_backlog_mean"]:.3f} / {r["qapg_backlog_max"]:.6f}' if r['qapg_backlog_mean']!='' else '—'
        saving=f'{r["energy_saving_vs_best_percent"]:.2f}%' if r['energy_saving_vs_best_percent']!='' else '—'
        ce=f'{r["best_comparator_energy_mean"]:.3f} ± {r["best_comparator_energy_sd"]:.4f}' if r['best_comparator_energy_mean']!='' else '—'
        lines.append(f'| {r["beta"]} | {r["rho"]:g} | {r["backlog_target_mbit"]:.1f} | {energy}{status} | {backlog} | {r["best_comparator"]} | {ce} | {saving} |')
    tight=min(qp,key=lambda r:float(r['backlog_target_mbit'])-float(r['backlog_max']))
    lines+=['',f'最接近门限的是 β={tight["beta"]}、ρ={tight["rho"]}：最差测试种子的平均积压为 {float(tight["backlog_max"]):.9f} Mbit，低于上限 {float(tight["backlog_target_mbit"])-float(tight["backlog_max"]):.9f} Mbit。其五组测试均达标，但余量很小。', '', '![独立测试：相同积压门限下的能耗](figures/heldout_energy_at_matched_backlog.png)','','## 六种方法的完整结果','','每格为能耗均值 ± 种子间样本标准差。† 表示测试积压未达标，其能耗不参与该门限的排名；“无可行点”仅指本轮九档参数网格。']
    methods=['QAPG-R','QAPG-capped','NoQuad-capped','GUPA-O-capped','EEDO-adapted','BP-Greedy-capped']
    for beta in [1,2,3]:
        lines+=['',f'### β={beta}','','| ρ | '+' | '.join(methods)+' |','|---:|'+'---:|'*len(methods)]
        for rho in [1.5,2,3,5]:
            vals=[]
            for method in methods:
                r=next(r for r in rows if int(r['beta'])==beta and float(r['rho'])==rho and r['method']==method)
                vals.append('无可行点' if r['status']=='NO_FEASIBLE_GRID_POINT' else f'{fmt(r["energy_mean"])} ± {fmt(r["energy_sd"],4)}'+(' †' if r['status']=='TEST_BACKLOG_FAIL' else ''))
            lines.append(f'| {rho:g} | '+' | '.join(vals)+' |')
    lines+=['','### 未达标情况','','| 方法 | β | ρ | 积压上限 | 最差测试种子的平均积压 | 超出上限 | 达标种子数 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in failed:lines.append(f'| {r["method"]} | {r["beta"]} | {r["rho"]} | {fmt(r["backlog_target_mbit"])} | {fmt(r["backlog_max"],6)} | {fmt(r["worst_seed_excess_mbit"],6)} | {r["pass_seeds"]}/5 |')
    if not failed:lines+=['| 无 | — | — | — | — | — | — |']
    lines+=['','网格内无可行点：'+('；'.join(f'{r["method"]}，β={r["beta"]}，ρ={r["rho"]}' for r in nf) if nf else '无')+'。','',
      '## 运行与复核','',
      '全部原始轨迹的积压、能耗、完成量、分项能耗和逐槽守恒已重新计算。独立选择审计检查了 324 条轨迹、162 个网格摘要、648 条门限资格记录和 72 项参数选择。算法验证包括独立标量最优化、势函数差值、资源约束及来源队列账本；两个预先指定的确定性重跑案例中，所有非计时数组完全一致。','']
    conv=[float(r['qapg_convergence_fraction_mean']) for r in q if r['qapg_convergence_fraction_mean']!=''];gain=[float(r['qapg_max_best_deviation_gain']) for r in q if r['qapg_max_best_deviation_gain']!='']
    ratios=[float(r['completion_ratio_mean']) for r in q if r['completion_ratio_mean']!=''];slopes=[float(r['late_slope_max']) for r in q if r['late_slope_max']!='']
    lines += [f'QAPG-R 已测试配置的逐槽收敛比例（每配置五种子均值）为 {min(conv)*100:.2f}%–{max(conv)*100:.2f}%；最大剩余单设备改进量为 {max(gain):.3g}（归一化目标单位）。固定 12 次扫描结束时仍有部分时隙未达到收敛容差，运行日志保留这些情况。', '',
      f'QAPG-R 的配置级平均完成/到达比为 {min(ratios):.6f}–{max(ratios):.6f}，各种子后半程积压斜率的最大值为 {max(slopes):.6g} Mbit/slot。这些是有限窗口诊断，未被用于额外筛选。', '',
      'GUPA-O 保留 Yang 原文可辨识的效用与更新机制，但使用本文的正交上行链路；EEDO 也按共同物理模型作了明确迁移。两者的标签与说明均保留“适配”范围。完整映射见 GUPA_MAPPING.md、EEDO_MAPPING.md。', '',
      '本轮证据支持在列出的仿真条件、公共参数网格和积压上限下比较能耗；不将有限次块更新、五组随机种子或单次排序解释为全局最优和长期稳定性证明。新算法尚未替换旧 SR3 投稿稿中的公式与实验，后续正文须按本轮新方法、完整达标结果同步修改。', '',
      '## 文件索引','',
      '- `HOLDOUT_COMPARISON.csv`：全部 72 格，含所选参数、积压均值/最差值、达标种子数、能耗和完成量诊断。',
      '- `HOLDOUT_PAIRED_DIFFERENCES.csv`：同种子的配对能耗差与全部合格对照比较。',
      '- `ALL_TARGET_SUMMARY.csv`：12 个门限条件的汇总。',
      '- `figures/`：PDF、SVG 矢量图及 PNG 预览。',
      '- `../selection/`、`../test/`：输入、完整逐槽结果、每次运行记录与冻结选择。',
      '- `../../../matched_backlog_v1/REPRODUCE.md`：算法、依赖和复现说明。', '',
      '## Material Passport','',
      '- Artifact: experiment result and descriptive statistical validation; language zh-CN.',
      '- Mode: run / validate; data: local simulations, no human participants.',
      '- Verification: all saved runs audited; two designated deterministic replays passed. No claim of replaying every run.',
      '- Independent units: five test seeds; no p-values or confidence guarantees.',
      f'- Generated UTC: {datetime.now(timezone.utc).isoformat()}.','',
      '## 统计解释核查（11/11）','',
      '| 核查项 | 本报告处理 |','|---|---|',
      '| Simpson 反转 | 逐负载、逐门限报告，不用混合均值代替条件结果 |',
      '| 生态谬误 | 聚合积压不解释为某用户或某任务的时延 |',
      '| Berkson 选择偏差 | 可行性规则事先固定，完整保留被排除出能耗排名的失败项 |',
      '| 碰撞变量偏差 | 无额外回归或事后条件调整；只使用预设积压约束 |',
      '| 基础率忽略 | 不作诊断概率推断，达标种子分子分母明确 |',
      '| 均值回归 | 选参和测试种子分离，测试失败不重选 |',
      '| 幸存者偏差 | 全部网格点、未达标目标和无可行点均保留 |',
      '| 多重寻找效应 | 完整报告 12 条件及六方法，不报告挑选后的显著性 |',
      '| 分析路径自由度 | 冻结算法与协议、选参规则及测试前的参数文件；旧开发分支另存 |',
      '| 相关与因果 | 结论限于共同仿真输入下的控制方法比较，不外推实际部署因果效应 |',
      '| 反向因果 | 时隙决策只使用当时队列、信道及配置到达均值，不读取未来随机到达 |','']
    (OUT/'RESULTS_REPORT.zh-CN.md').write_text('\n'.join(lines))
    print(OUT/'RESULTS_REPORT.zh-CN.md')
if __name__=='__main__':main()
