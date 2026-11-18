"""Render actual development/validation results; no invented missing endpoints."""
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'benchmark'))
from physcheck.relations.contracts import save,sha

LABELS={'cotracker_native':'CoTracker3 native','cotracker_cycle':'CoTracker3 reverse gate',
        'mast3r_raw':'MASt3R forced matching','mast3r_cycle':'MASt3R reciprocal gate'}

def load(file):return json.loads(file.read_text())
def pct(x):return 'N/A' if x is None else f'{x*100:.2f}%'
def fraction(counts,a,b):return f"{counts[a]}/{counts[b]}（{pct(counts[a]/counts[b] if counts[b] else None)}）"

def table(summary):
    rows=['| 方法 | 接受精度 | 正确保留率 | 全可见点坐标准确率 | 接受的重现恢复 | 假可见 |',
          '|---|---:|---:|---:|---:|---:|']
    for arm,label in LABELS.items():
        c=summary['arms'][arm]['counts']
        rows.append('| '+label+' | '+' | '.join(fraction(c,a,b) for a,b in
            [('correct','accepted'),('correct','positive'),('located_all_visible','positive'),('recovered','returned_points'),('false_visible','negative')])+' |')
    return '\n'.join(rows)

def pairing_text(p):
    c=p['all_visible_pairing']
    return (f"标注重现事件 {p['return_events']} 次，分布在 {p['return_participants']} 位参与者。"
        f"原生坐标正确 {p['native_return_coordinates_correct']} 次，MASt3R 坐标正确 {p['descriptor_return_coordinates_correct']} 次；"
        f"新增正确重现坐标 {p['new_return_coordinates']} 次，丢失原有正确重现坐标 {p['lost_return_coordinates']} 次。"
        f"新增被接受的正确恢复 {p['additional_accepted_returns']} 次，其中 {p['additional_accepted_returns_already_located_by_native']} 次原生坐标已经正确。\n\n"
        f"全部 {c['positive']} 个已知可见且坐标有效的点时刻中，两方法都定位正确 {c['both_located']} 个，"
        f"仅原生正确 {c['native_only_located']} 个，仅 MASt3R 正确 {c['descriptor_only_located']} 个，均不正确 {c['neither_located']} 个。"
        "这是按固定4/256阈值做的配对诊断，没有用新结果重选模型或阈值。")

def main():
    output=ROOT/'report';output.mkdir(exist_ok=False)
    development=load(ROOT/'analysis_development/summary.json')
    dev_pair=load(ROOT/'pairing_development/summary.json')
    selection=load(ROOT/'selection.json');manifest=load(ROOT/'manifest_validation.json')
    new=load(ROOT/'analysis_validation/summary.json') if manifest['cases'] else None
    new_pair=load(ROOT/'pairing_validation/summary.json') if new else None
    now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    versions={}
    for name in ['torch','torchvision','numpy','opencv-python','requests','matplotlib','huggingface-hub','einops','roma','pytest']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    save(output/'runtime.json',{'generated_at':now,'python':sys.version,'packages':versions,'seed_bootstrap':20261117,
                              'bootstrap_draws':10000,'api_inference_cost_usd':0,'local_compute_cost_measured':False})
    splits=[('Development only',development,ROOT/'analysis_development/per_video.json')]
    if new:splits.append(('New participants; return-enriched',new,ROOT/'analysis_validation/per_video.json'))
    fig,axes=plt.subplots(len(splits),3,figsize=(13,3.7*len(splits)),squeeze=False)
    colors=['#3977b5','#80b3d8','#d98036','#b34d45']
    for row,(title,summary,path) in enumerate(splits):
        detail=load(path)
        for col,metric in enumerate(['precision','retention','recovery']):
            ax=axes[row,col]
            for i,arm in enumerate(LABELS):
                value=summary['arms'][arm]['pooled'][metric]
                if value is not None:ax.bar(i,100*value,color=colors[i],width=.65,alpha=.8)
                ci=summary['arms'][arm]['participant_bootstrap'][metric]['ci95']
                if ci is not None:
                    ax.vlines(i,ci[0]*100,ci[1]*100,color='black',linewidth=1)
                values=[r[metric]*100 for r in detail if r['arm']==arm and r[metric] is not None]
                jitter=np.linspace(-.19,.19,len(values))
                ax.scatter(i+jitter,values,s=12,c='#222222',alpha=.65,zorder=5)
            groups=summary['participant_clusters']
            group_label=f'total participants={groups}'
            if metric=='recovery':
                relevant=summary['arms']['cotracker_native']['participant_bootstrap']['recovery']['denominator_clusters']
                group_label=f'return participants={relevant}/{groups}'
            ax.set(ylim=(-2,105),xticks=range(4),xticklabels=['Co native','Co gate','MA raw','MA gate'],ylabel='Percent',
                   title=f'{title}\n{metric}; {group_label}')
            ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Bars: pooled counts; dots: individual clips; lines: descriptive participant bootstrap95% CI',fontsize=11)
    fig.tight_layout()
    for suffix in ['png','pdf','svg']:fig.savefig(output/('comparison.'+suffix),dpi=160)
    plt.close(fig)
    new_section='预定获取窗口内无符合条件的新参与者重现片段，新验证指标全部 N/A。开发结果不能替代外部验证。'
    if new:
        ci_rows=['| 方法 | 接受精度95%描述区间 | 保留率95%描述区间 | 重现恢复95%描述区间 |','|---|---:|---:|---:|']
        for arm,label in LABELS.items():
            cells=[]
            for metric in ['precision','retention','recovery']:
                interval=new['arms'][arm]['participant_bootstrap'][metric]
                ci=interval['ci95'];cells.append('N/A（不足或退化）' if ci is None else f'[{pct(ci[0])}, {pct(ci[1])}]')
            ci_rows.append('| '+label+' | '+' | '.join(cells)+' |')
        new_section=table(new)+'\n\n'+pairing_text(new_pair)+'\n\n'+'\n'.join(ci_rows)
    text=f'''# 参考帧全局重定位实验收尾

生成时间：{now}。研究目录保留开始日期20261117；本报告使用实际完成时间。

本轮完成公开 MASt3R 静态基线的真实推理及四方法比较。PointSt3R 官方分享链接已过期，
没有运行其动态微调或可见性权重。旧样本只用于开发检查，新验证与旧10位参与者隔离。
“往返可匹配”不等于人工身份正确；本轮不会据此升级物理测量器或公布生成模型排名。

## 设计、材料和单位

固定协议见 [PROTOCOL.md](../PROTOCOL.md)，方法适配与边界见
[METHOD_NOTES.zh.md](../METHOD_NOTES.zh.md)。同一公开 EgoPoints 作者归档，复用已校验旧区段，
最多取得首3GiB逻辑前缀；排除旧10位参与者，按归档顺序选每位新参与者首个完整、64–1200帧、
有人工标注出画→可见重现的片段，上限6位。此为重现富集样本，不是随机代表性测试。
没有新增人工标签，也没有把AI意见当作真值。参与者是聚类单位，点/帧不是独立重复。

实际前缀 {selection['bytes']:,} 字节，完整扫描 {len(selection['scanned'])} 段，
选入 {len(manifest['cases'])} 段 / {len({c['participant'] for c in manifest['cases']})} 位新参与者，
共 {sum(c['frames'] for c in manifest['cases'])} 帧、{sum(c['queries'] for c in manifest['cases'])} 个初始查询。
边界未完成片段 {selection['boundary_clip']} 不纳入。
原图缩放为512×288；归一化误差为 sqrt((dx×256/W)²+(dy×256/H)²)，严格小于4算定位正确。
仅在人工 vis_valids 已知的未来时刻评分，初始查询时刻及未知帧不进入分母。
所有计划内推理失败保留分母，不能通过删去失败视频提高结果。

## 新参与者验证

{new_section}

新样本既没有训练也没有调参。置信区间为参与者聚类重采样10000次，种子20261117，
只描述选中样本；不足两个参与者或退化分布不显示区间。尤其零假阳性重采样不能给出
总体假阳性风险上界。每段完整计数、失败状态见 `analysis_validation/`（若无样本则不产生该目录）。
条件选择、少量参与者、稀疏标签及未知预训练重叠均限制外推；没有进行显著性或因果推断。

## 旧开发样本：不充当新增验证

12段 / 10位参与者 / 119查询 / 5021帧。这里复用前轮冻结 CoTracker3 缓存，
对同样输入新跑 MASt3R；没有把复用结果计为新增生成视频或新人工重复。

{table(development)}

{pairing_text(dev_pair)}

MASt3R 往返门通过的51个错误位置说明自洽不足以保证正确。原生完整跟踪在全部可见点的
定位正确数是141/206，MASt3R 为116/206；原生接受的重现恢复12/16与新方法15/16的差异，
不能直接称为新增坐标恢复。示例采用固定规则：最大被接受位置误差，以及首个新增接受的正确重现，
不是代表性抽样，也没有用示例选择新阈值。

![固定四方法结果](comparison.png)

![开发集失配与接受行为示例](../development_examples_v2/examples.png)

## 实现、校验和失败

模型源码固定于 a03c21f9244f937e85e160971a255e2d529740bf；MASt3R PTH SHA256
`e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2`。
严格核对无缺失或意外权重键。原始PyTorch RoPE回退及上游 autocast 弃用警告保留在日志中，
没有修改发布源码或声称编译CUDA扩展。模型推理全部本地，外部推理API费用0；未计量电力和硬件成本。

六个匹配器反例测试覆盖亚像素、边界、并列最大值、非往返、零/非有限描述子和外观一致却可能缺席的点。
另有实际分析CLI测试，检验失败保留分母、未知标签、严格恢复与复制缓存重放。
真实24维描述子上，8个查询与作者原始采样及逐点余弦代码坐标完全一致，缓存和哈希见 `matching_audit/`。
扩大实现校验后，全部12段、23个图像对、227个请求查询均与作者逐点匹配一致，重复推理坐标也完全一致，见 `matching_audit_full/`。
全套测试的最新实际结果见 `tests_receipt.json`；没有把未运行检查称为通过。

独立 GLM/Qwen 同题审查与候选实现均保存了原始请求、输出、usage和时延。
两者候选采样器分别发生网格维度错误、NxD写入Nx2错误，审查文字也有误读。
这些失败已实际运行复现，不能包装成独立审查通过，详见 [REVIEW_RULINGS.md](../REVIEW_RULINGS.md)。
参与者隔离、源/输入/预测哈希和标签掩码由确定性最终审计再次核对。

最初源码直连/代理请求有SSL失败，已使用官方GitHub API和固定归档取得源码；
PointSt3R 分享页过期保留原HTML，不能称为已复现该模型。
原12并发下载缓慢；36并发尝试使连接重置增多，失败及部分字节均保留，随后回退8并发1MiB请求。
没有改1081、全局代理或系统服务。LSP basedpyright 未安装且用户已拒绝安装，本轮没有LSP通过声明。
更多获取状态、原始报错和审计路径见 `FAILURES_AND_LIMITS.md`。

## 结论和后续

当前证据支持将全局描述子对应作为一个明确的参考对照，而不是生产级身份恢复器。
开发结果尤其说明“接受更多正确位置”与“生成新的正确位置”必须分开；往返门也会自洽地接受错误对应。
新验证的真实坐标增益与损失以上述配对统计为准，不根据旧开发集15/16的单项数字宣布成功。

下一阶段优先比较有显式可见性/遮挡输出的动态跟踪方法，并使用新的参与者测试实际坐标增益，
同时保留当前失败样例和静态基线。先解决可见性、错误位置和覆盖的联合问题，再扩大生成实验。
Part A有效物理拟合、真实I2V完整验证、Part B裁判校准与真实域物理假阳性控制仍未完成；
本轮人工点标注测试不能替代这些物理学终点。

完整运行方法与复制重放见 [REPRODUCE.md](../REPRODUCE.md)。本地版本化交付路径和实际重放校验
写入 `handoff_receipt.json`；没有向GitHub或其他公开平台上传。
'''
    (output/'REPORT.zh.md').write_text(text)
    print(json.dumps({'report':str(output/'REPORT.zh.md'),'new_validation_cases':len(manifest['cases'])}))

if __name__=='__main__':main()
