"""Artifact-derived report and descriptive figure, never generator rankings."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
LABELS={'native':'Native','confidence_0995':'Frozen score >0.995','reciprocal':'Reciprocal check'}
COLORS={'native':'#23649b','confidence_0995':'#bb661c','reciprocal':'#277749'}


def pct(value):
    return 'N/A' if value is None else f'{value*100:.2f}%'


def interval(row,key):
    values=row['participant_bootstrap'][key]['ci95']
    return 'N/A' if values is None else '['+', '.join(pct(v) for v in values)+']'


def main(output):
    output.mkdir(exist_ok=False)
    summary=json.loads((ROOT/'analysis/summary.json').read_text())
    rows=json.loads((ROOT/'analysis/per_video.json').read_text())
    manifest=json.loads((ROOT/'manifest.json').read_text())
    tests=json.loads((ROOT/'tests_receipt.json').read_text())
    prefix=json.loads((ROOT/'archive_metadata/prefix_receipt.json').read_text())
    table=['|方法|正确/接受|接受精度（描述性95%区间）|正确点保留率（描述性95%区间）|遮挡/OOV误接受|重现恢复|严格恢复|经验门|',
           '|---|---:|---|---|---:|---:|---:|---|']
    for arm,result in summary['arms'].items():
        count=result['counts']
        rate=result['pooled']
        table.append(f'|{LABELS[arm]}|{count["correct"]}/{count["accepted"]}|{pct(rate["precision"])} {interval(result,"precision")}|'
                     f'{pct(rate["retention"])} {interval(result,"retention")}|{count["false_visible"]}/{count["negative"]}|'
                     f'{count["recovered"]}/{count["returned_points"]}|{count["strict_recovered"]}/{count["returned_points"]}|{result["empirical_gate_passed"]}|')
    table='\n'.join(table)
    per=['|视频|方法|精度|保留率|错误可见声明/负例|错误位置|运行|','|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        per.append(f'|{row["case"]}|{LABELS[row["arm"]]}|{pct(row["precision"])}|{pct(row["retention"])}|'
                   f'{row["false_visible"]}/{row["negative"]}|{row["wrong_location"]}|{row["inference_status"]}|')
    with (output/'per_video.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    for arm in LABELS:
        selected=[r for r in rows if r['arm']==arm and r['precision'] is not None]
        axes[0].scatter([r['retention']*100 for r in selected],[r['precision']*100 for r in selected],
                        label=LABELS[arm],color=COLORS[arm],s=35,alpha=.75)
    axes[0].axhline(95,color='0.5',linestyle='--',linewidth=.8)
    axes[0].axvline(20,color='0.5',linestyle='--',linewidth=.8)
    axes[0].set(xlim=(-2,102),ylim=(-2,102),xlabel='Correct retained / known-visible valid points (%)',
                ylabel='Correct / accepted known-label points (%)',title='Per-video precision and coverage\nZero-acceptance precision omitted (N/A)')
    axes[0].legend(frameon=False,fontsize=8,loc='lower right')
    for i,(arm,result) in enumerate(summary['arms'].items()):
        rate=result['pooled']['false_visible_rate']
        ci=result['participant_bootstrap']['false_visible_rate']['ci95']
        if rate is not None:
            axes[1].scatter(i,rate*100,color=COLORS[arm],s=65)
            if ci is not None:
                axes[1].plot([i,i],[ci[0]*100,ci[1]*100],color=COLORS[arm])
    axes[1].set(xticks=range(3),xticklabels=['Native','Score>0.995','Reciprocal'],ylim=(-2,102),
                ylabel='False visible / known occluded or OOV points (%)',
                title=f'Pooled false-visible rate\n95% participant-cluster bootstrap; {summary["participant_clusters"]} groups')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    for suffix in ['png','pdf','svg']:
        fig.savefig(output/f'measurement.{suffix}',dpi=180)
    plt.close(fig)
    native=summary['arms']['native']['counts']
    differences=[]
    for arm in ['confidence_0995','reciprocal']:
        count=summary['arms'][arm]['counts']
        differences.append(f'- {LABELS[arm]}相对原生：少接受错误{native["accepted"]-native["correct"]-(count["accepted"]-count["correct"])}个，'
                           f'同时丢失原本正确测量{native["correct"]-count["correct"]}个。坐标没有更新，不能称定位/身份恢复改善。')
    report=f'''# 外部真实域测量与误接受验证（2026-11-17）

## 方向与问题

优先优化评测器的真实域对应/可见性测量；裁判校准次之；评价链路有效前不扩大生成。
本轮问题：已冻结的CoTracker3及DAVIS置信门能否在新的人工标签上同时保持精度、覆盖，
加入原始参考帧往返核验能否挡住错误而保留有用测量？模型权重不训练、门槛不按成绩搜索。

## 设计与数据

原作者EgoPoints人工标注，{summary['planned_clips']}段、{summary['participant_clusters']}参与者组、
{sum(c['queries'] for c in manifest['cases'])}条查询、{sum(c['frames'] for c in manifest['cases'])}帧。
每个原EPIC视频只取一个片段；参与者内仍相关。来源前缀{prefix['bytes']:,}字节，
原包{prefix['whole_archive_bytes']:,}字节，不宣称取得完整数据集。
选择按运行前记录的PROTOCOL/V2/V3：原包至多前3GiB中的归档顺序前12个完整、64–1200帧、
不同原视频片段。数据可用性修订全部早于新推理/成绩，未按结果换片段。
归档顺序不是均匀随机抽样；这是外部小试，不是完整EgoPoints榜单或生成片域的验证。

作者只在少量帧人工标注。只有vis_valids有效的帧进入可见性计分；初始查询帧排除。
坐标正确采用已知可见且有有效坐标、归一化至256×256后L2<4像素。
未标注的默认0不当作遮挡；已知负例的错误接受保留在精度分母；推理失败保留计划分母。
原始RGB帧全部进入原生跟踪，512×288输入；查询外的真值不进入网络。
三种方法均允许知道请求测量的时间戳；往返核验只在这些时间戳调用，故不声称密集身份可测。

三臂为原生可见声明、旧DAVIS固定v*c>0.995门、以及原生声明加反向对应<4归一化像素的往返门。
反向只输入候选帧与原参考帧，从自身候选位置追回，另外要求反向原生可见和图内坐标。
网络版本/查询支持网格/迭代固定；未来真值另存于truth.npz并只由计分器读取。

## 实际结果

{table}

{chr(10).join(differences)}

原生各点坐标的可见位置正确数为{native['located_all_visible']}/{native['positive']}，
这是不依赖接受门的定位诊断；三个门使用相同坐标。重现事件仅指人工标注的OOV→随后可见
转换，不估计未标注中间帧的精确遮挡时长；严格恢复还要求先前OOV没有错误可见声明。
逐视频和宏均值、分母有效视频数、OOV/图内遮挡的误接受分解均保存在analysis JSON中。

![逐视频精度/保留率及误接受](measurement.png)

## 不确定性、验证和局限

置信区间对参与者整组bootstrap10000次，seed20261117；方法差值使用同一组抽样。
它只是此固定样本的描述性区间，不含训练不确定性、标注误差或跨生成域保证；空分母记N/A，
有效抽样次数逐项保存。没有把查询/帧当独立视频，也没有计算独立点帧显著性P值。
经验门定义为池化精度≥95%、正确保留≥20%、至少一半计划视频有≥5个正确接受点时，
即使经验门通过也不构成未知域的可靠性保证。

软件检查：`{tests['stdout'].strip().splitlines()[-1]}`，compile退出{tests['compile_exit']}。
反例包括未知帧、推理失败、不可核实接受、坐标缩放、以及“往返一致但身份错误”的假阳性。
不同家族本地GLM/Qwen实际审查及逐条裁定见REVIEW_RULINGS.md；其中无证据指控没有驱动修改，
不将模型意见包装成独立全面通过。LSP未安装，未宣称静态检查通过。

当前标签只支持点对应及可见性终点，不能直接证明物体质心、g/T或物理VLM裁判可靠。
这里的误接受是点定位/可见性错误，不是“误判物理违反”的假阳性率，不能替代裁判校准。
本轮没有训练生成模型，没有新增生成片、调到通过或修复旧Part B的失败门。
自身生成视频的人类物理标签仍缺。下一步应针对本轮暴露的坐标错误或拒绝损失选方法，
并使用新的验证集合；不能继续把此样本当未见验证反复优化。

## 文献借鉴、复现与声明

- [EgoPoints，Darkhalil等，WACV2025](https://arxiv.org/abs/2412.04592)：借鉴人工可见性/OOV分离、稀疏标注与重现终点；未复刻作者完整排名。
- [PointSt3R，Guerrier等](https://arxiv.org/abs/2510.26443)：参考跨帧对应思路；本轮没有运行它的MASt3R模型或训练。
- [BootsTAP](https://arxiv.org/abs/2402.00847)：参考查询一致性的研究动机；本轮不是该方法的自监督训练复现。
- 实验设计工作流参考Kassis、Agarwal、He、Patel、Brueckner，2026，
  [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)，v2元数据已核实，未视为方法有效性证据。

实际软件版本见tests_receipt.json；权重和作者源码继承已有assets.json固定哈希。
完整协议、样本、源图哈希、原始标签/预测、首次失败均保留；仅本地交接，原视频权利仍归原来源。
复现入口为prepare.py→run_models.py两分片→analyze.py→report.py；准备/推理默认不覆盖，
重跑请复制到新目录并调整清单。交接包实际执行复制源码的缓存计分重放；不宣称干净机器网络安装。

AI披露：GPT辅助实现/分析/写作，GLM与Qwen作独立上下文审查；真值来自作者人工标签，
没有把AI意见写作人工标注。没有公开上传、付费API或Google/gingersea操作。

## 逐视频结果

{chr(10).join(per)}
'''
    (output/'REPORT.zh.md').write_text(report)
    print(json.dumps({'report':str(output/'REPORT.zh.md'),'figure':str(output/'measurement.png')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'report')
    main(parser.parse_args().output)
