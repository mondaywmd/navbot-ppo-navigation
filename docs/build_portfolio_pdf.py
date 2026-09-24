import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.tools', 'reportlab')))
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

OUT=os.path.join(os.path.dirname(__file__),'2026-09-05_clean_start_robot_learning_portfolio_summary.pdf')
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
FONT='STSong-Light'
styles=getSampleStyleSheet()
title=ParagraphStyle('C_Title',fontName=FONT,fontSize=22,leading=30,textColor=colors.HexColor('#16324F'),alignment=TA_CENTER,spaceAfter=10*mm)
h1=ParagraphStyle('C_H1',fontName=FONT,fontSize=15,leading=22,textColor=colors.HexColor('#176B87'),spaceBefore=6*mm,spaceAfter=3*mm)
body=ParagraphStyle('C_Body',fontName=FONT,fontSize=10.2,leading=16,textColor=colors.HexColor('#20242A'),spaceAfter=2.2*mm)
small=ParagraphStyle('C_Small',fontName=FONT,fontSize=8.2,leading=12,textColor=colors.HexColor('#4A5560'))
bullet=ParagraphStyle('C_Bullet',parent=body,leftIndent=5*mm,firstLineIndent=-3*mm,bulletIndent=1*mm)

def P(text,style=body):return Paragraph(text,style)
def table(rows,widths=None):
    cooked=[[P(str(c),small) for c in row] for row in rows]
    t=Table(cooked,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCEEF5')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#16324F')),('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#AAB7C1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]));return t

story=[P('程序化机器人学习仿真与导航实验',title),P('阶段总结与求职作品集技术留档',ParagraphStyle('sub',parent=title,fontSize=14,leading=20,textColor=colors.HexColor('#4A5560'))),P('记录日期：2026-09-05　｜　ROS Noetic + Gazebo Classic　｜　Clean-start BC/PPO',ParagraphStyle('meta',parent=small,alignment=TA_CENTER)),Spacer(1,8*mm)]

sections=[
('1. 项目摘要',["本项目构建了差速移动机器人的多障碍导航学习闭环。机器人只读取时间序列 LiDAR、目标相对信息与动作历史；Actor 不接收柱子数量、坐标或半径。场景由固定 seed 可复现生成，并随机改变 1–5 根柱子的位置、半径、目标位置和初始朝向。","最终 clean-start 路线不继承旧模型，从随机初始化开始训练严格左右镜像等变的 Actor。匹配新鲜场景中，BC-only 达到 27/30 成功，PPO 达到 29/30 成功。PPO 消除了两个复杂四柱超时，但没有消除同一个五柱镜像场景的安全失败。"]),
('2. 个人贡献与问题解决',["关键贡献不是单纯运行训练，而是持续定义问题、观察异常并建立验证协议：识别单侧 BC 的方向偏差；要求镜像成对场景和结构级等变；引入前后帧 LiDAR；区分正前方和侧面障碍；禁止柱子与目标重叠；定位 gzclient 缓存和跨进程残留模型；通过回放识别目标擦边；冻结测试数据并拒绝覆盖结果。","实现过程大量使用 AI 辅助。求职时应如实称为 AI-assisted development，并能够独立解释系统架构、实验设计、关键代码和结果边界。"]),
('3. 系统架构',["Gazebo 场景 → 360° LiDAR → 36 维当前扇区距离 + 36 维变化率 → 78D 观测 → Actor → 线/角速度 → LiDAR 安全层 → cmd_vel","观测包括 36 维当前 LiDAR、36 维帧间有符号距离变化、2 维上一实际动作，以及 4 维目标几何信息。世界坐标只用于生成、合法性检查和统计，不进入 Actor。"]),
('4. 场景与安全协议',["每回合使用新 seed，柱数在线均匀随机 1–5；检查场地边界、起点/目标/柱间余量、黄色目标不重叠，并用栅格搜索验证存在可行路线。","期望净空通常为 0.40 m，接近目标时放宽至最低 0.25 m；0.20 m 是严重 LiDAR 代理违规阈值。回合上限 200 步，来源是专家最复杂成功约 149 步加约 50 步余量。"]),
]
for head,paras in sections:
    story.append(P(head,h1));story.extend(P(x) for x in paras)

story += [P('5. BC 数据与训练',h1),table([
['项目','结果'],['数据','30 对镜像 / 60 成功回合 / 5518 样本'],['覆盖','1–5 柱每种 6 对'],['Actor','78→256→256→2；严格镜像等变'],['验证','1000 样本最大镜像误差 0'],['BC','400 epochs；最佳验证 MSE 0.0147177'],['初始化','完全随机；不使用旧 BC/权重/OOD']], [48*mm,122*mm]),
P('6. BC-only 新鲜闭环验收',h1),table([['指标','结果'],['成功','17/20（85%）'],['碰撞','0/20'],['超时','3/20'],['平均步数（含失败）','110.65'],['最小 LiDAR','0.2111 m']],[70*mm,100*mm]),
P('回放发现两个镜像超时均从目标边缘擦过：最近距离约 0.475/0.477 m、目标位于约 ±90°，但线速度仍约 0.176–0.177 m/s，角速度仅约 ±0.05 rad/s，且安全层为 clear。由此排除单纯步数不足，定位为末端目标捕获不足。'),
P('7. Clean-start PPO',h1),P('PPO seed 9051006 从冻结 BC Actor 初始化；在线持续生成新 seed、随机 1–5 柱场景；保留 200 步、严格镜像 Actor 和 LiDAR 安全层。奖励针对目标侧方高速擦过和接近后远离。训练共 10010 环境步、12 次 PPO 更新。'),
table([['训练统计','结果'],['回合','98'],['成功','92（93.88%）'],['碰撞','1（1.02%）'],['超时','5（5.10%）'],['最终 Critic loss','8.8219']],[70*mm,100*mm]),
P('8. BC 与 PPO 同场景公平对比',h1),table([['模型','成功','碰撞','超时','成功中位步数'],['BC-only','27/30（90.00%）','1','2','93'],['PPO','29/30（96.67%）','1','0','91']],[42*mm,38*mm,25*mm,25*mm,40*mm]),
P('PPO 把两个四柱超时变为 160/161 步成功，且没有增加超时；共同可解场景上步数持平或略快。但是两者在同一个五柱镜像场景均触及 0.20 m 阈值，因此安全失败仍未解决。当前结果是新鲜非 OOD 评估，不是永久 OOD 认证。'),
P('9. 工程问题与修复',h1)]
for x in ['目标显示陈旧：目标只生成一次，reset 使用 set_model_state，并由 gzserver 验证。','跨进程柱子残留：reset 前枚举删除旧动态模型，并核对实时柱数。','左右偏差：由软惩罚升级为 Actor 前向结构的严格等变。','单帧信息不足：加入 LiDAR 帧间距离变化率。','侧面障碍错误降速：按未来扫掠通道判断是否真正阻挡前进。','超时误判：用逐步轨迹证明接近目标后远离，而非简单增加步数。']:
    story.append(P('• '+x,bullet))
story += [P('10. 求职定位',h1),P('适合 Robotics Simulation Engineer、Robot Learning Simulation/Evaluation Engineer、Synthetic Data Engineer、Digital Twin Engineer、Autonomous Systems Simulation Engineer，以及初级 Robotics Software/Research Engineer。核心差异化是 Houdini 程序化三维能力与机器人传感器、学习和评估闭环的结合。'),P('建议能力投入比例：约 70% 深化 Houdini/USD/Isaac Sim/程序化环境/合成数据/评估系统，约 30% 补 ROS2、Nav2、机器人基础与足够审核 C++ 的能力。'),
P('11. 局限与下一步',h1)]
for x in ['尚无真机或 sim-to-real 证据。','障碍主要是静态圆柱，尚未覆盖复杂形状和动态行人。','期望 0.40 m 净空没有被严格保持。','同一五柱镜像场景在 BC/PPO 中均出现安全失败。','10k PPO 规模较小；永久 OOD 尚未用于 clean-start 最终考试。','下一步应先复盘重复安全失败，再扩展 Houdini→USD→Isaac Sim 的程序化场景与批量评估管线。']:
    story.append(P('• '+x,bullet))
story += [P('12. 关键资料与 SHA-256',h1),table([
['资料','相对路径','SHA-256（前 16 位）'],['BC 数据','datasets/clean_bc_1to5_seed9051001/clean_bc.npz','7439162ed1ba05a3'],['BC Actor','runs/clean_bc_1to5_seed9051002/actor_bc_best.pth','96433e12828532f8'],['PPO Actor','runs/clean_ppo_10k_seed9051006/actor_latest.pth','afae1f29f130ca1cc'],['BC 公平评估','results/fair_eval_seed9051007_bc/summary.json','2db84daba862c5ef'],['PPO 公平评估','results/fair_eval_seed9051007_ppo/summary.json','9aa17fafaeaa7576']],[30*mm,100*mm,40*mm]),Spacer(1,8*mm),P('作品集一句话描述',h1),P('使用 Houdini 式程序化思维构建可复现的多难度机器人导航仿真与评估系统，以时间序列 LiDAR、严格镜像 Actor、行为克隆、PPO 和确定性安全层实现随机 1–5 障碍环境中的学习与故障诊断。')]

def footer(canvas,doc):
    canvas.saveState();canvas.setFont(FONT,8);canvas.setFillColor(colors.HexColor('#69757F'))
    canvas.drawString(20*mm,12*mm,'Clean-start Robot Learning Portfolio Record · 2026-09-05')
    canvas.drawRightString(190*mm,12*mm,'第 %d 页'%doc.page);canvas.restoreState()

doc=SimpleDocTemplate(OUT,pagesize=A4,rightMargin=18*mm,leftMargin=18*mm,topMargin=17*mm,bottomMargin=19*mm,title='程序化机器人学习仿真与导航实验：阶段总结',author='Project Portfolio')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUT)
