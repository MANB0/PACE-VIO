#!/usr/bin/env python3
"""Build the concise, plain-language weekly MACVO-VIO report."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    NextPageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.utils import ImageReader

from build_weekly_group_report_document_pdf import (
    BLUE,
    FONT,
    INK,
    LIGHT,
    NAVY,
    ORANGE,
    OUT,
    PALE_BLUE,
    PALE_ORANGE,
    PALE_RED,
    PALE_TEAL,
    RED,
    REPORT_ROOT,
    ROOT,
    TEAL,
    ReportDocTemplate,
    bullet,
    callout,
    figure,
    p,
    register_fonts,
    styled_table,
    styles,
)


FIG = REPORT_ROOT / "figures"


def two_column_callouts(left, right):
    table = Table([[left, right]], colWidths=[80 * mm, 80 * mm], hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def image_flowable(path: Path, width: float, max_height: float) -> Image:
    reader = ImageReader(str(path))
    image_w, image_h = reader.getSize()
    scale = min(width / image_w, max_height / image_h)
    image = Image(str(path), width=image_w * scale, height=image_h * scale)
    image.hAlign = "CENTER"
    return image


def build_story(st):
    story = []

    # Page 1: cover and summary
    story.append(Spacer(1, 18 * mm))
    story.append(Paragraph("WEEKLY REVIEW · 2026.07.13--2026.07.18", st["CoverKicker"]))
    story.append(Paragraph("MACVO 与 IMU 融合", st["CoverTitle"]))
    story.append(Paragraph("这一周做了什么，结果怎么样，下一步怎么走", st["CoverSubtitle"]))
    story.append(Paragraph("T-pose · U1 · SA-v1 · SA-v2", st["CoverKicker"]))
    story.append(Spacer(1, 53 * mm))
    story.append(
        callout(
            "一句话结论",
            "IMU 的基本计算链已经检查通过。现在影响轨迹的主要问题不是某个公式写错，而是视觉信息怎样进入后端、加速度计 Bias 是否过度变化，以及两帧窗口怎样传递历史信息。",
            st,
            fill=PALE_TEAL,
            accent=TEAL,
        )
    )
    story.append(Spacer(1, 5 * mm))
    for text in [
        "U1 直接使用点级 UVD，比把视觉先压缩成一个 T_ij 更合适。",
        "Sampling-aware covariance 是正确的，但它不是平滑器，不能单独消除抖动。",
        "当前高频抖动最明显的来源是加速度计 Bias 在两帧窗口内变化过快。",
        "SA-v2 的晚段跳变已经定位到低秩 covariance 与历史 prior 的组合问题。",
    ]:
        story.append(bullet(text, st))
    story.append(NextPageTemplate("Normal"))
    story.append(PageBreak())

    # Page 2: system and formulas
    story.append(p("1. 当前系统到底在做什么", st, "Heading1"))
    story.append(
        p(
            "每来一对相机帧，MACVO 提供两帧之间的视觉运动信息，IMU 提供这段时间内的旋转、速度和位置增量。后端同时调整前后两帧的姿态、位置、速度和两类 Bias，并把更早历史压缩成一个 prior。当前实现是两帧 fixed-lag 优化，不是 EKF。",
            st,
        )
    )
    method_data = [
        ["方法", "直白解释", "主要特点"],
        ["T-pose", "MACVO 先把所有点压成一个相对位姿，再和 IMU 融合", "快，但会丢点级信息"],
        ["U1", "不先压成位姿，直接把每个 UVD 点送进融合优化", "信息更多，计算更慢"],
        ["SA-v1", "在 U1 上补上单段 IMU 插值带来的 covariance", "统计更正确，不等于更平滑"],
        ["SA-v2", "继续考虑相邻两段 IMU 共用边界采样", "信息最完整，prior 处理更难"],
    ]
    story.append(styled_table(method_data, [26 * mm, 90 * mm, 49 * mm], font_size=8.4))
    story.append(Spacer(1, 5 * mm))
    story.append(image_flowable(FIG / "core_equations.png", 165 * mm, 74 * mm))
    story.append(Paragraph("图 1  本报告只保留的核心状态和残差公式。公式由数学渲染器生成。", st["Caption"]))
    story.append(
        callout(
            "怎样理解这些公式",
            "它们都在比较“状态预测出的运动”和“传感器测到的运动”。残差越小，说明两者越一致；covariance 决定优化器应该相信哪一边更多。",
            st,
            fill=PALE_BLUE,
            accent=BLUE,
        )
    )
    story.append(PageBreak())

    # Page 3: weekly investigation
    story.append(p("2. 这一周先排除了哪些基础错误", st, "Heading1"))
    timeline = [
        ["阶段", "检查内容", "结果"],
        ["Jacobian", "优化器算出的导数是否正确", "修复 PyPose translation() 问题后通过"],
        ["IMU 预积分", "是否错误地使用外部姿态提前减重力", "改成标准 body-frame local preintegration"],
        ["GTSAM 对照", "两边 Delta 和 covariance 是否一致", "数值几乎一致，非 MACVIO 单方错误"],
        ["视觉接入", "T-pose 压缩是否损失信息", "U1 全圆结果明显更好"],
        ["Noise 抖动", "covariance、Bias、窗口谁是主因", "Bias 自由度是当前第一主因"],
        ["SA-v2 跳变", "是否为单条坏测量", "不是；历史 prior 在重写旧状态"],
    ]
    story.append(styled_table(timeline, [34 * mm, 72 * mm, 59 * mm], font_size=8.4))
    story.append(Spacer(1, 6 * mm))
    story.append(
        two_column_callouts(
            callout(
                "导数检查",
                "随机状态下，自动求导与中心有限差分的最大绝对误差为 5.16e-8；300 帧中的 209 条有效边全部收敛。",
                st,
                fill=PALE_BLUE,
                accent=BLUE,
            ),
            callout(
                "标准预积分",
                "使用 GT 状态时，速度残差中位数从 1.63e-3 m/s 降到 2.52e-6 m/s，已经接近重新积分参考值。",
                st,
                fill=PALE_TEAL,
                accent=TEAL,
            ),
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        callout(
            "与 GTSAM 对比说明什么",
            "两边的预积分增量和 covariance 基本一致，NIS 也表现为相似的保守程度。因此，当前轨迹抖动不能简单归因于 MACVIO 的预积分矩阵写错。",
            st,
            fill=PALE_ORANGE,
            accent=ORANGE,
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        p(
            "到这一步，可以把问题分成两层：底层数学计算已经基本可信；上层仍要解决视觉信息压缩、Bias 可观性和历史 prior 的误差分配。",
            st,
        )
    )
    story.append(PageBreak())

    # Page 4: T-pose vs U1
    story.append(p("3. 为什么从 T-pose 转向 U1", st, "Heading1"))
    story.append(
        p(
            "T-pose 的流程是：MACVO 先用所有匹配点单独优化出一个 T_ij，然后后端只看到这个 6 维结果。U1 的流程是：MACVO 的 T_ij 只做初值，后端继续直接看所有 UVD 点。二者使用相同视觉数据，但 U1 少了一次信息压缩。",
            st,
        )
    )
    compare_data = [
        ["圆形完整序列", "Pure MACVO", "T-pose", "U1", "U1 相对 T-pose"],
        ["XY ATE RMSE / m", "2.5709", "3.2380", "2.3041", "改善 28.8%"],
        ["Translation RPE / m", "0.01183", "0.02500", "0.01592", "改善 36.3%"],
        ["Rotation RPE / rad", "0.000194", "0.000385", "0.000271", "改善 29.7%"],
    ]
    story.append(styled_table(compare_data, [49 * mm, 28 * mm, 26 * mm, 26 * mm, 36 * mm], font_size=8.4))
    story.append(Spacer(1, 6 * mm))
    story.append(
        callout(
            "能得出的结论",
            "把点级 UVD 直接送入联合优化，确实比先压成 6D T_ij 更好。Pose factor 的信息压缩和它的 covariance 是原方法精度下降的重要来源。",
            st,
            fill=PALE_TEAL,
            accent=TEAL,
        )
    )
    story.append(Spacer(1, 5 * mm))
    story.append(
        callout(
            "还不能得出的结论",
            "U1 仍没有解决圆心偏移、半径误差和局部抖动；它的计算量也很高，圆形 1890 帧约运行 58 分钟。因此 U1 更适合作为精度基线，而不是当前实时方案。",
            st,
            fill=PALE_ORANGE,
            accent=ORANGE,
        )
    )
    story.append(Spacer(1, 7 * mm))
    for text in [
        "T-pose：低计算量基线。",
        "U1：保留点级视觉信息的精度基线。",
        "后续实验不应删除其中任何一个，否则难以判断改动到底改善了什么。",
    ]:
        story.append(bullet(text, st))
    story.append(PageBreak())

    # Page 5: noise and bias
    story.append(p("4. 加入 normal noise 后为什么会抖", st, "Heading1"))
    story.extend(figure(FIG / "bias_window_ablation.png", "图 2  Bias 消融与窗口长度/计算时间对比。", st, width=165 * mm, max_height=88 * mm))
    story.append(
        p(
            "Sampling-aware covariance 已通过 Monte Carlo 检查，但它通常比旧 covariance 更小，也就是更相信 IMU。在只有两帧的窗口中，这会更容易把白噪声分配给速度和 Bias，因此轨迹不一定更平滑。",
            st,
        )
    )
    bias_data = [
        ["实验", "XY 高频误差 / m", "说明"],
        ["正常优化 ba/bg", "0.012535", "当前基线"],
        ["固定 ba，只优化 bg", "0.005462", "高频下降 56.4%"],
        ["窗口 N=2 -> N=10", "0.012534 -> 0.009895", "有帮助，但计算从 0.37 s 增到 4.32 s/次"],
    ]
    story.append(styled_table(bias_data, [54 * mm, 42 * mm, 69 * mm], font_size=8.3))
    story.append(Spacer(1, 5 * mm))
    story.append(
        callout(
            "当前最可能的主因",
            "加速度计 Bias ba 在每个两帧窗口里都可以独立变化，它会吸收本应属于白噪声、速度或视觉误差的部分。正常运行中，ba 的变化约为理论 random walk 尺度的 85.7 倍。",
            st,
            fill=PALE_RED,
            accent=RED,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(p("因此下一步应先限制 ba 的更新频率或让一个短窗口共享同一个 ba，而不是继续手工调大/调小 IMU 权重。", st))
    story.append(PageBreak())

    # Page 6: metrics overview
    story.append(p("5. 三个 normal-noise 场景的总体结果", st, "Heading1"))
    story.extend(figure(FIG / "cross_scene_metrics.png", "图 3  圆形、停转矩形和直线场景的 ATE 与 RPE。", st, width=165 * mm, max_height=92 * mm))
    story.append(
        p(
            "三个场景的运动方式不同，所以不能只用一个平均分选“冠军”。圆形持续转弯，更容易暴露航向和长期 prior 问题；矩形包含停车和急转；直线主要暴露横向漂移。",
            st,
        )
    )
    summary_data = [
        ["场景", "最低 XY ATE", "最低局部 RPE", "直观结论"],
        ["圆形", "SA-v2: 2.1666 m", "Pure MACVO", "SA-v2 有晚段跳变，低 ATE 不能代表稳定"],
        ["矩形", "T-pose: 0.6645 m", "Pure MACVO / U1", "U1 在全局和局部之间更均衡"],
        ["直线", "T-pose: 0.3072 m", "Pure MACVO", "T-pose 低频摆动，U1/SA 基本重合"],
    ]
    story.append(styled_table(summary_data, [24 * mm, 41 * mm, 43 * mm, 57 * mm], font_size=8.2))
    story.append(Spacer(1, 6 * mm))
    story.append(
        callout(
            "最重要的观察",
            "当前不存在所有场景都最好的一种方法。评估时必须同时看 ATE、RPE、高频误差和最大单步，不能只看一条轨迹是否更接近 GT。",
            st,
            fill=PALE_BLUE,
            accent=BLUE,
        )
    )
    story.append(PageBreak())

    # Page 7: circle
    story.append(p("6. 圆形轨迹", st, "Heading1"))
    story.extend(figure(FIG / "trajectory_circle.png", "图 4  圆形 normal-noise：GT、Pure MACVO、T-pose、U1、SA-v1、SA-v2。箭头表示机头方向。", st, width=165 * mm, max_height=188 * mm))
    for text in [
        "U1、SA-v1 和 SA-v2 的大部分轨迹比较接近，整体优于 T-pose。",
        "所有方法仍有圆心、半径和闭合误差，问题不只是一个固定初始 yaw。",
        "SA-v2 在末段出现明显离群点，因此它虽然 ATE 最低，却不能判为最稳定。",
    ]:
        story.append(bullet(text, st))
    story.append(PageBreak())

    # Page 8: rectangle and straight
    story.append(p("7. 停转矩形与直线轨迹", st, "Heading1"))
    story.append(image_flowable(FIG / "trajectory_rectangle.png", 165 * mm, 112 * mm))
    story.append(Paragraph("图 5  停转矩形 normal-noise 轨迹。", st["Caption"]))
    story.append(
        p(
            "矩形中 T-pose 的 XY ATE 最低，但转角附近存在明显环状扰动；U1/SA-v1 的位置接近 T-pose，局部 RPE 更好，表现更均衡。",
            st,
        )
    )
    story.append(image_flowable(FIG / "trajectory_straight.png", 165 * mm, 54 * mm))
    story.append(Paragraph("图 6  直线 normal-noise 轨迹。", st["Caption"]))
    story.append(
        p(
            "直线中 Pure MACVO 的局部 RPE 最好，但横向误差持续积累；T-pose 最终位置更接近 GT，却有低频摆动；U1、SA-v1 和 SA-v2 基本重合。",
            st,
        )
    )
    story.append(PageBreak())

    # Page 9: output-only filtering on rectangle and straight scenes
    story.append(p("8. 矩形与直线的输出滤波复验", st, "Heading1"))
    story.extend(
        figure(
            ROOT
            / "analysis_rectangle_straight_output_eskf3d_ablation_20260718"
            / "rectangle_straight_mode_e_xy.png",
            "图 7  矩形与直线完整序列的 E 模式轨迹；完整 A-E 数值见下表。",
            st,
            width=165 * mm,
            max_height=91 * mm,
        )
    )
    story.append(
        p(
            "该实验不重新运行 VIO，而是对已经完成的 T-pose、U1、SA-v1 和 SA-v2 轨迹执行因果输出滤波。A 为原始轨迹，B 为二维 XY/yaw EKF，C 为无门控三维 ESKF，D 加入分块门控，E 再加入自适应过程噪声。矩形使用完整 1890 帧，直线使用完整 630 帧；所有方法与 GT 的时间戳逐帧相等。",
            st,
        )
    )
    filter_data = [
        ["场景", "方法", "XY ATE: A -> E", "二阶差分下降", "E 耗时/帧"],
        ["矩形", "T-pose", "0.6645 -> 0.6687 m", "81.5%", "243 us"],
        ["矩形", "U1", "0.6893 -> 0.6920 m", "80.7%", "275 us"],
        ["矩形", "SA-v1", "0.6934 -> 0.6957 m", "80.4%", "261 us"],
        ["矩形", "SA-v2", "0.7756 -> 0.7796 m", "82.0%", "251 us"],
        ["直线", "T-pose", "0.3072 -> 0.3072 m", "88.7%", "225 us"],
        ["直线", "U1", "0.4151 -> 0.4151 m", "87.9%", "245 us"],
        ["直线", "SA-v1", "0.4119 -> 0.4119 m", "87.9%", "242 us"],
        ["直线", "SA-v2", "0.4173 -> 0.4173 m", "88.7%", "224 us"],
    ]
    story.append(
        styled_table(
            filter_data,
            [22 * mm, 27 * mm, 47 * mm, 38 * mm, 31 * mm],
            font_size=7.7,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        callout(
            "这一页能证明什么",
            "三维 ESKF 能以约 0.22--0.28 ms/帧的纯滤波开销显著压低逐帧折线：矩形约 80%--82%，直线约 88%。但 XY ATE 基本不改善，矩形甚至略微上升。因此输出滤波适合实时显示层抑制高频抖动，不能替代对视觉低频偏差、Bias 状态设计和窗口误差分配的修正。C 往往最平滑，D/E 用一部分平滑度换取异常门控。",
            st,
            fill=PALE_BLUE,
            accent=BLUE,
        )
    )
    story.append(PageBreak())

    # Page 10: SA-v2 jump and next steps
    story.append(p("9. SA-v2 跳变与下一步", st, "Heading1"))
    story.append(image_flowable(ROOT / "analysis_circle_sa_v2_outlier_6167_20260717" / "sa_v2_late_outlier_diagnostic.png", 165 * mm, 82 * mm))
    story.append(Paragraph("图 8  SA-v2 末段保存轨迹单步、edge 内运动与下一 edge 重写。", st["Caption"]))
    story.append(
        p(
            "在 61.67 s 附近，单条 edge 自己估计的相对平移只有约 0.0016 m，但保存轨迹出现约 0.47 m 的单步。这说明坏点不是当前 measurement 直接造成，而是下一条 edge 重新修改了上一帧。",
            st,
        )
    )
    story.append(
        callout(
            "用直白的话解释根因",
            "相邻两段 IMU 共用了边界采样。SA-v2 把这部分相关性拆出来后，剩余 covariance 只有 6 个有效方向。旧代码却强行把另外 3 个方向也变得非常“可信”；历史 prior 长期累积这种过强约束，最后在重写旧状态时形成大跳变。",
            st,
            fill=PALE_RED,
            accent=RED,
        )
    )
    story.append(Spacer(1, 5 * mm))
    next_data = [
        ["顺序", "下一步", "为什么"],
        ["1", "保留 T-pose 与 U1 两个基线", "一个代表速度，一个代表点级信息"],
        ["2", "让短窗口共享 ba，或降低 ba 更新频率", "先处理当前最明显的抖动来源"],
        ["3", "正确处理 SA-v2 的低秩 covariance", "消除跳变，同时尽量保留相关信息"],
        ["4", "再比较 N=2/5/10 和 ESKF", "在模型正确后讨论实时性取舍"],
    ]
    story.append(styled_table(next_data, [18 * mm, 70 * mm, 77 * mm], font_size=8.3))
    story.append(Spacer(1, 5 * mm))
    story.append(
        callout(
            "最终结论",
            "这一周已经把底层计算错误基本排除，也找到了 U1 的视觉收益、Bias 抖动主因和 SA-v2 跳变原因。下一阶段应先改 ba 的状态设计，再完善相关 covariance，而不是继续调权重。",
            st,
            fill=PALE_TEAL,
            accent=TEAL,
        )
    )
    return story


def main() -> None:
    register_fonts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    st = styles()
    doc = ReportDocTemplate(
        str(OUT),
        pagesize=A4,
        title="MACVO 与 IMU 融合：一周简明报告",
        author="MACVO-VIO project",
    )
    doc.multiBuild(build_story(st))
    print(OUT)


if __name__ == "__main__":
    main()
