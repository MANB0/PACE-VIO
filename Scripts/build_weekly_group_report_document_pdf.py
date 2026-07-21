#!/usr/bin/env python3
"""Build the portrait, chapter-based MACVO-VIO weekly technical report."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "analysis_weekly_group_report_20260718"
FIG = REPORT_ROOT / "figures"
IMU_CENTER_ROOT = ROOT / "analysis_imu_center_all_methods_20260719"
OUT = ROOT / "output" / "pdf" / "macvo_vio_weekly_group_meeting_report_20260719.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")

FONT = "SimHei"
NAVY = HexColor("#0F172A")
INK = HexColor("#1E293B")
MUTED = HexColor("#64748B")
BLUE = HexColor("#2563EB")
TEAL = HexColor("#059669")
ORANGE = HexColor("#F97316")
RED = HexColor("#DC2626")
PURPLE = HexColor("#7C3AED")
LIGHT = HexColor("#F8FAFC")
LINE = HexColor("#CBD5E1")
PALE_BLUE = HexColor("#EFF6FF")
PALE_TEAL = HexColor("#ECFDF5")
PALE_ORANGE = HexColor("#FFF7ED")
PALE_RED = HexColor("#FEF2F2")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(FONT, str(FONT_PATH)))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT, italic=FONT, boldItalic=FONT)


def cover_page(c, doc) -> None:
    width, height = A4
    c.saveState()
    c.setFillColor(NAVY)
    c.rect(0, height - 97 * mm, width, 97 * mm, stroke=0, fill=1)
    c.setFillColor(BLUE)
    c.rect(0, 0, 7 * mm, height, stroke=0, fill=1)
    c.setStrokeColor(HexColor("#E2E8F0"))
    c.line(22 * mm, 28 * mm, width - 20 * mm, 28 * mm)
    c.setFont(FONT, 8)
    c.setFillColor(MUTED)
    c.drawString(22 * mm, 20 * mm, "MACVO-VIO weekly technical report")
    c.drawRightString(width - 20 * mm, 20 * mm, "2026-07-19")
    c.restoreState()


def normal_page(c, doc) -> None:
    width, height = A4
    c.saveState()
    c.setStrokeColor(LINE)
    c.line(20 * mm, height - 18 * mm, width - 20 * mm, height - 18 * mm)
    c.setFont(FONT, 8)
    c.setFillColor(MUTED)
    c.drawString(20 * mm, height - 13 * mm, "MACVO-VIO 方法决策：EKF、优化器选择与自适应收益")
    c.line(20 * mm, 16 * mm, width - 20 * mm, 16 * mm)
    c.drawString(20 * mm, 10 * mm, "NWU-XY · 无轨迹对齐 · 无 SE(3) 拟合 · 无尺度修正")
    c.drawRightString(width - 20 * mm, 10 * mm, str(doc.page - 1))
    c.restoreState()


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        cover_frame = Frame(
            22 * mm,
            30 * mm,
            A4[0] - 42 * mm,
            A4[1] - 50 * mm,
            id="cover-frame",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        body_frame = Frame(
            20 * mm,
            20 * mm,
            A4[0] - 40 * mm,
            A4[1] - 42 * mm,
            id="body-frame",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="Cover", frames=[cover_frame], onPage=cover_page),
                PageTemplate(id="Normal", frames=[body_frame], onPage=normal_page),
            ]
        )

    def afterFlowable(self, flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in {"Heading1", "Heading2"}:
            return
        level = 0 if style_name == "Heading1" else 1
        text = flowable.getPlainText()
        key = f"heading-{level}-{self.seq.nextf('heading')}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page - 1, key))


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    result: dict[str, ParagraphStyle] = {}
    result["CoverKicker"] = ParagraphStyle(
        "CoverKicker",
        fontName=FONT,
        fontSize=10,
        leading=14,
        textColor=HexColor("#BFDBFE"),
        spaceAfter=13,
    )
    result["CoverTitle"] = ParagraphStyle(
        "CoverTitle",
        fontName=FONT,
        fontSize=27,
        leading=37,
        textColor=colors.white,
        spaceAfter=10,
    )
    result["CoverSubtitle"] = ParagraphStyle(
        "CoverSubtitle",
        fontName=FONT,
        fontSize=15,
        leading=23,
        textColor=HexColor("#E2E8F0"),
        spaceAfter=10,
    )
    result["CoverMeta"] = ParagraphStyle(
        "CoverMeta",
        fontName=FONT,
        fontSize=9.5,
        leading=15,
        textColor=MUTED,
    )
    result["Title"] = ParagraphStyle(
        "Title",
        parent=base["Title"],
        fontName=FONT,
        fontSize=22,
        leading=30,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=10,
    )
    result["Heading1"] = ParagraphStyle(
        "Heading1",
        fontName=FONT,
        fontSize=17,
        leading=24,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=10,
        keepWithNext=True,
        borderColor=BLUE,
        borderWidth=0,
        borderPadding=(0, 0, 5, 0),
    )
    result["Heading2"] = ParagraphStyle(
        "Heading2",
        fontName=FONT,
        fontSize=13,
        leading=19,
        textColor=BLUE,
        spaceBefore=9,
        spaceAfter=6,
        keepWithNext=True,
    )
    result["Heading3"] = ParagraphStyle(
        "Heading3",
        fontName=FONT,
        fontSize=11.5,
        leading=17,
        textColor=TEAL,
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True,
    )
    result["Body"] = ParagraphStyle(
        "Body",
        fontName=FONT,
        fontSize=10.2,
        leading=17,
        textColor=INK,
        alignment=TA_JUSTIFY,
        firstLineIndent=20,
        spaceAfter=7,
    )
    result["BodyNoIndent"] = ParagraphStyle(
        "BodyNoIndent",
        parent=result["Body"],
        firstLineIndent=0,
    )
    result["Bullet"] = ParagraphStyle(
        "Bullet",
        fontName=FONT,
        fontSize=10,
        leading=16,
        textColor=INK,
        leftIndent=16,
        firstLineIndent=-9,
        bulletIndent=2,
        spaceAfter=4,
    )
    result["Equation"] = ParagraphStyle(
        "Equation",
        fontName=FONT,
        fontSize=10.2,
        leading=16,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=4,
    )
    result["Caption"] = ParagraphStyle(
        "Caption",
        fontName=FONT,
        fontSize=8.7,
        leading=13,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=10,
    )
    result["Small"] = ParagraphStyle(
        "Small",
        fontName=FONT,
        fontSize=8.5,
        leading=13,
        textColor=MUTED,
    )
    result["CalloutTitle"] = ParagraphStyle(
        "CalloutTitle",
        fontName=FONT,
        fontSize=11,
        leading=16,
        textColor=NAVY,
        spaceAfter=4,
    )
    result["CalloutBody"] = ParagraphStyle(
        "CalloutBody",
        fontName=FONT,
        fontSize=9.5,
        leading=15,
        textColor=INK,
    )
    result["TOCTitle"] = ParagraphStyle(
        "TOCTitle",
        fontName=FONT,
        fontSize=22,
        leading=30,
        textColor=NAVY,
        spaceAfter=14,
    )
    return result


def p(text: str, st: dict[str, ParagraphStyle], style: str = "Body") -> Paragraph:
    return Paragraph(text, st[style])


def bullet(text: str, st: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, st["Bullet"], bulletText="•")


def callout(title: str, body: str, st, fill=PALE_BLUE, accent=BLUE):
    content = [
        Paragraph(title, st["CalloutTitle"]),
        Paragraph(body, st["CalloutBody"]),
    ]
    table = Table([[content]], colWidths=[165 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("LINEBEFORE", (0, 0), (0, 0), 4, accent),
                ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#E2E8F0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def equation(text: str, st):
    table = Table([[Paragraph(escape(text), st["Equation"])]], colWidths=[155 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def figure(path: Path, caption: str, st, width=165 * mm, max_height=205 * mm):
    reader = ImageReader(str(path))
    image_w, image_h = reader.getSize()
    scale = min(width / image_w, max_height / image_h)
    img = Image(str(path), width=image_w * scale, height=image_h * scale)
    img.hAlign = "CENTER"
    return [img, Paragraph(caption, st["Caption"])]


def styled_table(data, col_widths, font_size=8.2, repeat_rows=1, aligns=None):
    table = Table(data, colWidths=col_widths, repeatRows=repeat_rows, hAlign="CENTER")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size * 1.45),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row in range(1, len(data)):
        if row % 2 == 0:
            style.append(("BACKGROUND", (0, row), (-1, row), LIGHT))
    if aligns:
        for col, align in enumerate(aligns):
            style.append(("ALIGN", (col, 1), (col, -1), align))
    table.setStyle(TableStyle(style))
    return table


def add_scene_table(story, st, scene: str, rows: list[list[str]]) -> None:
    story.append(p(scene, st, "Heading3"))
    data = [["方法", "XY ATE RMSE / m", "Translation RPE / m", "Rotation RPE / rad"]] + rows
    story.append(
        styled_table(
            data,
            [48 * mm, 39 * mm, 39 * mm, 39 * mm],
            font_size=8.4,
            aligns=["LEFT", "RIGHT", "RIGHT", "RIGHT"],
        )
    )
    story.append(Spacer(1, 4 * mm))


def build_story(st):
    story = []

    # Cover
    story.append(Spacer(1, 18 * mm))
    story.append(Paragraph("WEEKLY TECHNICAL REVIEW · 2026.07.13--2026.07.19", st["CoverKicker"]))
    story.append(Paragraph("MACVO-VIO 两状态融合", st["CoverTitle"]))
    story.append(Paragraph("从数学审计到 Normal-noise 三场景验证", st["CoverSubtitle"]))
    story.append(Paragraph("T-pose factor · Direct UVD U1 · SA-v1 · SA-v2", st["CoverKicker"]))
    story.append(Spacer(1, 56 * mm))
    story.append(
        callout(
            "报告范围",
            "本报告汇总 2026-07-13 至 2026-07-19 的方法设计、代码审计、消融实验和完整轨迹结果。最新评估已把 GT、Pure MACVO 和全部融合结果统一换算到 IMU 中心；轨迹不做后验 SE(3)、yaw 或尺度拟合。",
            st,
            fill=PALE_BLUE,
            accent=BLUE,
        )
    )
    story.append(Spacer(1, 8 * mm))
    story.append(
        callout(
            "核心判断",
            "基础 IMU 数学链和 Sampling-aware covariance 已由数值测试验证；历史评估中的相机/IMU参考点混用也已纠正。最新 Schur 分段实验不支持按帧硬切换平移/旋转视觉模式，当前主要矛盾仍是点级视觉、N=2 状态耦合与相关 prior 的误差分配。",
            st,
            fill=PALE_TEAL,
            accent=TEAL,
        )
    )
    story.append(NextPageTemplate("Normal"))
    story.append(PageBreak())

    # Abstract
    story.append(Paragraph("摘要", st["Title"]))
    story.append(
        p(
            "本周工作围绕一个核心原则展开：先证明实现的数学正确性，再讨论轨迹精度。我们从两状态 fixed-lag VIO 基线出发，依次修复 IMU pose-rotation Jacobian、重构为标准局部坐标系预积分、完成 MACVIO/GTSAM covariance-NIS 对比，并将视觉接入由 6D 相对位姿因子扩展为点级 Direct UVD U1。随后进一步实现 Sampling-aware covariance（SA-v1）和跨 edge 相关噪声（SA-v2），通过 Bias、窗口和 prior 消融定位 normal-noise 高频抖动与晚段离群点。",
            st,
        )
    )
    story.append(
        p(
            "结果表明：基础 IMU 数学链已通过数值审计；U1 在圆形全序列上显著优于 T-pose 压缩；Sampling-aware covariance 在 Monte Carlo 中统计正确，却不会自动平滑 N=2 轨迹；SA-v2 的 61.67 s 跳变来自低秩独立 covariance 的错误满秩化、连续 Schur prior 放大和旧状态回写的共同作用。参考点审计确认后端内部状态本来就是 IMU 中心，但历史 poses.csv 与 GT 的评估曾混用相机、body/root 和 IMU 参考点。统一换算 63 条历史估计后，方法排序的主结论没有翻转。最后，矩形分段 Schur 反事实实验显示 UVD 平移/旋转信息高度耦合，translation-only 和 rotation-only 都没有相对完整 UVD 获得实质、联合收益。",
            st,
        )
    )
    story.append(
        callout(
            "一句话结论",
            "当前问题已经不是“IMU 公式是否写错”或“轨迹是否在同一物理点”，也没有证据支持用硬平移/旋转模式切换解决；下一步应保持完整 UVD，直接验证短窗口长度的影响。",
            st,
            fill=PALE_ORANGE,
            accent=ORANGE,
        )
    )
    story.append(Spacer(1, 6 * mm))
    story.append(p("关键词：MACVO；视觉惯性里程计；IMU 预积分；Fixed-lag smoothing；UVD factor；Sampling-aware covariance；Bias random walk；Schur complement。", st, "BodyNoIndent"))

    story.append(PageBreak())
    story.append(Paragraph("目录", st["TOCTitle"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", fontName=FONT, fontSize=9.8, leading=13.2, leftIndent=0, textColor=NAVY, spaceBefore=1),
        ParagraphStyle("TOC2", fontName=FONT, fontSize=8.3, leading=11.2, leftIndent=12, textColor=MUTED, spaceBefore=0),
    ]
    story.append(toc)
    story.append(PageBreak())

    # Chapter 1
    story.append(p("1. 研究背景与本周问题", st, "Heading1"))
    story.append(
        p(
            "项目目标是在 MACVO 双目视觉前端上融合 IMU，使系统在 normal-noise 条件下获得更稳定的平移、旋转、速度与 Bias 估计。由于双目已经提供公制尺度，本研究的主要困难不是尺度初始化，而是如何正确使用点级视觉不确定性、IMU 预积分 covariance、Bias random walk 和固定滞后 prior。",
            st,
        )
    )
    story.append(p("1.1 本周需要回答的九个问题", st, "Heading2"))
    for item in [
        "两状态 IMU 因子在当前流形扰动约定下是否拥有正确 Jacobian？",
        "预积分是否错误地使用外部姿态进行重力补偿或世界系积分？",
        "MACVIO 与 GTSAM 的预积分均值和 covariance 是否一致？",
        "MACVO 视觉信息应以 6D 相对位姿还是点级 UVD 残差进入后端？",
        "相机/IMU 边界插值造成的采样相关性是否被正确传播？",
        "Normal-noise 高频抖动主要来自 covariance、Bias 自由度还是短窗口？",
        "SA-v2 晚段离群点是单条视觉/IMU 测量异常，还是 prior 结构问题？",
        "不同场景下，T-pose、U1、SA-v1 和 SA-v2 是否存在统一最优方法？",
        "完整 UVD 是否应按运动段硬切换为 translation-only 或 rotation-only？",
    ]:
        story.append(bullet(item, st))
    story.append(p("1.2 评价原则", st, "Heading2"))
    story.append(
        p(
            "所有报告同时给出 XY ATE、translation/rotation RPE、局部高频误差、最大单步和数值稳定性。求解器收敛不等于模型正确；ATE 较低不等于局部运动稳定；NIS 总量正常也不等于各轴和各分块均正确。",
            st,
        )
    )

    # Chapter 2
    story.append(p("2. 数据契约与系统架构", st, "Heading1"))
    story.append(p("2.1 HoloOcean 数据契约", st, "Heading2"))
    for item in [
        "imu_data.csv：IMUSocket FLU -> world FLU；角速度和线加速度均为 body FLU；线加速度包含重力，静止期望约为 [0,0,+9.8]。",
        "ref_pose.csv：位置为 world NWU；相机姿态为 body NWU -> world NWU；速度为 world NWU。",
        "相机、IMU 和 GT 时间戳来自同一 tick，当前导出数据应视为同步，offset=0。",
        "Normal-noise 在传感器端真实加入白噪声与 Bias random walk；metadata 中生成参数的单位契约为 per-sample standard deviation。",
    ]:
        story.append(bullet(item, st))
    story.append(p("2.2 两状态后端", st, "Heading2"))
    story.append(p("每个图像时刻的惯性状态定义为：", st))
    story.append(equation("x_k = {T_WB,k, v_k^W, b_a,k, b_g,k}", st))
    story.append(
        p(
            "活动图同时包含前后两个状态 x_i 与 x_j。历史信息通过作用在第一个状态上的 Schur prior 进入，而不是把上一帧固定为绝对真值。当前系统是窗口大小 N=2 的非线性 fixed-lag MAP 优化器，不是 EKF。",
            st,
        )
    )
    story.append(p("2.3 标准 IMU 因子", st, "Heading2"))
    story.append(equation("r_v = R_i^T (v_j - v_i - g dt) - Delta_v_ij", st))
    story.append(equation("r_p = R_i^T (p_j - p_i - v_i dt - 0.5 g dt^2) - Delta_p_ij", st))
    story.append(equation("r_R = Log(Delta_R_ij^{-1} R_i^T R_j)", st))
    story.append(
        p(
            "九维残差和 9×9 covariance 的公共排列均为 [p,v,R]。Bias 通过预积分 Jacobian 做一阶修正；当 Bias 改变量超出一阶有效范围时，理论上应由 raw IMU repropagation 更新缓存。",
            st,
        )
    )
    story.append(p("2.4 Bias random-walk 与 prior", st, "Heading2"))
    story.append(equation("r_b = [b_a,j - b_a,i, b_g,j - b_g,i]", st))
    story.append(
        p(
            "Bias RW covariance 随 dt 增长，描述 Bias 状态的过程噪声，并不与 IMU measurement factor 重复。窗口滑动时，旧状态和与其相关的旧因子通过 Schur complement 压缩为新 prior；新 prior 围绕边缘化时保存的线性化状态求值。",
            st,
        )
    )
    story.append(p("2.5 位姿参考点契约", st, "Heading2"))
    story.append(
        p(
            "优化器内部状态是 IMU/body 位姿 T_WI。输入视觉状态先按 T_WI=T_WC T_CI 从 CameraLeftSocket 转到 IMU，Direct UVD 因子求值时再由 body 状态和固定外参恢复相机位姿；优化完成后，为兼容 MACVO Map，历史 poses.csv 仍写回相机中心。最新导出同时提供 poses_imu.csv，后续评价以该文件为规范输入。",
            st,
        )
    )
    story.append(equation("p_WI = p_WC + R_WC (t_BI - t_BC)", st))
    story.append(equation("p_WI,gt = p_WB,gt + R_WB,gt t_BI", st))
    story.append(
        p(
            "其中 t_BI 和 t_BC 来自 metadata。ref_pose.csv 的位置字段没有在 metadata 中明确命名物理传感器点；矩形原地转向的微米级位移强烈支持其为 body/root 旋转原点，因此本报告将 GT 从 body/root 转换到 IMU 中心，并保留这一经验假设的审计记录。",
            st,
        )
    )

    # Chapter 3
    story.append(p("3. 四种视觉接入方法", st, "Heading1"))
    story.append(p("3.1 T-pose factor", st, "Heading2"))
    story.append(
        p(
            "MACVO 先使用点级观测求出视觉相对位姿均值 Z_ij，再把点级信息压缩为 6D pose covariance。VIO 后端使用以下残差：",
            st,
        )
    )
    story.append(equation("r_pose = Log(Z_ij^{-1} T_ij(x_i, x_j))", st))
    story.append(
        p(
            "该方法每条视觉 edge 只有 6 个残差标量，接口清晰且计算量低。代价是点级非高斯结构、鲁棒权重和局部几何被压缩，pose covariance 的近似误差会直接改变视觉与 IMU 的竞争关系。",
            st,
        )
    )
    story.append(p("3.2 U1：Direct UVD factor", st, "Heading2"))
    story.append(
        p(
            "U1 保留 MACVO 原生匹配点、UVD 观测和 covariance，MACVO 的相对位姿只作为 warm start。联合优化器直接计算每个点的 UVD 预测误差，使点级视觉和 IMU 在同一次求解中共同决定前后状态。它去掉了“点 -> MACVO 位姿 -> 6D pose factor”的信息压缩，但圆形全序列运行约 57.9 分钟，当前更适合作为点级精度基线。",
            st,
        )
    )
    story.append(p("3.3 SA-v1：单 edge Sampling-aware covariance", st, "Heading2"))
    story.append(
        p(
            "SA-v1 保持 U1 的状态、视觉因子、LM 和四个 sigma 不变，只重新传播单条 IMU edge 的 covariance。它显式考虑图像边界插值和 midpoint 中原始采样的复用，Delta 均值不变。SA-v1 是统计模型修复，不是平滑后处理。",
            st,
        )
    )
    story.append(p("3.4 SA-v2：跨 edge 相关噪声", st, "Heading2"))
    story.append(
        p(
            "SA-v2 在 SA-v1 上继续建模相邻 edge 共享边界 IMU 样本产生的交叉 covariance。单边 covariance 被分解为独立部分与前后端点共享部分：",
            st,
        )
    )
    story.append(equation("P_total = P_unique + S_in S_in^T + S_out S_out^T", st))
    story.append(
        p(
            "SA-v2 的信息结构更完整，但 P_unique 可以合法低秩，因此白化和 Schur prior 必须支持奇异高斯；把零特征值简单抬高会制造虚假的超强信息方向。",
            st,
        )
    )
    method_data = [
        ["方法", "视觉信息", "covariance", "主要优势", "主要风险"],
        ["T-pose", "6D 相对位姿", "6×6 pose", "快、接口简单", "点级压缩损失"],
        ["U1", "点级 UVD", "逐点", "保留视觉结构", "计算量高"],
        ["SA-v1", "点级 UVD", "单 edge sampling", "采样统计正确", "更信任带噪 IMU"],
        ["SA-v2", "点级 UVD", "跨 edge correlation", "保留共享噪声", "低秩与 prior 数值风险"],
    ]
    story.append(styled_table(method_data, [26 * mm, 33 * mm, 37 * mm, 34 * mm, 35 * mm], font_size=8.2))

    # Chapter 4
    story.append(p("4. 数学正确性与预积分审计", st, "Heading1"))
    story.append(p("4.1 IMU Jacobian 修复", st, "Heading2"))
    story.append(
        p(
            "审计发现 IMU factor 的 pose-rotation Jacobian 错误，与 PyPose issue #395 的 translation accessor backward 问题高度吻合。将 rel_ij.translation() 替换为数学等价的 rel_ij.Act(zeros) 后，重新执行 50 组以上随机、非零旋转和非零 Bias 状态的中心有限差分检查。",
            st,
        )
    )
    jac_data = [
        ["检查项", "结果"],
        ["非零项最大绝对误差", "5.157596e-8"],
        ["非零项最大相对误差", "3.013280e-6"],
        ["理论零块最大绝对值", "1.332268e-15"],
        ["300 帧 replay", "209/209 edge 收敛，无 NaN/Inf"],
    ]
    story.append(styled_table(jac_data, [82 * mm, 82 * mm], font_size=9))
    story.append(
        callout(
            "结论边界",
            "该结果证明当前 IMU 因子的导数与流形扰动定义一致；它不证明视觉 covariance、Bias 可观性或窗口长度已经最优。",
            st,
            fill=PALE_BLUE,
            accent=BLUE,
        )
    )
    story.append(p("4.2 标准局部坐标系预积分", st, "Heading2"))
    story.append(
        p(
            "Legacy 路径曾使用外部/全局姿态做重力补偿或世界系积分，导致 factor cache 与独立姿态传播线程耦合。标准实现只接收 raw body-frame measured acceleration、measured angular velocity、dt、Bias 线性化点和噪声参数；delta_R、delta_v、delta_p 内部不使用外部 T_WB，重力只出现在 IMU residual。",
            st,
        )
    )
    preint_data = [
        ["模式", "GT r_v median / m/s"],
        ["Legacy external-attitude compensation", "1.630e-3"],
        ["Standard local-frame preintegration", "2.519e-6"],
        ["Truth reintegration reference", "1.293e-6"],
    ]
    story.append(styled_table(preint_data, [112 * mm, 52 * mm], font_size=8.8))
    story.append(p("4.3 MACVIO 与 GTSAM covariance / NIS", st, "Heading2"))
    story.append(
        p(
            "跨项目审计确认两边使用相同数据集、相同 metadata、相同 IMU CSV、相同 edge 和相同四个运行时 sigma。统一到公共 [P,V,R] 切空间后，Normal-noise 下两边的 Delta 和完整 covariance 高度一致。",
            st,
        )
    )
    for item in [
        "Noisy Delta 最大差：dp=1.301e-8 m，dv=8.844e-7 m/s，dR=1.527e-7 rad。",
        "NIS9 mean：MACVIO=6.6450，GTSAM=6.0349，均小于理论期望 9，说明 covariance 整体偏保守。",
        "Bias RW 公共 NIS6 mean=5.2903；GT/Q 随机游走模型通过，但 MACVIO 在线 Bias 仍明显过度活跃。",
    ]:
        story.append(bullet(item, st))
    story.append(
        callout(
            "因果判断",
            "当前抖动不能简单归因为 MACVIO 独有的 F/G/Q、dt 缩放或排列错误。两边预积分基本一致，下一层问题在状态可观性和误差分配。",
            st,
            fill=PALE_TEAL,
            accent=TEAL,
        )
    )

    # Chapter 5
    story.append(p("5. Sampling-aware、Bias 与窗口消融", st, "Heading1"))
    story.extend(figure(FIG / "bias_window_ablation.png", "图 1  Bias 消融与 fixed-lag 窗口长度/计算开销对比。", st, width=165 * mm, max_height=95 * mm))
    story.append(p("5.1 Sampling-aware covariance 的统计验证", st, "Heading2"))
    story.append(
        p(
            "Sampling-aware Monte Carlo 的 NIS9 均值为 8.9970，P/V/R 分块均值分别为 3.0029、2.9976、2.9954。完整 9×9 covariance 相对 Monte Carlo 的 Frobenius 误差中位数从旧模型的 16.71% 降至 2.93%。因此采样映射和 covariance 传播在统计上通过。",
            st,
        )
    )
    story.append(
        p(
            "但是，统计正确不等于轨迹更平滑。Sampling-aware P 的 trace 平均更小，意味着 IMU 信息量更高。在 N=2 中，它更紧地追随带噪 IMU，使 XY 高频 RMS 和 pose correction 二阶差分均恶化约 10.6%。不能为了观感回退到错误 covariance，也不能把 covariance 修复当作平滑算法。",
            st,
        )
    )
    story.append(p("5.2 Bias 消融", st, "Heading2"))
    bias_data = [
        ["模式", "XY 高频 RMS / m", "t-RPE / m", "ATE / m", "解释"],
        ["B1 optimize ba/bg", "0.012535", "0.018605", "0.2113", "正常基线"],
        ["B2 fixed static ba/bg", "0.009876", "0.012669", "0.6361", "低频姿态恶化"],
        ["B3 GT bias oracle", "0.009924", "0.011257", "0.2484", "仅用于诊断"],
        ["B4 optimize ba only", "0.016076", "0.022075", "1.2017", "最差"],
        ["B5 fixed ba / opt bg", "0.005462", "0.006665", "0.1758", "高频最佳"],
    ]
    story.append(styled_table(bias_data, [43 * mm, 31 * mm, 27 * mm, 25 * mm, 39 * mm], font_size=7.8))
    story.append(
        p(
            "正常 B1 的 ba increment 约为理论 RW 尺度的 85.7 倍。固定 ba、继续优化 bg 的 B5 使 XY 高频、correction 二阶差分和 velocity 高频分别下降 56.4%、51.7% 和 68.7%。B2 固定全部静止 Bias 虽能降低高频，却因静止 bg 不准而造成明显低频误差，因此“永久固定全部 Bias”不是最终方案。",
            st,
        )
    )
    story.append(p("5.3 窗口长度", st, "Heading2"))
    window_data = [
        ["N", "XY 高频 / m", "t-RPE / m", "ATE / m", "ba RW norm", "ms/solve"],
        ["2", "0.012534", "0.018609", "0.2121", "85.8", "367.6"],
        ["3", "0.012202", "0.018132", "0.2107", "85.8", "668.2"],
        ["5", "0.011507", "0.017213", "0.2081", "85.7", "1461.3"],
        ["10", "0.009895", "0.015097", "0.2028", "84.5", "4315.8"],
    ]
    story.append(styled_table(window_data, [20 * mm, 31 * mm, 30 * mm, 27 * mm, 28 * mm, 29 * mm], font_size=8.2))
    story.append(
        callout(
            "主次关系",
            "Bias 自由度是当前短片段抖动的主因；更长窗口只能部分缓解。N=10 的计算量超过 4.3 s/solve，且 ba 活跃度仍约为理论尺度的 84.5 倍。",
            st,
            fill=PALE_RED,
            accent=RED,
        )
    )

    # Chapter 6
    story.append(p("6. Normal-noise 三场景完整轨迹", st, "Heading1"))
    story.append(
        p(
            "本章统一比较 GT、Pure MACVO、T-pose factor、U1、SA-v1 和 SA-v2。所有估计从 CameraLeftSocket 转到 IMU 中心，GT 从经验确认的 body/root 原点转到 IMU 中心；每条轨迹仅减去自身首帧平移，不做 SE(3)、yaw 或尺度拟合。圆形和停转矩形时长约 63 s，直线约 21 s。",
            st,
        )
    )
    story.extend(figure(IMU_CENTER_ROOT / "imu_center_raw_cross_scene_metrics.png", "图 2  统一到 IMU 中心后的三场景 XY ATE、translation RPE 与 rotation RPE。", st, width=165 * mm, max_height=88 * mm))
    story.extend(figure(IMU_CENTER_ROOT / "imu_center_raw_three_scenes_xy.png", "图 3  统一到 IMU 中心后的三场景原始优化器轨迹。", st, width=165 * mm, max_height=61 * mm))

    add_scene_table(
        story,
        st,
        "6.1 圆形场景",
        [
            ["Pure MACVO", "2.4214", "0.01134", "0.000194"],
            ["T-pose factor", "3.0246", "0.02470", "0.000385"],
            ["U1", "2.1152", "0.01557", "0.000271"],
            ["SA-v1", "2.1603", "0.01633", "0.000286"],
            ["SA-v2", "1.9703", "0.03929", "0.001363"],
        ],
    )
    story.append(
        p(
            "参考点统一后，所有方法的圆形 XY ATE 均降低约 0.15--0.21 m，但排序未翻转。U1/SA-v1/SA-v2 仍明显优于 T-pose 的全局 ATE；SA-v2 虽拥有最低 ATE，却在 61.67 s 附近出现大跳变，translation/rotation RPE 最差。U1 仍是局部稳定性和全局误差更均衡的点级基线。",
            st,
        )
    )

    add_scene_table(
        story,
        st,
        "6.2 停转矩形场景",
        [
            ["Pure MACVO", "0.8674", "0.00975", "0.000212"],
            ["T-pose factor", "0.4500", "0.01872", "0.000342"],
            ["U1", "0.5582", "0.01232", "0.000221"],
            ["SA-v1", "0.5518", "0.01301", "0.000232"],
            ["SA-v2", "0.6789", "0.01308", "0.000651"],
        ],
    )
    story.append(
        p(
            "矩形受杆臂参考点影响最大：Pure MACVO 的 XY ATE 从旧评估的 1.2544 m 降为 0.8674 m。T-pose 仍有最低 ATE，U1/SA-v1 的局部 RPE 更好。Pure MACVO 转角弧线在转到 IMU 中心后缩小，但没有消失，说明外参参考点混用只解释了部分弧线，剩余部分仍是视觉相对运动误差。",
            st,
        )
    )

    add_scene_table(
        story,
        st,
        "6.3 直线场景",
        [
            ["Pure MACVO", "0.5636", "0.00265", "0.000078"],
            ["T-pose factor", "0.3066", "0.01288", "0.000262"],
            ["U1", "0.4140", "0.00616", "0.000122"],
            ["SA-v1", "0.4108", "0.00672", "0.000134"],
            ["SA-v2", "0.4162", "0.00655", "0.000130"],
        ],
    )
    story.append(
        p(
            "直线几乎没有持续转动，杆臂转换对指标影响不足 0.003 m。Pure MACVO 仍拥有最小局部 RPE，但横向漂移持续积累；T-pose ATE 最低，U1、SA-v1 和 SA-v2 基本重合。该场景也反向验证了本次指标变化主要来自转弯时的参考点几何，而不是全局缩放或拟合。",
            st,
        )
    )
    story.append(
        callout(
            "跨场景结论",
            "统一参考点没有改变“无统一冠军”的结论。SA-v2 圆形 ATE 最低但局部失稳；T-pose 在矩形和直线的 ATE 较低却牺牲局部 RPE；U1/SA-v1 仍是更均衡的点级基线。旧指标不能继续与新指标混用。",
            st,
            fill=PALE_BLUE,
            accent=BLUE,
        )
    )

    # Chapter 7
    story.append(p("7. SA-v2 61.67 s 跳变根因", st, "Heading1"))
    story.extend(
        figure(
            ROOT / "analysis_circle_sa_v2_outlier_6167_20260717" / "sa_v2_late_outlier_diagnostic.png",
            "图 4  保存轨迹单步、edge 内相对平移和下一 edge 旧状态回写对比。",
            st,
            width=165 * mm,
            max_height=112 * mm,
        )
    )
    story.append(p("7.1 不是单条 measurement 突然变大", st, "Heading2"))
    story.append(
        p(
            "在 frame 1850，内部 edge 相对平移只有 0.001588 m，但保存轨迹的 XY 单步达到 0.466748 m。可见大位移不是当前视觉或 IMU measurement 直接产生，而是下一条 two-state edge 对旧状态 state_i 的共同模式重写。",
            st,
        )
    )
    story.append(p("7.2 低秩 P_unique 是触发条件", st, "Heading2"))
    story.append(
        p(
            "1799 条有效 edge 中，600 条 P_unique 的有效秩为 6/9。三个缺失方向不是“零噪声真值”，而是已经被共享端点 latent noise 解释。Legacy 白化对零特征值施加固定小正数下限，相当于在这些方向注入近乎无限的信息。",
            st,
        )
    )
    story.append(p("7.3 Continuous prior 是放大器", st, "Heading2"))
    story.extend(
        figure(
            ROOT / "analysis_circle_sa_v2_prior_rank_aware_20260717" / "prior_condition_and_common_update.png",
            "图 5  Rank-aware 处理对共同平移更新与边缘化 Hessian 条件数的影响。",
            st,
            width=165 * mm,
            max_height=108 * mm,
        )
    )
    for item in [
        "仅在 source frame 1794 清空一次 prior，59.8--63 s 最大 XY 单步从 0.931217 m 降至 0.007767 m。",
        "Rank-aware 保留 continuous prior 时，晚段最大单步为 0.013007 m，且 frame 1850 的 H 条件数改善约 42.7 倍。",
        "Rank-aware 的全局 XY ATE 从 legacy 的 2.166606 m 恶化到 2.342218 m，说明安全 fallback 丢失了相关信息。",
    ]:
        story.append(bullet(item, st))
    story.append(
        callout(
            "根因链",
            "低秩 P_unique + 硬 eigen floor -> 虚假强信息；continuous Schur prior -> 长期放大共同模式；下一 edge 回写旧状态 -> 保存轨迹出现可见跳变。",
            st,
            fill=PALE_RED,
            accent=RED,
        )
    )
    story.append(
        p(
            "Rank-aware fallback 是有效的防跳保护，但不是最终精度解。完整方案应在 range(P_unique) 中白化独立噪声，并通过 null-space/QR 显式保留共享端点 latent-noise，同时对每次 Schur complement 保留二次型等价断言。",
            st,
        )
    )

    # Chapter 8
    story.append(p("8. 相机、body 与 IMU 中心统一", st, "Heading1"))
    story.append(p("8.1 发现的问题不是后端状态定义错误", st, "Heading2"))
    story.append(
        p(
            "代码审计确认，T-pose、U1、SA-v1 和 SA-v2 的优化状态均以 IMU/body 为中心；视觉相对位姿均值和 covariance 在进入后端前也通过固定外参转换到同一 body 切空间。问题发生在输出与历史评价层：为兼容 MACVO Map，poses.csv 保存的是相机中心，而 ref_pose.csv 的位置更符合 body/root 旋转原点。直接比较二者会把杆臂圆弧错误计入定位误差。",
            st,
        )
    )
    story.append(p("8.2 GT 参考点的数值证据", st, "Heading2"))
    for item in [
        "矩形四段约 90 度原地转向中，raw GT 起止 chord 仅约 6--9 微米。",
        "metadata 的 body->IMU 水平杆臂约 0.12 m；转到 IMU 中心后，GT 转向 chord 约 0.1655 m，符合 90 度杆臂圆弧量级。",
        "若把 raw GT 错当相机中心再转 IMU，转向 chord 约 0.629 m，与原地转向几何不符。",
        "metadata 未直接命名 ref_pose 位置传感器，因此 body/root 结论是强数值证据支持的经验分类，而不是无条件事实。",
    ]:
        story.append(bullet(item, st))
    story.append(p("8.3 统一转换与回归结果", st, "Heading2"))
    story.append(
        p(
            "本次共转换三场景、每场景 21 条估计轨迹，合计 63 条。估计使用 camera->IMU 杆臂，GT 使用 body/root->IMU 杆臂；所有转换后的轨迹分别在首帧平移重基准化。运行时 tensor_map 中的 T_CI 在 15 个原始运行中均逐帧恒定，并与 metadata 匹配，最大误差为 7.15e-9。31 项参考点、视觉因子、两状态 VIO 与输出滤波回归测试通过。",
            st,
        )
    )
    story.append(
        callout(
            "物理解释更新",
            "Pure MACVO 矩形平均转向 chord 从相机中心的 1.1955 m 降至 IMU 中心的 0.8704 m，但 IMU-center GT 仅为 0.1655 m。参考点混用是真问题，却不是视觉转向误差的全部原因。",
            st,
            fill=PALE_ORANGE,
            accent=ORANGE,
        )
    )
    story.append(p("8.4 输出 ESKF 的最新边界", st, "Heading2"))
    output_filter_data = [
        ["场景/输入", "输出", "XY ATE / m", "t-RPE / m", "r-RPE / rad"],
        ["Circle / U1", "raw", "2.1152", "0.01557", "0.000271"],
        ["Circle / U1", "3D gate", "2.1253", "0.00790", "0.000139"],
        ["Circle / SA-v2", "raw", "1.9703", "0.03929", "0.001363"],
        ["Circle / SA-v2", "3D gate", "1.9905", "0.00808", "0.000155"],
        ["Rectangle / U1", "raw", "0.5582", "0.01232", "0.000221"],
        ["Rectangle / U1", "3D gate", "0.5716", "0.01033", "0.005263"],
        ["Straight / U1", "raw", "0.4140", "0.00616", "0.000122"],
        ["Straight / U1", "3D gate", "0.4140", "0.00325", "0.000056"],
    ]
    story.append(
        styled_table(
            output_filter_data,
            [37 * mm, 27 * mm, 31 * mm, 31 * mm, 38 * mm],
            font_size=7.6,
        )
    )
    story.append(
        p(
            "3D output ESKF 对圆形和直线的局部 translation/rotation RPE 有明显改善，并能压制 SA-v2 晚段跳变；但在停转矩形中，U1 的 rotation RPE 从 2.21e-4 rad 恶化到 5.26e-3 rad。它目前只能作为离线输出平滑实验，不能反馈因子图，也不能宣称已成为跨场景可靠的实时姿态滤波器。完整 63 条结果可在交互页面中按优化器和滤波方法筛选。",
            st,
        )
    )

    # Chapter 9
    story.append(p("9. U1 冻结与平移/旋转 Schur 分解", st, "Heading1"))
    story.append(p("9.1 为什么重新检验自适应硬切换", st, "Heading2"))
    story.append(
        p(
            "此前提出的自适应方案假设：直线、转弯或过渡阶段可能分别更适合只保留视觉平移、只保留视觉旋转或使用完整 UVD。为避免启发式固定变量和额外 IMU 锚造成重复计数，本轮从同一 UVD 局部二次型出发，用 Schur complement 严格边缘化另一组位姿自由度。完整非线性 Direct-UVD U1 已冻结到 Baselines/direct_uvd_u1_standard_20260719，并通过 SHA256 复核；生产默认未被诊断分支替换。",
            st,
        )
    )
    story.append(
        p(
            "设局部增量为 [dt,dR]，完整视觉二次型分块为 H_tt、H_tR、H_Rt、H_RR 和 g_t、g_R。保留平移时使用 H_t^m = H_tt - H_tR H_RR^dagger H_Rt，g_t^m = g_t - H_tR H_RR^dagger g_R；保留旋转时使用对称形式。鲁棒权重固定在 UVD 线性化点，不加入额外 IMU anchor。",
            st,
        )
    )
    schur_test_data = [
        ["验证项", "结果"],
        ["完整局部因子 H/g", "与直接鲁棒 UVD 一致"],
        ["Schur 边缘二次型", "与完整 profiled quadratic 一致"],
        ["被消去变量列", "数值为零"],
        ["两状态 Jacobian", "中心有限差分通过"],
        ["UVD 相关测试", "14 passed，无 NaN/Inf"],
    ]
    story.append(styled_table(schur_test_data, [73 * mm, 91 * mm], font_size=8.8))

    story.append(p("9.2 矩形分段反事实协议", st, "Heading2"))
    for item in [
        "Rectangle normal-noise，活动首帧至 frame 724，共捕获 633 条有效 edge。",
        "Static、straight、transition、turn 各 12 个 seed，共 48 个相同起点、192 个分支。",
        "比较 nonlinear full、linearized full control、translation marginal 和 rotation marginal；每个 seed 向后看 3 条 edge。",
        "所有分支共享 incoming state、prior、IMU、UVD 和初始化；候选 prior 不回写生产链。",
        "GT 只用于运动段标注和离线评分，不进入因子构造、门控或求解。",
    ]:
        story.append(bullet(item, st))
    story.append(
        p(
            "Linearized-full 控制组相对非线性 U1 的 pair-state 差异 P95 为 0.004577，最大值为 0.009745，低于预设 0.01 门限；最大误差集中在转弯段，说明固定线性化在最强非线性阶段的代表性最弱。",
            st,
        )
    )

    story.append(p("9.3 三边 lookahead 结果", st, "Heading2"))
    schur_result_data = [
        ["阶段", "模式", "XY RMSE / m", "姿态 RMSE / rad", "收敛率"],
        ["Static", "Full", "0.773120", "0.006938", "91.7%"],
        ["", "Translation", "0.773771", "0.007053", "100%"],
        ["", "Rotation", "0.775271", "0.006942", "100%"],
        ["Straight", "Full", "0.749588", "0.006369", "100%"],
        ["", "Translation", "0.749377", "0.006389", "100%"],
        ["", "Rotation", "0.750341", "0.006391", "100%"],
        ["Transition", "Full", "0.808599", "0.008463", "100%"],
        ["", "Translation", "0.809190", "0.008619", "100%"],
        ["", "Rotation", "0.810840", "0.008455", "100%"],
        ["Turn", "Full", "0.907994", "0.019299", "91.7%"],
        ["", "Translation", "0.904510", "0.019422", "100%"],
        ["", "Rotation", "0.905564", "0.019403", "100%"],
    ]
    story.append(styled_table(schur_result_data, [25 * mm, 36 * mm, 34 * mm, 42 * mm, 27 * mm], font_size=7.5))
    for item in [
        "Straight 的 translation marginal 位置只改善 0.028%，姿态反而恶化 0.31%。",
        "Turn 的 translation marginal 位置改善 0.384%，姿态恶化 0.637%；位置胜出 10/12，姿态仅 3/12。",
        "Turn 的 rotation marginal 位置改善 0.268%，姿态仍恶化 0.537%。",
        "局部 Schur 分支更容易收敛，但固定二次型的数值难度更低，不能把收敛率直接解释为精度优势。",
    ]:
        story.append(bullet(item, st))

    story.append(p("9.4 高耦合与最终判定", st, "Heading2"))
    coupling_data = [
        ["阶段", "t/R Hessian 耦合中位数", "P95"],
        ["Static", "0.9428", "0.9548"],
        ["Straight", "0.9479", "0.9545"],
        ["Transition", "0.9464", "0.9588"],
        ["Turn", "0.9532", "0.9644"],
    ]
    story.append(styled_table(coupling_data, [55 * mm, 65 * mm, 44 * mm], font_size=8.5))
    story.append(
        callout(
            "判定更新",
            "不批准在生产后端中按帧硬切换 full / translation-only / rotation-only。UVD 平移与旋转信息高度耦合，测试分支没有获得幅值足够、跨 seed 稳定且同时改善位置与姿态的收益。生产继续使用完整非线性 U1；Schur 因子仅保留为诊断工具。下一项优先实验改为在完整 UVD、相同 covariance 和初始化下比较 N=2/N=5/N=10 fixed-lag。",
            st,
            fill=PALE_RED,
            accent=RED,
        )
    )
    story.append(
        p(
            "证据边界：本轮只覆盖矩形第一段长直线、停车/过渡和第一次完整转弯，不能证明所有场景下分解模式永远无效；但已足以否决立即训练模式分类器或接入生产。",
            st,
        )
    )

    # Chapter 10
    story.append(p("10. 结论、证据分级与下一步", st, "Heading1"))
    story.append(p("10.1 已由数值实验直接证明", st, "Heading2"))
    for item in [
        "IMU factor Jacobian、[p,v,R] 排列和理论零块正确。",
        "标准 local-frame preintegration 在 GT 上闭合到数值积分误差量级。",
        "MACVIO 与 GTSAM 的预积分均值和 covariance 基本一致。",
        "Sampling-aware covariance 的采样传播在 Monte Carlo 中统计正确。",
        "点级 UVD 在圆形完整序列上显著优于 T-pose 压缩。",
        "加速度计 Bias 过度活跃是当前短片段高频抖动的首要贡献。",
        "SA-v2 晚段跳变由低秩 whitening、continuous prior 和旧状态回写共同造成。",
        "后端内部状态为 IMU 中心；历史 poses.csv 的相机中心输出与 body/root GT 参考点曾在评价层混用。",
        "63 条历史估计统一到 IMU 中心后，三场景方法排序的主结论未发生翻转。",
        "3D output ESKF 可改善圆形/直线局部 RPE，但当前矩形转角姿态更新不安全。",
        "UVD 局部 Schur 分解的 14 项测试通过；矩形分段实验没有发现平移/旋转硬切换的实质联合收益。",
    ]:
        story.append(bullet(item, st))
    story.append(p("10.2 代码与数学审查认为合理", st, "Heading2"))
    for item in [
        "T-pose factor 的 6D 压缩会丢失点级鲁棒权重、非高斯结构和局部几何。",
        "N=2 对 Bias 与速度的瞬时误差分配过于自由，较长窗口只能部分缓解。",
        "Rank-aware fallback 比硬特征值 floor 更安全，但会丢失跨 edge 相关信息。",
        "Pure MACVO 转角弧线在参考点转换后仍显著大于 GT，剩余误差主要来自视觉相对运动，而不是杆臂漏算。",
        "UVD 平移/旋转 Hessian 的高耦合解释了为什么硬删除一组自由度很难得到无代价收益。",
    ]:
        story.append(bullet(item, st))
    story.append(p("10.3 尚未验证", st, "Heading2"))
    for item in [
        "奇异高斯/null-space SA-v2 能否兼顾 legacy ATE 与 rank-aware 稳定性。",
        "相同因子、相同 covariance 下 N=2/5/10 完整序列的收益是否值得实时成本。",
        "在不硬删除位姿分量的前提下，连续软门控或条件 covariance 是否能提供跨场景稳定收益。",
        "离线 3D ESKF 的平滑收益能否在不反馈因子图、不重复计数的实时输出线程中复现。",
        "HoloOcean 生成侧能否从源代码明确确认 ref_pose.csv 的 position reference point。",
    ]:
        story.append(bullet(item, st))
    story.append(p("10.4 建议执行顺序", st, "Heading2"))
    next_data = [
        ["优先级", "工作", "保持不变", "验收要点"],
        ["P0", "冻结 IMU-center 评估契约", "原始结果文件", "禁止混用旧相机中心指标"],
        ["P1", "N=2/5/10 fixed-lag", "完整 UVD 与 covariance", "ATE/RPE/高频/成本联合 gate"],
        ["P2", "奇异相关高斯 SA-v2", "edge 均值与状态", "Schur 二次型等价、无 jump"],
        ["P3", "连续软门控诊断", "不硬删除 t/R", "跨 seed 联合收益"],
        ["P4", "实时 3D output ESKF", "不反馈 factor graph", "只改善输出平滑且不重复计数"],
    ]
    story.append(styled_table(next_data, [18 * mm, 47 * mm, 48 * mm, 52 * mm], font_size=8))
    story.append(
        callout(
            "最终判断",
            "本周已经把数学正确性、参考点正确性和算法精度分开验证。继续单独收紧 ba 的边际收益有限，平移/旋转硬模式切换也没有获得生产依据。下一步保持完整非线性 UVD、相同 covariance 与初始化，优先完成 N=2/N=5/N=10 fixed-lag 对照；在窗口证据明确前不训练模式分类器，也不通过手工权重掩盖短窗误差分配。",
            st,
            fill=PALE_TEAL,
            accent=TEAL,
        )
    )

    # Appendix
    story.append(p("附录 A. 一周更新树与复现入口", st, "Heading1"))
    timeline = [
        ["日期", "阶段", "目的", "结果"],
        ["7/13--14", "两状态基线与静止初始化", "建立完整因子链", "可运行，暴露 Jacobian/预积分风险"],
        ["7/15", "数学审计", "验证扰动、排列、prior", "Jacobian 通过；209/209 收敛"],
        ["7/15", "标准 local preintegration", "移除外部姿态重力补偿", "GT residual 降至积分误差量级"],
        ["7/15", "MACVIO/GTSAM NIS", "区分实现与共同模型问题", "Delta/P 一致；均偏保守"],
        ["7/16", "Direct UVD U1", "检查 pose 压缩损失", "圆形 ATE/RPE 显著改善"],
        ["7/16", "SA-v1/Bias/Window", "采样统计与抖动定位", "Sampling 正确；ba 为主因"],
        ["7/17", "SA-v2 跳变审计", "解释 61.67 s 离群点", "低秩 + prior + rewrite 根因链"],
        ["7/18", "三场景完整对比", "检查泛化", "无统一冠军"],
        ["7/19", "IMU 中心统一复评估", "消除参考点混用", "63 条轨迹完成转换，排序未翻转"],
        ["7/19", "U1 Schur 分段反事实", "检验 t/R/full 硬切换", "无实质联合收益，不接入生产"],
    ]
    story.append(styled_table(timeline, [22 * mm, 42 * mm, 46 * mm, 55 * mm], font_size=7.6))
    story.append(p("A.1 主要复现产物", st, "Heading2"))
    for item in [
        "三场景交互轨迹：analysis_normal_noise_u1_sa_v2_full_three_scenes_20260718/interactive_all_methods_three_scenes.html",
        "三场景数值指标：analysis_normal_noise_u1_sa_v2_full_three_scenes_20260718/trajectory_metrics.csv",
        "U1/T-pose 全圆报告：analysis_circle_direct_uvd_u1_vs_pose_factor_full_20260716/full_comparison_report_cn.md",
        "Sampling/Bias/Window：analysis_normal_noise_sampling_aware_20260716/macvio_sampling_aware_covariance_report_cn.md",
        "SA-v2 跳变：analysis_circle_sa_v2_prior_rank_aware_20260717/sa_v2_prior_rank_aware_root_cause_report_cn.md",
        "MACVIO/GTSAM NIS：analysis_macvio_vs_gtsam_normal_noise_covariance_nis_20260715/macvio_vs_gtsam_normal_noise_comparison_cn.md",
        "IMU 中心交互轨迹：analysis_imu_center_all_methods_20260719/interactive_imu_center_all_methods.html",
        "IMU 中心指标：analysis_imu_center_all_methods_20260719/imu_center_accuracy_metrics.csv",
        "参考点审计：analysis_imu_center_all_methods_20260719/imu_center_reference_audit_report_cn.md",
        "U1 Schur 分段报告：analysis_rectangle_uvd_schur_marginal_20260719/rectangle_uvd_schur_marginal_report_cn.md",
        "U1 Schur 决策：analysis_rectangle_uvd_schur_marginal_20260719/rectangle_uvd_schur_marginal_decision.json",
    ]:
        story.append(bullet(item, st))
    story.append(p("A.2 建议组会上重点讨论", st, "Heading2"))
    for item in [
        "完整 UVD 的 N=2/N=5/N=10 fixed-lag 收益是否足以覆盖实时计算成本？",
        "U1 的点级精度收益是否值得当前约 58 分钟/圆形序列的计算成本？",
        "实时 3D output ESKF 是否只作为展示层，不向因子图反馈？",
        "生产验收应如何权衡 ATE、RPE、高频误差与 max-step safety gate？",
    ]:
        story.append(bullet(item, st))
    return story


def build_focused_story(st):
    """Build the decision-focused report requested after the broad audit report."""
    story = []

    # Cover
    story.append(Spacer(1, 18 * mm))
    story.append(Paragraph("METHOD DECISION REVIEW · 2026.07.19", st["CoverKicker"]))
    story.append(Paragraph("MACVO-VIO 方法决策报告", st["CoverTitle"]))
    story.append(Paragraph("EKF · 优化器选择 · 自适应模式收益", st["CoverSubtitle"]))
    story.append(Paragraph("统一 IMU 中心 · Normal-noise · NWU-XY", st["CoverKicker"]))
    story.append(Spacer(1, 48 * mm))
    story.append(
        callout(
            "报告范围",
            "正文只回答三个工程问题：三维输出 EKF 是否可用；T-factor、U1、SA-v1、SA-v2 如何平衡精度与计算开销；按帧切换 full、translation-only、rotation-only 是否有收益。数学正确性、预积分和参考点审计只作为附录证据。",
            st,
            fill=PALE_BLUE,
            accent=BLUE,
        )
    )
    story.append(Spacer(1, 7 * mm))
    story.append(
        callout(
            "当前决策",
            "生产研究默认保持完整非线性 Direct-UVD U1；T-factor 保留为低成本基线。当前输出 ESKF 不跨场景安全，自适应平移/旋转硬切换没有实证收益。下一步优先比较完整 UVD 的 N=2/N=5/N=10 fixed-lag。",
            st,
            fill=PALE_TEAL,
            accent=TEAL,
        )
    )
    story.append(NextPageTemplate("Normal"))
    story.append(PageBreak())

    # Executive summary and TOC
    story.append(Paragraph("决策摘要", st["Title"]))
    decision_data = [
        ["决策块", "证据摘要", "结论"],
        ["三维输出 ESKF", "圆形/直线 RPE 约减半；矩形转角旋转 RPE 最多放大约 23 倍", "暂不接生产"],
        ["四种优化器", "T-factor 最快；U1 最均衡；SA-v1 无稳定收益；SA-v2 有跳变风险", "U1 默认，T 降级"],
        ["自适应硬切换", "位置收益小于 0.4%，姿态多为恶化；t/R 耦合约 0.94--0.95", "不批准"],
    ]
    story.append(styled_table(decision_data, [35 * mm, 88 * mm, 42 * mm], font_size=8.2))
    story.append(Spacer(1, 7 * mm))
    story.append(
        p(
            "所有精度结果已统一到 IMU 中心。每条轨迹只减去自身首帧平移，不做 SE(3)、yaw 或尺度拟合；GT 只用于离线评分。运行时间来自现有缓存回放的墙钟记录，不同批次并发数不完全一致，因此只解释工程量级。",
            st,
        )
    )
    story.append(PageBreak())
    story.append(Paragraph("目录", st["TOCTitle"]))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("FocusTOC1", fontName=FONT, fontSize=10, leading=14, leftIndent=0, textColor=NAVY),
        ParagraphStyle("FocusTOC2", fontName=FONT, fontSize=8.5, leading=12, leftIndent=12, textColor=MUTED),
    ]
    story.append(toc)
    story.append(PageBreak())

    # Part 1: EKF
    story.append(p("1. 三维输出 ESKF 的结果", st, "Heading1"))
    story.append(p("1.1 它只平滑输出，不修复因子图", st, "Heading2"))
    story.append(
        p(
            "当前 ESKF 位于因子图之后，状态包含三维位置、速度、SO(3) 姿态和角速度，观测为优化器输出的三维位置与姿态。它不反馈 IMU 预积分、Bias、LM、Schur prior 或视觉因子，因此不会重复使用同一测量，也不能修复已经形成的系统漂移。",
            st,
        )
    )
    story.append(
        callout(
            "圆形离线结果",
            "固定过程噪声并启用分块门控的 D 模式，使 translation RPE 降低约 46%--78%，位置二阶差分降低 88%--94%，旋转二阶差分降低 90%--98%；XY ATE 仅变化约 +0.2%--+0.8%。CPU 回放约 0.25--0.29 ms/帧。",
            st,
            fill=PALE_TEAL,
            accent=TEAL,
        )
    )
    story.append(p("1.2 三场景泛化", st, "Heading2"))
    ekf_data = [
        ["场景", "XY ATE Raw -> D / m", "t-RPE Raw -> D / m", "r-RPE Raw -> D / rad"],
        ["Circle / U1", "2.1152 -> 2.1253", "0.01557 -> 0.00790", "2.708e-4 -> 1.389e-4"],
        ["Straight / U1", "0.4140 -> 0.4140", "0.00616 -> 0.00325", "1.216e-4 -> 5.641e-5"],
        ["Rectangle / U1", "0.5582 -> 0.5716", "0.01231 -> 0.01033", "2.214e-4 -> 5.263e-3"],
        ["Circle / SA-v2", "1.9703 -> 1.9905", "0.03929 -> 0.00808", "1.363e-3 -> 1.548e-4"],
    ]
    story.append(styled_table(ekf_data, [34 * mm, 42 * mm, 42 * mm, 47 * mm], font_size=7.7))
    story.append(
        p(
            "圆形和直线的 U1 translation/rotation RPE 大约降低一半，SA-v2 圆形离群也被明显压制；但矩形 U1 rotation RPE 增大约 22.8 倍。四种优化器输入在矩形 D 模式下都出现转角姿态恶化，说明当前过程模型或门控不能可靠描述停车后原地转向。",
            st,
        )
    )
    story.extend(
        figure(
            IMU_CENTER_ROOT / "imu_center_selected_xy.png",
            "图 1  三场景 IMU-center 轨迹。图中为自适应 3D ESKF E 模式，用于说明平滑后低频轨迹形状仍基本由原优化器决定。",
            st,
            width=165 * mm,
            max_height=72 * mm,
        )
    )
    story.append(
        callout(
            "EKF 判定",
            "计算开销足够小，平滑能力已经证明；但它不改善 ATE，且矩形转角姿态不安全。当前只保留离线输出实验。修复转向过程模型、因果 Q/R、后端 pose covariance 和实时线程生命周期后，再决定是否接入生产输出。",
            st,
            fill=PALE_RED,
            accent=RED,
        )
    )
    story.append(PageBreak())

    # Part 2: optimizer choice
    story.append(p("2. T-factor、U1、SA-v1、SA-v2 的选择", st, "Heading1"))
    story.append(p("2.1 方法与信息保留程度", st, "Heading2"))
    method_data = [
        ["方法", "视觉输入", "IMU covariance", "工程定位"],
        ["T-factor", "6D 相对位姿", "标准 local-frame", "最快，信息压缩最多"],
        ["U1", "点级 UVD", "标准 local-frame", "点级信息完整，当前最均衡"],
        ["SA-v1", "点级 UVD", "单 edge sampling-aware", "统计研究，未形成轨迹收益"],
        ["SA-v2", "点级 UVD", "跨 edge correlation", "低秩/prior 风险，研究分支"],
    ]
    story.append(styled_table(method_data, [30 * mm, 39 * mm, 50 * mm, 46 * mm], font_size=7.9))
    story.extend(
        figure(
            IMU_CENTER_ROOT / "imu_center_raw_cross_scene_metrics.png",
            "图 2  统一到 IMU 中心后的三场景原始优化器指标。ATE 与局部 RPE 必须同时阅读。",
            st,
            width=165 * mm,
            max_height=62 * mm,
        )
    )
    story.append(p("2.2 精度对比", st, "Heading2"))
    accuracy_data = [
        ["场景", "方法", "XY ATE / m", "t-RPE / m", "r-RPE / rad"],
        ["Circle", "T-factor", "3.0246", "0.02470", "3.854e-4"],
        ["", "U1", "2.1152", "0.01557", "2.708e-4"],
        ["", "SA-v1", "2.1603", "0.01633", "2.857e-4"],
        ["", "SA-v2", "1.9703", "0.03929", "1.363e-3"],
        ["Rectangle", "T-factor", "0.4500", "0.01872", "3.424e-4"],
        ["", "U1", "0.5582", "0.01231", "2.214e-4"],
        ["", "SA-v1", "0.5518", "0.01301", "2.325e-4"],
        ["", "SA-v2", "0.6789", "0.01308", "6.510e-4"],
        ["Straight", "T-factor", "0.3066", "0.01288", "2.617e-4"],
        ["", "U1", "0.4140", "0.00616", "1.216e-4"],
        ["", "SA-v1", "0.4108", "0.00672", "1.342e-4"],
        ["", "SA-v2", "0.4162", "0.00655", "1.301e-4"],
    ]
    story.append(styled_table(accuracy_data, [30 * mm, 33 * mm, 31 * mm, 33 * mm, 38 * mm], font_size=7.3))
    story.append(Spacer(1, 3 * mm))
    story.append(p("2.3 缓存回放计算开销", st, "Heading2"))
    runtime_data = [
        ["方法", "Circle s/帧", "Rectangle s/帧", "Straight s/帧", "中位数", "相对 T"],
        ["T-factor", "0.454", "0.547", "0.395", "0.454", "1.00x"],
        ["U1", "1.839", "1.736", "1.747", "1.747", "3.85x"],
        ["SA-v1", "2.360", "2.451", "2.405", "2.405", "5.30x"],
        ["SA-v2", "3.039", "3.013", "2.802", "3.013", "6.64x"],
    ]
    story.append(styled_table(runtime_data, [27 * mm, 27 * mm, 31 * mm, 30 * mm, 25 * mm, 25 * mm], font_size=7.8))
    story.append(
        p(
            "现有实现均未达到 30 Hz。绝对时间不是严格同并发 microbenchmark，但足以显示量级：U1 约为 T-factor 的 3.85 倍，SA-v1/SA-v2 进一步增加到约 5.30/6.64 倍。点级残差、逐点 covariance 白化与自动微分是主要新增开销。",
            st,
        )
    )
    story.append(
        callout(
            "优化器选择",
            "T-factor 在矩形/直线 ATE 上占优且最快，但局部 RPE 较差；U1 的三场景精度与稳定性最均衡，作为当前研究默认。SA-v1 没有形成足以抵消额外开销的稳定收益；SA-v2 虽有最低圆形 ATE，却拥有最差圆形 RPE、最高开销和已知跳变风险。当前 Pareto 前沿主要由 T-factor 与 U1 构成。",
            st,
            fill=PALE_TEAL,
            accent=TEAL,
        )
    )
    story.append(PageBreak())

    # Part 3: adaptive method
    story.append(p("3. 自适应平移/旋转模式是否有收益", st, "Heading1"))
    story.append(p("3.1 数学上公平的检验", st, "Heading2"))
    story.append(
        p(
            "本轮没有使用早期“固定另一半位姿并加 IMU 锚”的启发式分支，而是在同一 UVD 线性化点形成 6D pose 二次型，再用 Schur complement 严格边缘化旋转或平移。该做法保留被消去变量的不确定性，不额外重复计入 IMU。相关 14 项 UVD/Schur 测试通过。",
            st,
        )
    )
    for item in [
        "Rectangle normal-noise，从活动首帧到 frame 724，共 633 条有效 edge。",
        "Static、straight、transition、turn 各 12 个 seed，共 48 个起点、192 个分支。",
        "每个分支共享 incoming state、prior、IMU、UVD 和初始化，并向后评价 3 条 edge。",
        "GT 只用于分段和评分，不进入因子构造、门控或求解。",
    ]:
        story.append(bullet(item, st))
    story.append(p("3.2 相对完整非线性 U1 的变化", st, "Heading2"))
    adaptive_data = [
        ["阶段", "分支", "XY RMSE 变化", "姿态 RMSE 变化"],
        ["Static", "Translation", "+0.084%", "+1.667%"],
        ["", "Rotation", "+0.278%", "+0.066%"],
        ["Straight", "Translation", "-0.028%", "+0.311%"],
        ["", "Rotation", "+0.101%", "+0.343%"],
        ["Transition", "Translation", "+0.073%", "+1.840%"],
        ["", "Rotation", "+0.277%", "-0.102%"],
        ["Turn", "Translation", "-0.384%", "+0.637%"],
        ["", "Rotation", "-0.268%", "+0.537%"],
    ]
    story.append(styled_table(adaptive_data, [37 * mm, 44 * mm, 42 * mm, 42 * mm], font_size=8.0))
    story.append(
        p(
            "负值表示改善。转弯 translation marginal 的位置胜出率为 10/12，但姿态仅 3/12；位置收益小于 0.4%，姿态同时恶化。其他阶段也没有幅值足够、跨 seed 稳定且位置/姿态联合改善的分支。",
            st,
        )
    )
    coupling_data = [
        ["阶段", "t/R Hessian 耦合中位数", "P95"],
        ["Static", "0.9428", "0.9548"],
        ["Straight", "0.9479", "0.9545"],
        ["Transition", "0.9464", "0.9588"],
        ["Turn", "0.9532", "0.9644"],
    ]
    story.append(styled_table(coupling_data, [55 * mm, 65 * mm, 45 * mm], font_size=8.4))
    story.append(
        p(
            "高耦合意味着同一批 UVD 点对平移和旋转的约束不是两组可无代价拆开的信息。Linearized-full 控制相对非线性 U1 的 pair-state 差异 P95 为 0.004577、最大值 0.009745，且最大差异集中于转弯段；局部因子更易收敛不能解释为精度更好。",
            st,
        )
    )
    story.append(
        callout(
            "自适应判定",
            "不批准 full / translation-only / rotation-only 硬切换，也不训练模式分类器。Schur 分支保留为诊断工具。若未来继续研究自适应，应考虑连续软门控、条件 covariance 或鲁棒信息缩放，但优先级低于完整 UVD 的窗口长度实验。",
            st,
            fill=PALE_RED,
            accent=RED,
        )
    )

    # Final plan and compact evidence appendix
    story.append(p("4. 最终建议与执行顺序", st, "Heading1"))
    final_data = [
        ["优先级", "工作", "保持不变", "验收重点"],
        ["P0", "冻结 U1 与 IMU-center 契约", "现有结果文件", "禁止混用旧参考点指标"],
        ["P1", "N=2/N=5/N=10 fixed-lag", "完整 UVD/covariance", "ATE、RPE、高频、成本"],
        ["P2", "T-factor/U1 严格 profile", "同硬件/同并发", "残差、白化、AD、求解占比"],
        ["P3", "修复 3D output ESKF 转角", "不反馈因子图", "矩形姿态回归与因果 Q/R"],
        ["P4", "SA-v2/软自适应研究", "非生产分支", "无 jump 且存在联合收益"],
    ]
    story.append(styled_table(final_data, [19 * mm, 45 * mm, 48 * mm, 53 * mm], font_size=7.9))
    story.append(Spacer(1, 7 * mm))
    story.append(
        callout(
            "一句话路线",
            "精度优先时保留 U1，算力优先时保留 T-factor；先解决完整 UVD 的短窗口误差分配，再决定输出 ESKF 与更复杂 covariance。当前没有理由把 SA-v1、SA-v2 或平移/旋转硬切换升级为生产默认。",
            st,
            fill=PALE_TEAL,
            accent=TEAL,
        )
    )
    story.append(p("附录 A. 支撑证据", st, "Heading1"))
    evidence_data = [
        ["证据", "结果", "本报告中的作用"],
        ["IMU Jacobian", "非零项 max abs 5.16e-8；209/209 收敛", "排除基础导数错误"],
        ["标准预积分", "GT r_v median 2.52e-6 m/s", "排除外部姿态重力补偿"],
        ["MACVIO/GTSAM", "Delta/P 高度一致；NIS 均偏保守", "不把抖动归因于独有 F/G/Q"],
        ["Sampling-aware", "Monte Carlo NIS9 mean 8.997", "证明 SA-v1 传播统计正确"],
        ["UVD Schur", "14 tests passed，无 NaN/Inf", "保证自适应反事实数学一致"],
        ["参考点", "63 条轨迹统一到 IMU 中心", "保证精度表物理点一致"],
    ]
    story.append(styled_table(evidence_data, [43 * mm, 58 * mm, 64 * mm], font_size=7.8))
    story.append(p("A.1 复现入口", st, "Heading2"))
    for item in [
        "统一精度：analysis_imu_center_all_methods_20260719/imu_center_accuracy_metrics.csv",
        "三维 ESKF：analysis_circle_output_eskf3d_ablation_20260718/offline_validation_report_cn.md",
        "四种方法：analysis_normal_noise_u1_sa_v2_full_three_scenes_20260718/trajectory_metrics.csv",
        "运行时间：各 Results 目录下 progress.csv",
        "自适应实验：analysis_rectangle_uvd_schur_marginal_20260719/rectangle_uvd_schur_marginal_report_cn.md",
        "冻结 U1：Baselines/direct_uvd_u1_standard_20260719/",
    ]:
        story.append(bullet(item, st))
    return story


def main() -> None:
    register_fonts()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    st = styles()
    doc = ReportDocTemplate(
        str(OUT),
        pagesize=A4,
        title="MACVO-VIO 方法决策：EKF、优化器选择与自适应收益",
        author="MACVO-VIO project",
    )
    doc.multiBuild(build_focused_story(st))
    print(OUT)


if __name__ == "__main__":
    main()
