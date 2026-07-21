#!/usr/bin/env python3
"""Build the landscape group-meeting PDF for the 2026-07-13--18 VIO review."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "analysis_weekly_group_report_20260718"
FIG = REPORT_ROOT / "figures"
OUT = ROOT / "output" / "pdf" / "macvo_vio_weekly_group_meeting_report_20260718.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")

PAGE_W, PAGE_H = landscape(A4)
FONT = "SimHei"

NAVY = HexColor("#0F172A")
INK = HexColor("#1E293B")
MUTED = HexColor("#64748B")
LIGHT = HexColor("#F8FAFC")
LINE = HexColor("#CBD5E1")
BLUE = HexColor("#2563EB")
TEAL = HexColor("#059669")
ORANGE = HexColor("#F97316")
RED = HexColor("#DC2626")
PURPLE = HexColor("#7C3AED")
PALE_BLUE = HexColor("#EFF6FF")
PALE_TEAL = HexColor("#ECFDF5")
PALE_ORANGE = HexColor("#FFF7ED")
PALE_RED = HexColor("#FEF2F2")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(FONT, str(FONT_PATH)))


def text_width(text: str, size: float) -> float:
    return pdfmetrics.stringWidth(text, FONT, size)


def wrap_text(text: str, width: float, size: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        if char == "\n":
            lines.append(current)
            current = ""
            continue
        candidate = current + char
        if current and text_width(candidate, size) > width:
            lines.append(current.rstrip())
            current = char.lstrip()
        else:
            current = candidate
    if current or not lines:
        lines.append(current.rstrip())
    return lines


def draw_wrapped(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    size: float = 11,
    leading: float | None = None,
    color=INK,
    max_lines: int | None = None,
) -> float:
    leading = leading or size * 1.45
    lines = wrap_text(text, width, size)
    if max_lines is not None:
        lines = lines[:max_lines]
    c.setFont(FONT, size)
    c.setFillColor(color)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_bullets(
    c: canvas.Canvas,
    items: list[str],
    x: float,
    y: float,
    width: float,
    size: float = 10.5,
    gap: float = 7,
    bullet_color=BLUE,
    text_color=INK,
) -> float:
    for item in items:
        lines = wrap_text(item, width - 20, size)
        c.setFillColor(bullet_color)
        c.circle(x + 4, y + 4, 2.7, stroke=0, fill=1)
        c.setFont(FONT, size)
        c.setFillColor(text_color)
        for index, line in enumerate(lines):
            c.drawString(x + 16, y - index * size * 1.42, line)
        y -= len(lines) * size * 1.42 + gap
    return y


def header(c: canvas.Canvas, title: str, section: str, page_no: int) -> None:
    c.setFillColor(NAVY)
    c.rect(0, PAGE_H - 58, PAGE_W, 58, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont(FONT, 20)
    c.drawString(36, PAGE_H - 38, title)
    c.setFont(FONT, 8.5)
    c.setFillColor(HexColor("#BFDBFE"))
    c.drawRightString(PAGE_W - 36, PAGE_H - 35, section)
    c.setStrokeColor(LINE)
    c.line(36, 27, PAGE_W - 36, 27)
    c.setFont(FONT, 7.5)
    c.setFillColor(MUTED)
    c.drawString(36, 14, "2026-07-13 至 2026-07-18 · NWU-XY · 无对齐 / 无尺度拟合")
    c.drawRightString(PAGE_W - 36, 14, f"{page_no:02d}")


def callout(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    fill=PALE_BLUE,
    accent=BLUE,
    title_size: float = 11,
    body_size: float = 9.5,
    title_color=None,
    body_color=None,
) -> None:
    c.setFillColor(fill)
    c.roundRect(x, y, width, height, 7, stroke=0, fill=1)
    c.setFillColor(accent)
    c.roundRect(x, y, 5, height, 2, stroke=0, fill=1)
    c.setFillColor(title_color or INK)
    c.setFont(FONT, title_size)
    c.drawString(x + 16, y + height - 22, title)
    draw_wrapped(
        c,
        body,
        x + 16,
        y + height - 40,
        width - 30,
        body_size,
        body_size * 1.35,
        body_color or MUTED,
    )


def draw_image_fit(
    c: canvas.Canvas,
    path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
    border: bool = True,
) -> None:
    reader = ImageReader(str(path))
    image_w, image_h = reader.getSize()
    scale = min(width / image_w, height / image_h)
    draw_w = image_w * scale
    draw_h = image_h * scale
    draw_x = x + (width - draw_w) / 2
    draw_y = y + (height - draw_h) / 2
    if border:
        c.setFillColor(white)
        c.setStrokeColor(LINE)
        c.roundRect(x, y, width, height, 6, stroke=1, fill=1)
    c.drawImage(reader, draw_x, draw_y, draw_w, draw_h, mask="auto")


def draw_table(
    c: canvas.Canvas,
    x: float,
    y_top: float,
    col_widths: list[float],
    rows: list[list[str]],
    row_height: float = 24,
    header_height: float = 29,
    font_size: float = 8.5,
) -> float:
    total_w = sum(col_widths)
    c.setFillColor(NAVY)
    c.roundRect(x, y_top - header_height, total_w, header_height, 5, stroke=0, fill=1)
    cursor_x = x
    c.setFillColor(white)
    c.setFont(FONT, font_size)
    for cell, width in zip(rows[0], col_widths):
        c.drawCentredString(cursor_x + width / 2, y_top - 19, cell)
        cursor_x += width
    y = y_top - header_height
    for row_index, row in enumerate(rows[1:]):
        c.setFillColor(white if row_index % 2 == 0 else LIGHT)
        c.rect(x, y - row_height, total_w, row_height, stroke=0, fill=1)
        cursor_x = x
        c.setFillColor(INK)
        c.setFont(FONT, font_size)
        for col_index, (cell, width) in enumerate(zip(row, col_widths)):
            if col_index >= 2:
                c.drawRightString(cursor_x + width - 8, y - 16, cell)
            else:
                c.drawString(cursor_x + 7, y - 16, cell)
            cursor_x += width
        c.setStrokeColor(HexColor("#E2E8F0"))
        c.line(x, y - row_height, x + total_w, y - row_height)
        y -= row_height
    c.setStrokeColor(LINE)
    c.roundRect(x, y, total_w, y_top - y, 5, stroke=1, fill=0)
    return y


def arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float) -> None:
    c.setStrokeColor(BLUE)
    c.setFillColor(BLUE)
    c.setLineWidth(2)
    c.line(x1, y1, x2, y2)
    c.saveState()
    c.translate(x2, y2)
    c.rotate(0 if x2 >= x1 else 180)
    c.line(0, 0, -9, 5)
    c.line(0, 0, -9, -5)
    c.restoreState()


def begin_page(c: canvas.Canvas, title: str, section: str, page_no: int) -> None:
    header(c, title, section, page_no)


def end_page(c: canvas.Canvas) -> None:
    c.showPage()


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(PAGE_W, PAGE_H))
    c.setTitle("MACVO-VIO 两状态融合：一周技术进展与 Normal-noise 结果")
    c.setAuthor("MACVO-VIO project")

    # 1. Cover
    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    c.setFillColor(BLUE)
    c.rect(0, 0, 13, PAGE_H, stroke=0, fill=1)
    c.setFillColor(HexColor("#93C5FD"))
    c.setFont(FONT, 11)
    c.drawString(54, PAGE_H - 72, "WEEKLY TECHNICAL REVIEW · 2026.07.13--2026.07.18")
    c.setFillColor(white)
    c.setFont(FONT, 34)
    c.drawString(54, PAGE_H - 150, "MACVO-VIO 两状态融合")
    c.setFont(FONT, 24)
    c.setFillColor(HexColor("#E2E8F0"))
    c.drawString(54, PAGE_H - 193, "从数学审计到 Normal-noise 三场景验证")
    c.setFillColor(HexColor("#94A3B8"))
    c.setFont(FONT, 12)
    c.drawString(56, PAGE_H - 235, "T-pose factor · Direct UVD U1 · SA-v1 · SA-v2")
    callout(
        c,
        54,
        118,
        360,
        120,
        "核心判断",
        "基础 IMU 数学链和 Sampling-aware covariance 已由数值测试验证；当前主要矛盾已转向点级视觉、过度自由的加速度计 Bias、N=2 窗口与相关 prior 的误差分配。",
        fill=HexColor("#172554"),
        accent=HexColor("#60A5FA"),
        body_size=11,
        title_color=white,
        body_color=HexColor("#CBD5E1"),
    )
    c.setFillColor(HexColor("#CBD5E1"))
    c.setFont(FONT, 10)
    c.drawString(56, 72, "主评估：NWU-XY · 无轨迹对齐 · 无尺度修正")
    c.setFillColor(HexColor("#475569"))
    c.circle(PAGE_W - 160, 170, 105, stroke=1, fill=0)
    c.setStrokeColor(HexColor("#2563EB"))
    c.setLineWidth(7)
    c.arc(PAGE_W - 265, 65, PAGE_W - 55, 275, 20, 245)
    c.setStrokeColor(HexColor("#10B981"))
    c.setLineWidth(4)
    c.arc(PAGE_W - 240, 90, PAGE_W - 80, 250, 215, 255)
    c.setFillColor(white)
    c.setFont(FONT, 13)
    c.drawCentredString(PAGE_W - 160, 170, "IMU + VISION")
    end_page(c)

    # 2. Executive summary
    begin_page(c, "本周结论：先证明正确，再讨论精度", "Executive summary", 2)
    callout(c, 38, 416, 245, 98, "01 · 数学链已通过", "IMU Jacobian 最大绝对误差 5.16e-8；标准预积分 GT r_v 中位数 2.52e-6 m/s；209/209 edge 收敛。", PALE_TEAL, TEAL)
    callout(c, 298, 416, 245, 98, "02 · 点级视觉更有效", "圆形全序列中，U1 相对 T-pose 的 ATE / 平移 RPE / 旋转 RPE 改善 28.8% / 36.3% / 29.7%。", PALE_BLUE, BLUE)
    callout(c, 558, 416, 245, 98, "03 · Sampling 不是平滑器", "Monte Carlo NIS9=8.997，但更正确的 P 给 IMU 更高信息量，N=2 高频反而约恶化 10.6%。", PALE_ORANGE, ORANGE)
    callout(c, 38, 286, 245, 98, "04 · Bias 是当前主因", "ba increment 约为理论 RW 的 85.7 倍；固定 ba、继续优化 bg 后，XY 高频误差下降 56.4%。", PALE_RED, RED)
    callout(c, 298, 286, 245, 98, "05 · SA-v2 跳变已定位", "低秩 P_unique + 硬 eigen floor 产生虚假强约束，continuous prior 放大后由旧状态回写显现。", PALE_ORANGE, ORANGE)
    callout(c, 558, 286, 245, 98, "06 · 没有统一冠军", "SA-v2 圆形 ATE 最低但局部失稳；T-pose 在矩形/直线 ATE 较低；U1/SA-v1 更均衡。", PALE_BLUE, PURPLE)
    c.setFillColor(NAVY)
    c.roundRect(38, 82, 765, 148, 8, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont(FONT, 15)
    c.drawString(58, 202, "当前工程判断")
    draw_bullets(
        c,
        [
            "不再通过调四个 sigma 或 LM damping 掩盖结构问题。",
            "下一优先级是共享/低频 ba 参数化，其次是奇异相关高斯 SA-v2。",
            "所有后续方法必须同时报告 ATE、RPE、高频误差和最大单步。",
        ],
        58,
        174,
        715,
        11,
        8,
        bullet_color=HexColor("#60A5FA"),
        text_color=HexColor("#E2E8F0"),
    )
    end_page(c)

    # 3. Data and architecture
    begin_page(c, "数据契约与系统边界", "What exactly is being estimated?", 3)
    callout(c, 38, 372, 235, 142, "HoloOcean 数据", "IMU 为 body FLU，线加速度包含重力；GT 位置/速度为 world NWU。相机、IMU、GT 共享 tick，时间 offset=0。Normal-noise 在传感器端真实加入。", PALE_BLUE, BLUE, body_size=10)
    callout(c, 303, 372, 235, 142, "视觉前端", "MACVO 输出双目匹配、UVD/3D 不确定性与相对位姿 warm start。相邻两帧的网络预测不依赖更早轨迹；全局 T_WC 连乘会累积历史误差。", PALE_ORANGE, ORANGE, body_size=10)
    callout(c, 568, 372, 235, 142, "IMU 预积分", "factor cache 只使用 raw body-frame IMU、dt、bias linearization point 和噪声；重力仅出现在 residual，不再依赖外部姿态。", PALE_TEAL, TEAL, body_size=10)
    c.setFillColor(white)
    c.setStrokeColor(LINE)
    c.roundRect(48, 104, 175, 175, 8, stroke=1, fill=1)
    c.roundRect(333, 104, 175, 175, 8, stroke=1, fill=1)
    c.roundRect(618, 104, 175, 175, 8, stroke=1, fill=1)
    for x, title, lines, color in [
        (48, "MACVO", ["UVD / covariance", "T_ij warm start", "point quality"], ORANGE),
        (333, "Two-state backend", ["pose / velocity", "ba / bg", "Schur prior"], BLUE),
        (618, "输出与诊断", ["T_WC trajectory", "ATE / RPE", "NIS / max step"], TEAL),
    ]:
        c.setFillColor(color)
        c.roundRect(x, 244, 175, 35, 8, stroke=0, fill=1)
        c.setFillColor(white)
        c.setFont(FONT, 12)
        c.drawCentredString(x + 87.5, 256, title)
        draw_bullets(c, lines, x + 18, 218, 145, 10, 6, bullet_color=color)
    arrow(c, 229, 191, 326, 191)
    arrow(c, 514, 191, 611, 191)
    c.setFillColor(MUTED)
    c.setFont(FONT, 8.5)
    c.drawCentredString(278, 202, "视觉因子")
    c.drawCentredString(563, 202, "优化结果")
    end_page(c)

    # 4. Backend math
    begin_page(c, "两状态 fixed-lag 后端：优化什么", "State and factors", 4)
    callout(c, 38, 421, 365, 94, "状态", "x_k = {T_WB,k, v_k^W, b_a,k, b_g,k}。活动图包含 x_i 与 x_j；历史信息通过 prior(x_i) 进入，而不是把上一帧固定成真值。", PALE_BLUE, BLUE, body_size=10.5)
    callout(c, 438, 421, 365, 94, "求解形式", "这是窗口 N=2 的非线性 MAP / fixed-lag smoother，不是 EKF。每条新边优化两状态，再边缘化旧状态并生成新 prior。", PALE_TEAL, TEAL, body_size=10.5)
    c.setFillColor(LIGHT)
    c.roundRect(38, 236, 765, 148, 8, stroke=0, fill=1)
    c.setFont(FONT, 13)
    c.setFillColor(INK)
    c.drawString(58, 355, "标准 IMU residual（公共排列 [p, v, R]）")
    c.setFont(FONT, 12)
    c.drawString(68, 321, "r_v = R_i^T (v_j - v_i - g Δt) - Δv_ij")
    c.drawString(68, 288, "r_p = R_i^T (p_j - p_i - v_i dt - 0.5 g dt^2) - Delta_p_ij")
    c.drawString(68, 255, "r_R = Log(Delta_R_ij^{-1} R_i^T R_j)")
    callout(c, 38, 82, 235, 116, "IMU factor", "用完整 9×9 covariance 白化 [r_p,r_v,r_R]；Bias 通过 Jacobian 一阶修正预积分。", PALE_BLUE, BLUE, body_size=10)
    callout(c, 303, 82, 235, 116, "Bias RW factor", "r_b=[b_a,j-b_a,i, b_g,j-b_g,i]，covariance 随 Δt 增长。它描述过程噪声，不是重复的 IMU measurement。", PALE_ORANGE, ORANGE, body_size=10)
    callout(c, 568, 82, 235, 116, "Marginal prior", "Schur complement 保留被移除状态的历史信息，并围绕固定线性化点求值；重复计数必须被显式避免。", PALE_TEAL, TEAL, body_size=10)
    end_page(c)

    # 5. Four methods
    begin_page(c, "四种视觉接入方法", "T-pose → U1 → SA-v1 → SA-v2", 5)
    methods = [
        ("T-pose", "6D pose residual", "点级视觉先由 MACVO 压缩成相对位姿均值和 6×6 covariance。快，但损失局部几何与鲁棒权重。", RED, PALE_RED),
        ("U1", "Direct UVD", "保留点级 UVD/covariance；MACVO pose 仅作 warm start。避免 pose-factor 压缩，但计算量显著增大。", BLUE, PALE_BLUE),
        ("SA-v1", "Single-edge sampling", "U1 状态和视觉不变；只修正单条 IMU edge 内边界插值与原始样本复用的 covariance。", PURPLE, HexColor("#F5F3FF")),
        ("SA-v2", "Cross-edge correlation", "继续建模相邻 edge 共享端点样本的交叉 covariance；信息更完整，也更依赖合法的低秩处理。", TEAL, PALE_TEAL),
    ]
    x_positions = [38, 237, 436, 635]
    for x, (name, subtitle, body, accent, fill) in zip(x_positions, methods):
        c.setFillColor(fill)
        c.roundRect(x, 170, 168, 335, 8, stroke=0, fill=1)
        c.setFillColor(accent)
        c.roundRect(x, 453, 168, 52, 8, stroke=0, fill=1)
        c.setFillColor(white)
        c.setFont(FONT, 17)
        c.drawCentredString(x + 84, 475, name)
        c.setFillColor(INK)
        c.setFont(FONT, 10.5)
        c.drawCentredString(x + 84, 425, subtitle)
        draw_wrapped(c, body, x + 15, 396, 138, 9.8, 14.2, MUTED)
        c.setStrokeColor(accent)
        c.setLineWidth(2)
        c.line(x + 22, 282, x + 146, 282)
        label = {
            "T-pose": "视觉残差：6",
            "U1": "视觉残差：3N",
            "SA-v1": "P：单 edge",
            "SA-v2": "P：跨 edge",
        }[name]
        c.setFillColor(accent)
        c.setFont(FONT, 11)
        c.drawCentredString(x + 84, 252, label)
        lower = {
            "T-pose": "低计算量基线",
            "U1": "点级精度基线",
            "SA-v1": "统计正确单边",
            "SA-v2": "相关噪声研究",
        }[name]
        c.setFillColor(INK)
        c.setFont(FONT, 9.5)
        c.drawCentredString(x + 84, 205, lower)
    c.setFillColor(MUTED)
    c.setFont(FONT, 10)
    c.drawCentredString(PAGE_W / 2, 112, "方法越往右，保留的信息越多；同时计算、数值秩和 prior 一致性问题也更突出。")
    end_page(c)

    # 6. Audit chain
    begin_page(c, "正确性审计链", "What is proven, and what is not", 6)
    audit_rows = [
        ["检查项", "关键结果", "能够证明", "不能证明"],
        ["IMU Jacobian", "max abs 5.16e-8", "导数与扰动一致", "融合精度最优"],
        ["理论零块", "1.33e-15", "依赖关系正确", "可观性充分"],
        ["300-frame replay", "209/209 收敛", "数值求解稳定", "长期轨迹稳定"],
        ["GT preintegration", "r_v median 2.52e-6", "局部预积分闭合", "噪声模型完美"],
        ["MACVIO vs GTSAM", "Delta/P 高度一致", "非单方 F/G/Q 错误", "P 已完全校准"],
        ["Sampling Monte Carlo", "NIS9 8.997", "采样映射正确", "轨迹一定更平滑"],
        ["Bias ablation", "固定 ba 高频 -56.4%", "ba 是主贡献", "最终应永久固定 ba"],
        ["SA-v2 reset", "max step 0.931→0.0078", "prior 是因果放大器", "reset 是生产方案"],
    ]
    draw_table(c, 40, 502, [145, 150, 220, 225], audit_rows, row_height=42, header_height=32, font_size=9.2)
    callout(c, 40, 65, 740, 64, "审计原则", "每一项测试只回答它能回答的问题。求解器“收敛”不等于模型正确；NIS 总量正常也不等于各轴/各分块正确；ATE 更低也不等于局部稳定。", PALE_ORANGE, ORANGE, body_size=10)
    end_page(c)

    # 7. U1 vs pose factor
    begin_page(c, "为什么从 T-pose 转向 U1", "Full circle · normal noise", 7)
    c.setFillColor(LIGHT)
    c.roundRect(38, 278, 765, 235, 8, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont(FONT, 15)
    c.drawString(58, 484, "圆形全序列（1890 帧，无对齐）")
    rows = [
        ["指标", "Pure MACVO", "T-pose", "U1", "U1 vs T-pose"],
        ["XY ATE RMSE / m", "2.5709", "3.2380", "2.3041", "-28.84%"],
        ["最终 XY 误差 / m", "3.2577", "3.6087", "2.5951", "-28.09%"],
        ["Translation RPE / m", "0.01183", "0.02500", "0.01592", "-36.32%"],
        ["Rotation RPE / rad", "0.000194", "0.000385", "0.000271", "-29.73%"],
    ]
    draw_table(c, 58, 450, [190, 130, 120, 120, 150], rows, row_height=36, header_height=31, font_size=9.5)
    callout(c, 38, 137, 365, 105, "证据支持", "将点级 UVD 直接送入联合优化器，明显优于把同一批视觉信息先压缩成 6D pose factor。视觉压缩与 pose covariance 是原精度下降的重要来源。", PALE_BLUE, BLUE, body_size=10.5)
    callout(c, 438, 137, 365, 105, "仍未解决", "U1 的局部 RPE 仍差于 Pure MACVO，且圆形仍有圆心、半径和闭合误差。IMU、Bias 和 fixed-lag prior 仍在给逐帧相对运动注入扰动。", PALE_ORANGE, ORANGE, body_size=10.5)
    c.setFillColor(MUTED)
    c.setFont(FONT, 9.5)
    c.drawString(40, 95, "代价：U1 圆形全序列运行约 57.9 分钟；它是点级精度基线，不是当前实时方案。")
    end_page(c)

    # 8. Bias/window figure
    begin_page(c, "Sampling-aware 之后，抖动指向 Bias 与窗口", "Controlled 300-frame ablations", 8)
    draw_image_fit(c, FIG / "bias_window_ablation.png", 38, 143, 765, 365)
    callout(c, 38, 67, 765, 58, "主次关系", "Sampling covariance 已通过 Monte Carlo；固定 ba 带来的高频改善（52%--69%）远大于单纯加窗。N=10 仅部分缓解，却把平均求解时间推到 4.3 s/solve。", PALE_RED, RED, body_size=10)
    end_page(c)

    # 9. Cross-scene metrics
    begin_page(c, "Normal-noise 三场景：没有统一冠军", "Cross-scene metrics", 9)
    draw_image_fit(c, FIG / "cross_scene_metrics.png", 38, 140, 765, 370)
    callout(c, 38, 65, 765, 58, "读图方式", "Circle、Rectangle、Straight 的运动激励不同。必须同时看 ATE 与局部 RPE：SA-v2 的圆形 ATE 最低，但其晚段跳变使局部误差最差。", PALE_BLUE, BLUE, body_size=10)
    end_page(c)

    # 10-12 trajectory pages
    trajectory_pages = [
        ("圆形轨迹：持续旋转暴露长期误差", "trajectory_circle.png", "U1/SA-v1/SA-v2 整体优于 T-pose，但所有方法仍有圆心、半径和闭合误差；SA-v2 晚段离群点说明仅看 ATE 会误判。", 10),
        ("停转矩形：转角与停车处的误差分配", "trajectory_rectangle.png", "T-pose 的 XY ATE 最低但局部环状扰动明显；U1/SA-v1 在全局位置与局部 RPE 之间更均衡。", 11),
        ("直线：低频横向漂移与局部 RPE", "trajectory_straight.png", "Pure MACVO 局部 RPE 最好但持续横漂；T-pose ATE 较低却有低频摆动；U1/SA-v1/SA-v2 基本重合。", 12),
    ]
    for title, filename, note, number in trajectory_pages:
        begin_page(c, title, "Normal-noise trajectory", number)
        image_height = 388 if "straight" not in filename else 350
        image_y = 120 if "straight" not in filename else 140
        draw_image_fit(c, FIG / filename, 38, image_y, 765, image_height)
        callout(c, 38, 58, 765, 52, "结论", note, PALE_BLUE, BLUE, body_size=9.8)
        end_page(c)

    # 13 detailed metrics
    begin_page(c, "三场景数值表", "ATE and RPE · no alignment", 13)
    metric_rows = [
        ["场景", "方法", "XY ATE / m", "t-RPE / m", "r-RPE / rad"],
        ["Circle", "Pure MACVO", "2.5709", "0.01183", "0.000194"],
        ["Circle", "T-pose", "3.2380", "0.02500", "0.000385"],
        ["Circle", "U1", "2.3041", "0.01592", "0.000271"],
        ["Circle", "SA-v1", "2.3515", "0.01667", "0.000286"],
        ["Circle", "SA-v2", "2.1666", "0.03924", "0.001363"],
        ["Rectangle", "Pure MACVO", "1.2544", "0.01075", "0.000212"],
        ["Rectangle", "T-pose", "0.6645", "0.01896", "0.000342"],
        ["Rectangle", "U1", "0.6893", "0.01269", "0.000221"],
        ["Rectangle", "SA-v1", "0.6934", "0.01337", "0.000233"],
        ["Rectangle", "SA-v2", "0.7756", "0.01342", "0.000651"],
        ["Straight", "Pure MACVO", "0.5660", "0.00263", "0.000078"],
        ["Straight", "T-pose", "0.3072", "0.01280", "0.000262"],
        ["Straight", "U1", "0.4151", "0.00612", "0.000122"],
        ["Straight", "SA-v1", "0.4119", "0.00668", "0.000134"],
        ["Straight", "SA-v2", "0.4173", "0.00651", "0.000130"],
    ]
    draw_table(c, 62, 505, [125, 180, 135, 135, 145], metric_rows, row_height=24, header_height=30, font_size=8.6)
    callout(c, 62, 65, 720, 52, "决策规则", "不能用跨场景平均分直接选冠军：SA-v2 的圆形低 ATE 被离群点污染；T-pose 的低 ATE 伴随更差局部 RPE。", PALE_ORANGE, ORANGE, body_size=9.5)
    end_page(c)

    # 14 outlier root cause
    begin_page(c, "SA-v2 61.67 s 跳变：测量不大，轨迹却被重写", "Root-cause evidence", 14)
    draw_image_fit(
        c,
        ROOT / "analysis_circle_sa_v2_outlier_6167_20260717" / "sa_v2_late_outlier_diagnostic.png",
        38,
        235,
        455,
        278,
    )
    draw_image_fit(
        c,
        ROOT / "analysis_circle_sa_v2_prior_rank_aware_20260717" / "prior_condition_and_common_update.png",
        512,
        235,
        291,
        278,
    )
    callout(c, 38, 65, 242, 140, "触发条件", "1799 条有效边中，600 条 P_unique 为 rank 6/9。硬 eigen floor 把零噪声方向错误地变成近乎无限信息。", PALE_RED, RED, body_size=9.5)
    callout(c, 300, 65, 242, 140, "放大与出口", "continuous prior 累积共同模式；下一 edge 同时回写旧 state_i。内部相对平移 0.001588 m，却产生 0.466748 m 保存轨迹单步。", PALE_ORANGE, ORANGE, body_size=9.5)
    callout(c, 562, 65, 241, 140, "修复边界", "rank-aware 将晚段 max step 降到 0.0130 m，但 ATE 恶化到 2.3422 m。它是防跳保护，不是最终相关高斯解。", PALE_TEAL, TEAL, body_size=9.5)
    end_page(c)

    # 15 conclusions and next steps
    begin_page(c, "证据分级与下一阶段", "Decision", 15)
    callout(c, 38, 331, 235, 183, "已证明", "IMU Jacobian 正确；标准预积分闭合；MACVIO/GTSAM Delta/P 基本一致；Sampling-aware 统计正确；U1 优于 T-pose 压缩；ba 与 SA-v2 prior 根因已由消融验证。", PALE_TEAL, TEAL, body_size=10)
    callout(c, 303, 331, 235, 183, "合理但未完全证明", "T-pose 压缩丢失点级结构；N=2 放大 Bias/速度耦合；rank-aware 比硬 floor 更安全，但会损失相关信息。", PALE_BLUE, BLUE, body_size=10)
    callout(c, 568, 331, 235, 183, "尚未验证", "共享或低频 ba 的全场景收益；奇异高斯 SA-v2；N=2/5/10 全量成本收益；基于 T_ij 的 stochastic-clone ESKF。", PALE_ORANGE, ORANGE, body_size=10)
    c.setFillColor(NAVY)
    c.roundRect(38, 70, 765, 225, 8, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont(FONT, 15)
    c.drawString(58, 264, "建议执行顺序")
    draw_bullets(
        c,
        [
            "冻结 T-pose（低计算量）与 U1（点级精度）两个长期基线。",
            "只改 ba 参数化：窗口共享、分段常量或低频更新；不改 sigma、视觉因子和 LM。",
            "实现 range/null-space 奇异相关高斯 SA-v2，并保留 Schur 二次型等价断言。",
            "Bias 活跃度回到 RW 同量级后，再进行 N=2/5/10 fixed-lag 对照。",
            "并行建立 T_ij stochastic-clone ESKF 基线，避免因子图后串 EKF 造成重复计数。",
        ],
        58,
        235,
        720,
        10.5,
        6,
        bullet_color=HexColor("#60A5FA"),
        text_color=HexColor("#E2E8F0"),
    )
    end_page(c)

    # 16 appendix
    begin_page(c, "复现入口与组会讨论问题", "Appendix", 16)
    c.setFillColor(INK)
    c.setFont(FONT, 14)
    c.drawString(40, 500, "主要产物")
    draw_bullets(
        c,
        [
            "三场景交互轨迹：analysis_normal_noise_u1_sa_v2_full_three_scenes_20260718/interactive_all_methods_three_scenes.html",
            "跨场景指标：analysis_normal_noise_u1_sa_v2_full_three_scenes_20260718/trajectory_metrics.csv",
            "Sampling/Bias/Window：analysis_normal_noise_sampling_aware_20260716/macvio_sampling_aware_covariance_report_cn.md",
            "SA-v2 跳变：analysis_circle_sa_v2_prior_rank_aware_20260717/sa_v2_prior_rank_aware_root_cause_report_cn.md",
            "完整文字版：analysis_weekly_group_report_20260718/macvo_vio_weekly_group_report_cn.md",
        ],
        42,
        466,
        745,
        9.5,
        8,
        bullet_color=BLUE,
    )
    c.setFillColor(INK)
    c.setFont(FONT, 14)
    c.drawString(40, 304, "建议组会上重点讨论")
    questions = [
        "1. 当前研究主线应优先共享/低频 ba，还是优先完成奇异相关高斯 SA-v2？",
        "2. U1 的点级精度收益是否值得当前约 58 分钟/圆形序列的计算成本？",
        "3. 实时方案更适合 N=2 factor graph，还是 T_ij stochastic-clone ESKF？",
        "4. 生产验收应如何权衡 ATE、RPE、高频误差与 max-step safety gate？",
    ]
    y = 270
    for question in questions:
        callout(c, 42, y - 48, 745, 48, "", question, LIGHT, PURPLE, title_size=1, body_size=10)
        y -= 56
    c.setFillColor(NAVY)
    c.setFont(FONT, 14)
    c.drawCentredString(PAGE_W / 2, 48, "目标不是让轨迹更好看，而是让每一次改善都能解释、复现并推广。")
    end_page(c)

    c.save()
    return OUT


def main() -> None:
    register_fonts()
    output = build()
    print(output)


if __name__ == "__main__":
    main()
