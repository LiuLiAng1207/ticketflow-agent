from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT.parent
ASSET_DIR = PROJECT_ROOT / "reports"
OUTPUT_PDF = BASE_DIR / "TicketFlow项目Demo说明.pdf"
ASCII_OUTPUT_PDF = BASE_DIR / "TicketFlow_Demo_Submission.pdf"


pdfmetrics.registerFont(TTFont("MSYH", r"C:\Windows\Fonts\msyh.ttc"))
pdfmetrics.registerFont(TTFont("MSYH-Bold", r"C:\Windows\Fonts\msyhbd.ttc"))

PAGE_SIZE = landscape(A4)
PAGE_WIDTH, PAGE_HEIGHT = PAGE_SIZE


def style(name: str, size: float, color: str = "#182235", bold: bool = False, leading: float | None = None, space_after: float = 4):
    return ParagraphStyle(
        name=name,
        fontName="MSYH-Bold" if bold else "MSYH",
        fontSize=size,
        leading=leading or size * 1.35,
        textColor=colors.HexColor(color),
        spaceAfter=space_after,
    )


STYLES = {
    "title": style("title", 26, "#0f172a", True, 34, 8),
    "subtitle": style("subtitle", 11, "#475569", False, 16, 8),
    "h1": style("h1", 17, "#0f172a", True, 23, 8),
    "h2": style("h2", 12, "#0f172a", True, 17, 5),
    "body": style("body", 9.4, "#334155", False, 14, 4),
    "small": style("small", 8.2, "#64748b", False, 11.5, 3),
}

ACCENT = colors.HexColor("#2563eb")
LINE = colors.HexColor("#dbe4f0")
MUTED = colors.HexColor("#64748b")
SOFT_BLUE = colors.HexColor("#eff6ff")
SOFT_GREEN = colors.HexColor("#ecfdf5")
SOFT_ORANGE = colors.HexColor("#fff7ed")


def card(flowables, width: float, bg=colors.white, pad: float = 9) -> Table:
    table = Table([[flowables]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), pad),
                ("RIGHTPADDING", (0, 0), (-1, -1), pad),
                ("TOPPADDING", (0, 0), (-1, -1), pad),
                ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
            ]
        )
    )
    return table


def bullets(items: list[str]):
    return [Paragraph("• " + item, STYLES["body"]) for item in items]


def image_flow(path: Path, max_width: float, max_height: float) -> Image:
    image = PILImage.open(path)
    width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def flow_diagram() -> Drawing:
    drawing = Drawing(730, 120)
    labels = ["工单输入", "附件解析", "分诊", "RAG证据", "充分性判断", "动作路由", "工具审批", "执行/回复", "事实校验"]
    x = 10
    y = 55
    box_width = 72
    gap = 9
    for index, label in enumerate(labels):
        drawing.add(
            Rect(
                x,
                y,
                box_width,
                28,
                rx=8,
                ry=8,
                fillColor=colors.HexColor("#f8fafc"),
                strokeColor=colors.HexColor("#93c5fd"),
            )
        )
        drawing.add(
            String(
                x + box_width / 2,
                y + 10,
                label,
                fontName="MSYH",
                fontSize=8,
                textAnchor="middle",
                fillColor=colors.HexColor("#1e293b"),
            )
        )
        if index < len(labels) - 1:
            drawing.add(Line(x + box_width, y + 14, x + box_width + gap, y + 14, strokeColor=ACCENT, strokeWidth=1.2))
        x += box_width + gap
    drawing.add(
        String(
            10,
            20,
            "核心原则：模型负责理解、检索和生成建议；治理链负责证据充分性、工具权限、人工审批和事实校验。",
            fontName="MSYH",
            fontSize=9,
            fillColor=colors.HexColor("#475569"),
        )
    )
    return drawing


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("MSYH", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(16 * mm, 9 * mm, "TicketFlow 项目 Demo 说明｜GitHub: LiuLiAng1207/ticketflow-agent")
    canvas.drawRightString(PAGE_WIDTH - 16 * mm, 9 * mm, str(doc.page))
    canvas.restoreState()


def build_story():
    story = []
    left_width = 330
    right_width = 390

    story.append(
        Table(
            [
                [
                    [
                        Paragraph("TicketFlow 工单协同智能体", STYLES["title"]),
                        Paragraph("面向客服与 IT 服务台场景的企业级工单处理与运营 Agent Demo", STYLES["subtitle"]),
                        Paragraph(
                            'GitHub 仓库：<font color="#2563eb">https://github.com/LiuLiAng1207/ticketflow-agent</font>',
                            STYLES["body"],
                        ),
                        Spacer(1, 8),
                        card(
                            [
                                Paragraph("Demo 交付说明", STYLES["h2"]),
                                *bullets(
                                    [
                                        "这是比赛提交用单文件说明 PDF，包含项目定位、运行入口、架构、核心功能和真实界面截图。",
                                        "完整源码请以 GitHub 仓库为准；本 PDF 可直接上传到只支持单文件提交的平台。",
                                        "本项目是受控型 Agent Workflow：强调证据、审批、审计和事实校验，而不是让模型自由执行高风险动作。",
                                    ]
                                ),
                            ],
                            left_width,
                            bg=SOFT_BLUE,
                        ),
                        Spacer(1, 9),
                        card(
                            [
                                Paragraph("运行方式", STYLES["h2"]),
                                *bullets(
                                    [
                                        "本地启动：进入项目目录后运行 start_ticketflow_demo.bat。",
                                        "手动启动：pip install -e . 后运行 streamlit run src/ticketflow/app.py。",
                                        "默认前端地址：http://127.0.0.1:8501/。",
                                    ]
                                ),
                            ],
                            left_width,
                            bg=SOFT_GREEN,
                        ),
                    ],
                    [
                        image_flow(ASSET_DIR / "ticketflow_home.png", right_width, 260),
                        Spacer(1, 6),
                        Paragraph("图 1：企业运维台首页，包含模型后端、RAG、外部邮箱、治理链路和专项评测看板。", STYLES["small"]),
                    ],
                ]
            ],
            colWidths=[left_width, right_width],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("系统架构与处理链路", STYLES["h1"]))
    story.append(flow_diagram())
    story.append(Spacer(1, 8))
    columns = [
        [
            Paragraph("1. 受控型 Agent Workflow", STYLES["h2"]),
            *bullets(["基于 LangGraph 组织状态化工单流程。", "每个节点把输入、输出、审计事件写入共享状态。", "高风险动作不由模型直接执行，而是进入工具级审批。"]),
        ],
        [
            Paragraph("2. 证据增强与治理", STYLES["h2"]),
            *bullets(["融合知识库、策略条款、订单、客户画像、历史工单和附件证据。", "Sufficiency-First：先判断证据是否足够，再决定动作路线。", "回复生成后进入事实校验，避免未执行动作被承诺给用户。"]),
        ],
        [
            Paragraph("3. 对话运营扩展方向", STYLES["h2"]),
            *bullets(["可扩展 TicketOps Agent：查询工单、解释证据链、统计风险工单。", "引入 Agent Harness：任务规范、工具权限、记忆、轨迹、失败归因。", "批量重检索、批量生成回复草稿、日报和知识库候选。"]),
        ],
    ]
    story.append(Table([[card(column, 235) for column in columns]], colWidths=[240, 240, 240]))
    story.append(Spacer(1, 10))
    story.append(
        card(
            [
                Paragraph("核心技术栈", STYLES["h2"]),
                Paragraph("Python、LangGraph、RAG、Chroma、Streamlit、SQLite、MCP、DeepSeek 云端模型、本地 MiniMind 双 LoRA 后端接入、SMTP 邮件工具。", STYLES["body"]),
            ],
            725,
            bg=colors.HexColor("#f8fafc"),
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("核心演示：单票工单处理", STYLES["h1"]))
    story.append(Table([[image_flow(ASSET_DIR / "ticketflow_workflow.png", 705, 325)]], colWidths=[725], style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")])))
    story.append(Spacer(1, 7))
    story.append(
        card(
            [
                Paragraph("演示讲法", STYLES["h2"]),
                *bullets(
                    [
                        "选择退款类工单后，系统先完成分诊，再检索策略、订单和历史证据。",
                        "上下文证据充分性节点判断退款路线是否可进入下一步。",
                        "退款属于财务高风险动作，因此即便证据充分，也必须进入工具级人工审批。",
                        "所有节点都有流转记录、工具日志、审计记录，方便复盘和定位失败原因。",
                    ]
                ),
            ],
            725,
            bg=SOFT_BLUE,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("多模态附件证据展示", STYLES["h1"]))
    story.append(Table([[image_flow(ASSET_DIR / "ticketflow_attachment.png", 705, 330)]], colWidths=[725], style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")])))
    story.append(Spacer(1, 7))
    story.append(
        card(
            [
                Paragraph("说明", STYLES["h2"]),
                *bullets(
                    [
                        "附件区展示原始文件、OCR 文本、视觉摘要和结构化字段。",
                        "附件证据进入 RAG 和证据链，但不会单独放行退款等高风险动作。",
                        "真实业务版本建议通过前端上传真实附件，并保存原始文件路径、哈希、解析结果和审计记录。",
                    ]
                ),
            ],
            725,
            bg=SOFT_ORANGE,
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("功能能力与评测结果", STYLES["h1"]))
    metric_data = [
        ["能力/指标", "结果", "说明"],
        ["检索增强专项评测 HitRate", "99.3%", "A40 环境完成，验证检索是否命中目标证据。"],
        ["检索增强专项评测 MRR", "0.570", "衡量第一个正确证据排得靠不靠前。"],
        ["检索错误率", "0.7%", "检索未命中或错误命中的比例。"],
        ["多模态证据召回增益", "+100.0%", "启用附件证据相对禁用附件证据的召回提升。"],
        ["附件实体抽取增益", "+77.2%", "衡量附件中订单号、金额、错误码等实体抽取收益。"],
        ["启用附件后动作准确率", "84.0%", "多模态评测下动作建议与标注的一致性。"],
    ]
    metric_table = Table(metric_data, colWidths=[185, 90, 430])
    metric_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "MSYH"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("GRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 10))
    story.append(
        Table(
            [
                [
                    card(
                        [
                            Paragraph("比赛提交建议", STYLES["h2"]),
                            *bullets(["如果平台只能交一个文件：提交本 PDF。", "如果平台允许填写链接：填写 GitHub 仓库链接。", "如果平台允许补充说明：附运行命令和截图页码。"]),
                        ],
                        350,
                        bg=SOFT_GREEN,
                    ),
                    card(
                        [
                            Paragraph("GitHub 仓库", STYLES["h2"]),
                            Paragraph('<font color="#2563eb">https://github.com/LiuLiAng1207/ticketflow-agent</font>', STYLES["body"]),
                            Paragraph("建议在仓库 README 中保留：环境安装、启动方式、配置说明和 Demo 截图。", STYLES["small"]),
                        ],
                        350,
                    ),
                ]
            ],
            colWidths=[360, 360],
        )
    )
    return story


def main() -> None:
    doc = SimpleDocTemplate(
        str(ASCII_OUTPUT_PDF),
        pagesize=PAGE_SIZE,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=15 * mm,
    )
    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    OUTPUT_PDF.write_bytes(ASCII_OUTPUT_PDF.read_bytes())
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
