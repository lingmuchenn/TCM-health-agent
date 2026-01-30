from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import streamlit as st
from openai import OpenAI


Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Question:
    qid: str
    title: str
    quick_options: list[str]
    detail_placeholder: str = ""


BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"


ANALYZE_NOW = "基于当前信息直接分析"

Q2_COLD_HOT = Question(
    qid="q2",
    title="下面的问题用于进一步判断你的整体体质情况。日常更怕冷还是怕热？",
    quick_options=["明显怕冷（喜热饮、手脚凉）", "明显怕热（喜冷饮、易上火）", "冷热都不明显", ANALYZE_NOW],
    detail_placeholder="冬天怕冷夏天怕热",
)
Q3_FATIGUE = Question(
    qid="q3",
    title="日常是否容易感到疲劳乏力？",
    quick_options=["经常乏力", "几乎不乏力", "偶尔乏力（熬夜/劳累后）", ANALYZE_NOW],
    detail_placeholder="晨起乏力",
)
Q4_STOOL = Question(
    qid="q4",
    title="大便性状更接近哪种？",
    quick_options=["稀溏不成形（或黏马桶）", "干结难解（或排便费力）", "软硬适中（排便顺畅）", ANALYZE_NOW],
    detail_placeholder="每天腹泻 1 次",
)
Q5_SLEEP = Question(
    qid="q5",
    title="睡眠情况更接近哪种？",
    quick_options=["入睡困难", "容易惊醒", "睡眠质量好", ANALYZE_NOW],
    detail_placeholder="总是睡不醒",
)
Q6_MOOD = Question(
    qid="q6",
    title="情绪状态更接近哪种？",
    quick_options=["经常烦躁焦虑", "经常低落压抑", "情绪平稳", ANALYZE_NOW],
    detail_placeholder="工作压力大时烦躁",
)

QUESTIONS: list[Question] = [Q2_COLD_HOT, Q3_FATIGUE, Q4_STOOL, Q5_SLEEP, Q6_MOOD]


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("stage", 0)  # 0=主诉, 1-5=问题2-6, 6=问题7(可选)+生成, 7=分析后自由对话
    st.session_state.setdefault("asked", set())
    st.session_state.setdefault("q1_main", "")
    st.session_state.setdefault("q7_extra", "")
    st.session_state.setdefault("api_key", "")
    # answers[qid] = str
    st.session_state.setdefault("answers", {})
    st.session_state.setdefault("generated", False)
    st.session_state.setdefault("final_output", "")
    st.session_state.setdefault("followup_messages", [])  # only used after analysis


def reset_state() -> None:
    for k in [
        "messages",
        "stage",
        "asked",
        "q1_main",
        "q7_extra",
        "api_key",
        "answers",
        "generated",
        "final_output",
        "followup_messages",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    init_state()


def append_message(role: Role, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def render_history() -> None:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def detect_red_flags(text: str) -> list[str]:
    red_flags = [
        "胸痛",
        "呼吸困难",
        "持续高烧",
        "抽搐",
        "昏厥",
        "意识不清",
        "呕血",
        "便血",
        "黑便",
        "剧烈腹痛",
        "剧烈头痛",
        "偏瘫",
        "口眼歪斜",
        "自杀",
        "自残",
    ]
    hits = [w for w in red_flags if w in text]
    return hits


def format_answer(qid: str) -> str:
    return (st.session_state.answers.get(qid) or "").strip() or "未填写"


def build_structured_summary(
    *,
    age: int | None,
    gender: str,
    menses: str,
    q1_main: str,
    q7_extra: str,
) -> str:
    parts: list[str] = []
    parts.append("【基础信息】")
    parts.append(f"- 年龄：{age if age else '未填写'}")
    parts.append(f"- 性别：{gender}")
    if gender == "女":
        parts.append(f"- 经期情况：{menses or '未填写'}")
    parts.append("")
    parts.append("【用户主诉】")
    parts.append(f"- 主诉（问题1）：{q1_main or '未填写'}")
    parts.append(f"- 补充说明（问题7）：{q7_extra or '未填写'}")
    parts.append("")
    parts.append("【体质相关回答】")
    parts.append(f"- 寒热感受（问题2）：{format_answer('q2')}")
    parts.append(f"- 乏力情况（问题3）：{format_answer('q3')}")
    parts.append(f"- 大便情况（问题4）：{format_answer('q4')}")
    parts.append("")
    parts.append("【局部症状】")
    parts.append(f"- 睡眠情况（问题5）：{format_answer('q5')}")
    parts.append(f"- 情绪状态（问题6）：{format_answer('q6')}")
    return "\n".join(parts)


SYSTEM_PROMPT = """
【角色】
- 你是「中医养生智能小助手」。
- 你的任务是：基于用户主诉、结构化问答与补充说明，从中医养生视角进行状态分析，并给出可执行的健康调理建议。

【能力边界】
- 你不是医生，不进行疾病诊断，不下医疗结论，不开处方，不推荐处方药或具体药物剂量。
- 分析以症状与状态为导向，聚焦体质倾向与生活方式影响，而非疾病名称。
- 所有结论需基于用户已提供的信息，避免过度推断。

【表达与风格要求】
- 使用用户能理解的日常语言；必要时可使用中医术语，但需用括号进行简要解释。
- 表达必须稳健，使用“不排除 / 可能 / 倾向于”等措辞，体现不确定性。
- 不渲染焦虑，不夸大风险。

【分析流程】
- 先整合信息，再进行分析，再给建议，不要跳步。
- 若用户选择“基于当前信息直接分析”或跳过部分问题，仅在关键信息缺失影响判断时，才说明信息有限，并指出1–2 个最关键的补充方向。
- 若用户在分析之后，对结果进行追问，只基于已有分析进行解释或总结，不重新问诊、不重复收集信息。

【安全与风险提示】
- 若用户描述中出现以下情况之一：症状明显加重、持续进展、剧烈疼痛、异常出血、高热、意识异常等，必须优先提示线下就医，并说明本助手的能力边界。

【首次分析的输出格式】
## 信息/症状摘要
- （整合用户主诉、关键选择与补充说明，用中性语言复述）

## 状态分析
- 体质/状态倾向：…（主倾向 + 是否存在夹杂）
- 形成判断的依据：…（症状或生活因素 → 中医解释）
- 当前状态特点：…（用户可能感受到的典型表现）

## 养生建议
- 作息：
- 饮食：
- 穴位按摩：
- 运动：
- 情志：


结尾固定追加免责声明：
以上建议仅供养生与健康管理参考，不构成医疗诊断或处方。"""


def stream_chat_completion(client: OpenAI, messages: list[dict[str, str]]) -> Iterable[str]:
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=0.3,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        token = getattr(delta, "content", None)
        if token:
            yield token


def get_deepseek_api_key() -> str:
    """请输入你的DEEPSEEK API密钥"""
    key = (st.session_state.get("api_key") or "").strip()
    if key:
        return key
    try:
        return (st.secrets.get("DEEPSEEK_API_KEY", "") or "").strip()
    except Exception:
        return ""


def ensure_question_asked(q: Question) -> None:
    if q.qid in st.session_state.asked:
        return False
    append_message("assistant", f"**{q.title}**")
    st.session_state.asked.add(q.qid)
    return True


st.set_page_config(page_title="中医智能小助手", page_icon="🌿", layout="centered")
init_state()

st.title("🌿 中医智能小助手")
st.caption("本产品仅为 AI 技术演示，内容仅供参考，不能替代专业医疗诊断。")

st.markdown(
    """
<style>
/* 让四个选项按钮高度一致，避免抖动 */
div.stButton > button {
  height: 3.1rem;
  padding-top: 0.35rem;
  padding-bottom: 0.35rem;
  white-space: normal;
}
</style>
""",
    unsafe_allow_html=True,
)

if st.session_state.stage == 0:
    with st.container(border=True):
        st.markdown("**你好，我是你的中医智能小助手。** 我会通过简单问诊了解你的健康状况，并从中医视角做整体分析，给出更贴近日常的养生建议。"
        )
        st.markdown("**操作流程**：**01** 填写基础信息 → **02** 描述症状 → **03** 诊疗建议")

with st.sidebar:
    st.subheader("模型配置")
    st.text_input(
        "",
        type="password",
        placeholder="请输入DEEPSEEK API密钥",
        key="api_key",
        label_visibility="collapsed",
    )

    st.subheader("基础信息")
    age = st.number_input(
        "年龄",
        min_value=0,
        max_value=120,
        value=st.session_state.get("age", 0),
        step=1,
        help="可不填；填写会让建议更贴合",
        key="age",
    )
    gender = st.selectbox("性别", options=["男", "女"], index=0, key="gender")
    menses = ""
    if gender == "女":
        menses = st.selectbox(
            "经期情况（选填）",
            options=["", "规律", "不规律", "痛经明显", "经量偏少/色淡", "经量偏多/色深", "不确定"],
            index=0,
            key="menses",
        )

    if st.button("重新分析（清空聊天）", type="secondary", use_container_width=True, key="reset_sidebar"):
        reset_state()
        st.rerun()

    st.divider()
    st.markdown(
        "**免责声明**：本产品仅为 AI 技术演示，内容仅供参考，不能替代专业医疗诊断。"
    )


def build_followup_model_messages(*, summary: str, analysis_text: str) -> list[dict[str, str]]:
    context = (
        "你正在继续与用户对话。以下是该用户的首次问诊信息摘要与此前你给出的分析。\n\n"
        f"{summary}\n\n"
        "【此前分析】\n"
        f"{analysis_text}\n\n"
        "请在此基础上回答用户后续问题，仍需遵守不诊断、不处方的安全边界。"
    )
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    for m in st.session_state.followup_messages:
        msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


def run_followup_query(*, user_text: str, age_val: int, gender_val: str, menses_val: str) -> None:
    """分析完成后，直接与 LLM 自由对话（会带上首次摘要+首次分析作为上下文）。"""
    append_message("user", user_text)
    st.session_state.followup_messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    api_key = get_deepseek_api_key()
    if not api_key:
        append_message("assistant", "未检测到 DeepSeek API Key：请在左侧栏输入，或在 `.streamlit/secrets.toml` 配置。")
        st.rerun()

    summary = build_structured_summary(
        age=int(age_val) if age_val else None,
        gender=gender_val,
        menses=menses_val,
        q1_main=st.session_state.q1_main,
        q7_extra=st.session_state.q7_extra,
    )
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    model_messages = build_followup_model_messages(summary=summary, analysis_text=st.session_state.final_output)
    with st.chat_message("assistant"):
        output = st.write_stream(stream_chat_completion(client, model_messages))
    append_message("assistant", output)
    st.session_state.followup_messages.append({"role": "assistant", "content": output})
    st.rerun()


# 进入某一阶段时，先把“问题”以 assistant 气泡抛出（只抛一次）
appended_prompt = False
if st.session_state.stage == 0 and "q1" not in st.session_state.asked:
    append_message(
        "assistant",
        "**请先简单描述你最主要的不适症状**",
    )
    st.session_state.asked.add("q1")
    appended_prompt = True

if st.session_state.stage in (1, 2, 3, 4, 5):
    q = QUESTIONS[st.session_state.stage - 1]
    appended_prompt = ensure_question_asked(q) or appended_prompt

if st.session_state.stage == 6 and not st.session_state.generated and "q7" not in st.session_state.asked:
    append_message(
        "assistant",
        "**补充说明（可选）**\n\n是否还有你觉得重要、但前面没问到的情况？如：饮食习惯、作息变化、近期情绪事件等",
    )
    st.session_state.asked.add("q7")
    appended_prompt = True

if st.session_state.stage >= 7 and st.session_state.generated and "postchat" not in st.session_state.asked:
    append_message("assistant", "分析已完成。你可以继续提问，我会结合前面的信息回答。")
    st.session_state.asked.add("postchat")
    appended_prompt = True

if appended_prompt:
    # 先写入问题气泡，再 rerun，让“问题”不会延迟到下一题才出现
    st.rerun()

render_history()

# 分析完成后：提供常见问题引导继续对话
preset_q: str | None = None
if st.session_state.stage >= 7 and st.session_state.generated:
    with st.container(border=True):
        st.markdown("**你可能还想问：**")
        c1, c2, c3 = st.columns(3)
        if c1.button("我更接近哪一种体质", use_container_width=True, key="faq_q1"):
            preset_q = "结合我前面提供的信息，我更接近哪一种体质？请说明理由。"
        if c2.button("你是根据哪些表现，判断我有这些问题的？", use_container_width=True, key="faq_q2"):
            preset_q = "结合我前面提供的信息，你是根据哪些表现，判断我有这些问题的？"
        if c3.button("如果只做一两件事，最重要建议是什么？", use_container_width=True, key="faq_q3"):
            preset_q = "结合我前面提供的信息，如果只做一两件事，最重要建议是什么？"

if preset_q:
    run_followup_query(user_text=preset_q, age_val=int(age), gender_val=gender, menses_val=menses or "")

# 问题阶段：展示快捷选项（仍是同一页聊天气泡样式，输入框固定在底部）
if st.session_state.stage in (1, 2, 3, 4, 5):
    q = QUESTIONS[st.session_state.stage - 1]
    with st.container(border=True):
        cols = st.columns(4)
        for i, opt in enumerate(q.quick_options[:4]):
            if cols[i].button(opt, key=f"{q.qid}_opt_{i}", use_container_width=True):
                st.session_state.answers[q.qid] = opt
                append_message("user", opt)
                if opt == ANALYZE_NOW:
                    st.session_state.stage = 6
                else:
                    st.session_state.stage += 1
                    if st.session_state.stage > 5:
                        st.session_state.stage = 6
                st.rerun()

# 生成分析按钮（阶段6才显示；分析完成后不再显示）
if st.session_state.stage == 6 and not st.session_state.generated:
    api_key = get_deepseek_api_key()
    if not api_key:
        st.warning("未检测到 DeepSeek API Key。请在左侧栏输入，或在 `.streamlit/secrets.toml` 中配置：`DEEPSEEK_API_KEY=\"你的key\"`。")

    col_a, col_b = st.columns([1, 1])
    start_clicked = col_a.button("开始分析", type="primary", use_container_width=True, disabled=not bool(api_key), key="start_analysis")
    if start_clicked and api_key:
        summary = build_structured_summary(
            age=int(age) if age else None,
            gender=gender,
            menses=menses,
            q1_main=st.session_state.q1_main,
            q7_extra=st.session_state.q7_extra,
        )

        all_text = "\n".join(
            [
                st.session_state.q1_main,
                st.session_state.q7_extra,
                format_answer("q2"),
                format_answer("q3"),
                format_answer("q4"),
                format_answer("q5"),
                format_answer("q6"),
                menses or "",
            ]
        )
        hits = detect_red_flags(all_text)
        if hits:
            append_message(
                "assistant",
                f"我注意到你提到了一些可能的危险信号关键词：{', '.join(hits)}。如症状明显/加重，请优先及时线下就医或急救。",
            )

        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        model_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "请基于以下信息生成分析与建议：\n\n" + summary},
        ]
        with st.chat_message("assistant"):
            output = st.write_stream(stream_chat_completion(client, model_messages))
        append_message("assistant", output)
        st.session_state.final_output = output
        st.session_state.generated = True
        st.session_state.stage = 7
        st.session_state.followup_messages = []
        st.rerun()

    if col_b.button("重新分析（清空聊天）", use_container_width=True, key="reset_stage6"):
        reset_state()
        st.rerun()

# 底部固定输入框：同一个 chat_input，根据阶段决定语义
placeholder = "请输入…"
if st.session_state.stage == 0:
    placeholder = "如：最近总是疲劳、胃口差，饭后腹胀，睡眠也不好"
elif st.session_state.stage in (1, 2, 3, 4, 5):
    q = QUESTIONS[st.session_state.stage - 1]
    placeholder = f"若无合适选项，您也可以直接输入（如：{q.detail_placeholder}）"
elif st.session_state.stage == 6 and not st.session_state.generated:
    placeholder = "若无合适选项，您也可以直接输入"
elif st.session_state.stage >= 7 and st.session_state.generated:
    placeholder = "继续提问"

user_text = st.chat_input(placeholder)
if user_text:
    user_text = user_text.strip()
    if not user_text:
        st.stop()

    # 主诉
    if st.session_state.stage == 0:
        st.session_state.q1_main = user_text
        append_message("user", user_text)
        st.session_state.stage = 1
        st.rerun()

    # 问题2-6：直接输入作为该题回答
    if st.session_state.stage in (1, 2, 3, 4, 5):
        q = QUESTIONS[st.session_state.stage - 1]
        st.session_state.answers[q.qid] = user_text
        append_message("user", user_text)
        st.session_state.stage += 1
        if st.session_state.stage > 5:
            st.session_state.stage = 6
        st.rerun()

    # 问题7：补充说明（可选）
    if st.session_state.stage == 6 and not st.session_state.generated:
        st.session_state.q7_extra = user_text
        append_message("user", user_text)
        st.rerun()

    # 分析后自由对话：直接与 LLM 对话
    if st.session_state.stage >= 7 and st.session_state.generated:
        run_followup_query(user_text=user_text, age_val=int(age), gender_val=gender, menses_val=menses or "")

