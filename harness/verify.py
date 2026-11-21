"""
Верификация ответа: check, off_topic, resolve, chat_tools, thought.
"""
import openai
import re
from .tools import read_passage
from .state import emit, say, short, LOG_CHARS
from .prompts import TOOLS, TOOL_CHOICE
from .render import route
import core


async def chat_tools(msgs: list[dict], usage: dict | None = None):
    """tool_choice=required, а если сервер его не поддерживает — один раз падаем на auto и запоминаем."""
    try:
        return await core.chat(msgs, usage=usage, tools=TOOLS, tool_choice=TOOL_CHOICE["choice"])
    except openai.BadRequestError as e:
        if TOOL_CHOICE["choice"] == "auto":
            raise
        log.warning("LLM-сервер не принял tool_choice=%s (%s), дальше auto", TOOL_CHOICE["choice"], e)
        TOOL_CHOICE["choice"] = "auto"
        return await core.chat(msgs, usage=usage, tools=TOOLS, tool_choice="auto")


async def check(st, text: str, sources: list[str] | None = None) -> list[dict]:
    """Верификация по процитированным пассажам (подгружаем, даже если агент их не читал).
    None от verify = проверка не выполнена: это не «всё подтверждено» и не «в документах нет»."""
    if sources is None:
        sources = [h for h in st["pid"] if f"[{h}]" in text]
    await read_passage(st, [h for h in sources if h not in st["read"]])
    res = await core.verify(text, {h: st["text"].get(st["pid"][h], "") for h in sources})
    if res is None:
        st["verify_error"] = True
        return []
    return res


RELEVANT_PROMPT = """Вопрос: {q}

Утверждения ответа:
{claims}

Для каждого утверждения реши: относится ли оно к вопросу — отвечает на него, его часть или поясняет ответ.
off_topic=true ТОЛЬКО если утверждение про другую тему: например, на «как работает поиск» — раскладка
интерфейса, порт сервера, поля событий. Сомневаешься — off_topic=false.
Верни СТРОГО JSON-массив: [{{"n": номер, "off_topic": true|false}}]
Только JSON, без пояснений и markdown.

JSON:"""


async def off_topic(st, claims: list[dict]) -> list[dict]:
    """Верификатор проверяет только «написано ли в источнике», поэтому верный, но посторонний факт проходил как ответ."""
    if not claims:
        return []
    listing = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(claims, 1))
    raw = await core.ask_llm(RELEVANT_PROMPT.format(q=st["query"], claims=listing))
    checks = core.parse_json(raw, None)
    if not isinstance(checks, list):  # сломанный JSON не должен выбрасывать ответ
        log.warning("Проверка релевантности вернула невалидный JSON: %s", short(raw))
        return []
    # выбрасываем только явное off_topic=true: ошибка судьи не должна стирать верный ответ
    drop = {int(c["n"]) for c in checks if isinstance(c, dict) and c.get("off_topic") is True and str(c.get("n")).isdigit()}
    if drop:
        say(st.get("sid", ""), "не по вопросу «%s»: %s", st["query"], short(raw))
    return [c for i, c in enumerate(claims, 1) if i in drop]


RESOLVE_PROMPT = """Перепиши реплику пользователя в самостоятельный вопрос к базе знаний.

Прошлые вопросы пользователя (последний внизу):
{questions}

Последний ответ ассистента:
{answer}

Новая реплика: «{q}»

Правила:
- реплика ссылается на прошлое («это», «он», «а где», «что это значит») — подставь, о чём речь;
- реплика — возражение или жалоба на ответ («это неправда», «это не ответ на мой вопрос», «не то») —
  верни дословно тот прошлый вопрос, на который ответ не устроил;
- реплика уточняет предмет («я про X», «имею в виду X») — перепиши ПОСЛЕДНИЙ вопрос с этим предметом
  (например, «что ещё умеет приложение?» + «я про Graph RAG» → «что ещё умеет приложение Graph RAG?»);
- реплика без вопроса (спасибо, оценка, «теперь лучше», согласие) — верни её дословно;
- реплика уже самостоятельный вопрос — верни её без изменений;
- реплика про сам сервис, интерфейс, код или сервер («что тут не так?», «почему не работает поиск?», «это баг?») —
  верни её дословно: в документах ответа на неё нет.
Верни только вопрос одной строкой, без пояснений."""


async def resolve(ses: dict, query: str) -> str:
    """
    «И что это значит», «ЭТО НЕ ПРАВДА» раньше уходили в поиск дословно: сущности и ранжирование связей
    шли по словам «запоминаются», «error», и агент цитировал случайные пассажи. Ищем по полному вопросу.
    """
    if not (past := ses.get("questions")):
        return query
    u: dict = {}
    prompt = RESOLVE_PROMPT.format(questions="\n".join(past[-5:]), answer=ses.get("prev", ""), q=query)
    out = core.strip_think((await core.chat([{"role": "user", "content": prompt}], usage=u)).content)
    ses["st"]["spent"] += u.get("prompt", 0) + u.get("completion", 0)
    line = next((x.strip(" «»\"'") for x in out.splitlines() if x.strip()), "")
    return line if 0 < len(line) <= 500 else query


# ─── Ход мыслей в лог ──────────────────────────────────────────
def thought(msg) -> str:
    """llama.cpp/vLLM отдают рассуждение в reasoning_content; иначе оно внутри </think> в content."""
    r = getattr(msg, "reasoning_content", None) or (msg.model_extra or {}).get("reasoning_content")
    if not r and (m := re.search(r"</think>(.*?)(?:</think>|$)", msg.content or "", re.S)):
        r = m[1]
    return (r or "").strip()


import logging
log = logging.getLogger("graph_rag.harness")