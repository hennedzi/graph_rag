"""
Рендеринг ответа: clean_prose, render, lines_to_claims, fmt.
"""
import re
from .state import handle
import core


def clean_prose(text: str) -> str:
    """Убираем markdown-разметку и заголовки: ответ — обычный текст."""
    text = re.sub(r"[*`]+", "", text or "")
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"\[([pPПп])(\d+)\]", r"[P\2]", text)  # [п3] [П3] [p3] → [P3]: кириллическая П → латинская P, только в хэндлах
    return text.strip()


def render(st, args: dict) -> tuple[str, list[dict], list[str], list[str]]:
    """
    answer: связный текст БЕЗ инлайн-хэндлов [P#]. Источники передаются отдельно в args["sources"] (список хэндлов ["P1", "P2"]).
    Совместимость: старый формат {"claims": [...]} и текст с [P#] тоже принимаются.
    Возвращает: (text, claims, errors, citations)
    """
    text = str(args.get("text") or "").strip()
    
    # Явно переданные источники (новый формат)
    explicit_sources = args.get("sources") or []
    if isinstance(explicit_sources, str):
        explicit_sources = [explicit_sources]
    explicit_sources = [h.upper() for h in explicit_sources]
    
    # Совместимость: claims → текст + источники
    if not text and isinstance(args.get("claims"), list):
        parts = []
        for c in args["claims"]:
            c = c if isinstance(c, dict) else {"text": str(c)}
            fact = re.sub(r"[*`]+", "", str(c.get("text", ""))).strip()
            srcs = list(dict.fromkeys(str(x).strip("[] ").upper() for x in (c.get("sources") or [])))
            if fact:
                parts.append(f"{fact} " + " ".join(f"[{x}]" for x in srcs))
        text = " ".join(parts).strip()
        # Собираем источники из claims
        for c in args["claims"]:
            if isinstance(c, dict):
                explicit_sources.extend(str(x).strip("[] ").upper() for x in (c.get("sources") or []))
    
    # reply не верифицируется
    if (reply := str(args.get("reply") or "").strip()) and not text:
        # Если в reply есть [P#] — это факт, обрабатываем как текст
        if re.search(r"\[P\d+\]", reply):
            text = reply
        elif is_question(st["query"]) and not META_Q.search(st["query"].lower()):
            if args.get("no_answer"):
                return core.NO_CONTEXT, [], [], []
            q = st["query"].lower()
            if "документ" in q or "список" in q:
                return reply, [], [], []
            return reply, [], ["это вопрос к базе: reply нельзя — дай text с sources или no_answer=true"], []
        elif len(reply) > 600:
            return reply, [], ["reply — короткая реплика без фактов; факты — только через text с sources"], []
        else:
            return reply, [], [], []
    
    if args.get("no_answer") and not text:
        return core.NO_CONTEXT, [], [], []
    if not text:
        return "", [], ["пустой ответ: передай text"], []
    
    text = clean_prose(text)
    
    # Извлекаем источники из текста (обратная совместимость: [P#] в тексте)
    cited_from_text = list(dict.fromkeys(re.findall(r"\[(P\d+)\]", text)))
    
    # Объединяем явные источники и из текста
    all_sources = list(dict.fromkeys(explicit_sources + cited_from_text))
    
    # Валидация источников
    unknown = [h for h in all_sources if h not in st["pid"]]
    unread = [h for h in all_sources if h in st["pid"] and h not in st["read"]]
    errors = []
    if unknown:
        errors.append(f"неизвестные хэндлы {' '.join(unknown)}")
    if unread:
        errors.append(f"источники не прочитаны: {' '.join(unread)} — вызови read_passage и отвечай по полному тексту")
    
    # claims для верификатора: из текста по [P#] (если есть) или пустые с явными источниками
    if cited_from_text:
        claims = lines_to_claims(st, text)
    else:
        # Нет инлайн-ссылок — создаём claims из явных источников
        claims = [{"text": text, "sources": all_sources}] if all_sources else []
    
    if errors:
        return text, claims, errors, all_sources
    return text, claims, [], all_sources


def lines_to_claims(st, text: str) -> list[dict]:
    """Ответ текстом → claims: строка без известного хэндла [P#] — не факт из источника, отбрасываем."""
    out = []
    for line in text.splitlines():
        srcs = [h for h in dict.fromkeys(re.findall(r"\[(P\d+)\]", line)) if h in st["pid"]]
        fact = re.sub(r"\s*\[P\d+\]", "", re.sub(r"[*`]+", "", line)).strip().lstrip("-•#0123456789.) ").strip()
        if srcs and fact:
            out.append({"text": fact, "sources": srcs})
    return out


def fmt(claims: list[dict]) -> str:
    return "\n".join(f"{c['text']} " + " ".join(f"[{x}]" for x in c["sources"]) for c in claims)


# Реплики без поиска
from .prompts import CHAT_WORDS, CHAT_REPLY
META_Q = re.compile(r"\b(?:кто ты|что ты (?:умеешь|можешь|знаешь|умеешь делать)|твои возможности|"
                    r"что ты такое|расскажи о себе)\b")
REWRITE = re.compile(r"\b(?:попроще|проще|покороче|короче|кратко|кратче|вкратце|в двух словах|своими словами|понятнее|"
                     r"объясни|перескажи|пересказать|простыми словами|простой язык)\b")


def is_question(text: str) -> bool:
    text = text.strip().lower()
    return text.endswith("?") or any(text.startswith(w) for w in ("кто", "что", "где", "когда", "почему", "зачем", "как", "какой", "сколько", "чего", "чей", "чей"))


def chat_reply(query: str) -> str | None:
    q = query.lower().strip()
    for cat, words in CHAT_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", q) for w in words):
            return CHAT_REPLY[cat]
    return None


def route(query: str, has_prev: bool) -> str:
    """chat | rewrite | agent"""
    q = query.strip()
    if not q:
        return "chat"
    if chat_reply(q):
        return "chat"
    if REWRITE.search(q.lower()):
        return "rewrite" if has_prev else "agent"
    if META_Q.search(q.lower()):
        return "chat"
    return "agent"