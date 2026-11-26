#!/usr/bin/env python3
"""Web research agent (local, Ollama).

plan (topic + queries) -> search -> fetch -> topic filter -> LLM relevance
scoring + fact extraction -> optional follow-up round -> streamed cited answer.
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import ollama
import requests
import trafilatura
from ddgs import DDGS

HEADERS = {"User-Agent": "Mozilla/5.0 (research-agent)"}
MAX_CHARS = 6000      # per-page cap; keeps prompts inside num_ctx
MIN_SCORE = 6         # drop sources the LLM rates below this (0-10)
MAX_SOURCES = 8       # sources passed to the final answer
PREFERRED = ("wikipedia.org", "britannica.com", ".gov", ".edu", ".ac.in", "nic.in")
# depth -> (queries/round, results/query, max rounds, max pages read/round)
DEPTH = {"Quick": (2, 4, 1, 5), "Standard": (3, 5, 2, 8), "Deep": (4, 6, 3, 12)}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------- LLM helpers ----------
def llm(prompt, model, json_mode=False, temperature=0.2,
        system="You are a precise research assistant."):
    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        format="json" if json_mode else None,
        options={"temperature": temperature, "num_ctx": 8192},
    )
    return resp["message"]["content"].strip()


def parse_json(text, default):
    for cand in (text, (re.search(r"\{.*\}", text, re.S) or [None])[0]):
        if cand:
            try:
                r = json.loads(cand)
                if isinstance(r, dict):
                    return r
            except json.JSONDecodeError:
                pass
    return default


def standalone(question, history, model):
    """Rewrite a follow-up ('what about its history?') into a standalone question."""
    if not history:
        return question
    out = llm(f"Earlier questions: {history[-2:]}\nNew message: \"{question}\"\n"
              "If the new message depends on earlier context (pronouns, 'it', 'what about...'), "
              "rewrite it as a complete standalone question. Otherwise return it unchanged. "
              "Return only the question.", model, temperature=0)
    return out.strip().strip('"') or question


# ---------- Tools ----------
def search(query, k):
    try:
        hits = DDGS().text(query, max_results=k)
    except Exception as e:
        log(f"search failed for '{query}': {e}")
        return []
    return [{"title": h.get("title", ""), "url": h.get("href", ""),
             "snippet": h.get("body", "")} for h in hits if h.get("href")]


def rank(hit):
    return 0 if any(d in hit["url"] for d in PREFERRED) else 1


def fetch(hit):
    try:
        r = requests.get(hit["url"], headers=HEADERS, timeout=12)
        r.raise_for_status()
        if "html" not in r.headers.get("content-type", ""):
            return None
        text = trafilatura.extract(r.text, include_comments=False, include_tables=False)
    except Exception:
        return None
    if not text or len(text) < 300:
        return None
    return {**hit, "text": text[:MAX_CHARS]}


def mentions_topic(text, topic):
    """Cheap pre-filter: page must contain most of the topic's key words."""
    toks = [t for t in re.findall(r"\w+", topic.lower()) if len(t) > 2]
    if not toks:
        return True
    low = text.lower()
    return sum(t in low for t in toks) / len(toks) >= 0.6


# ---------- Agent steps ----------
def plan(question, n, model):
    out = llm(
        f"Today is {date.today()}. Question: \"{question}\"\n"
        "Identify the main subject (specific entity/topic, 1-4 words) and write "
        f"{n} web search queries that retrieve information directly answering the question. "
        "Every query MUST contain the main subject's name. Vary the angle "
        "(overview, history, key facts, recent updates).\n"
        'Return JSON: {"topic": "...", "queries": ["...", "..."]}',
        model, json_mode=True)
    r = parse_json(out, {})
    topic = r.get("topic") if isinstance(r.get("topic"), str) and r["topic"].strip() else question
    gen = [q for q in r.get("queries", [])
           if isinstance(q, str) and q.strip() and q.lower() != question.lower()]
    # user's own wording first, plus a Wikipedia anchor for entity overviews
    return topic, [question] + gen[:max(n - 1, 0)] + [f"{topic} wikipedia"]


def assess(question, src, model):
    out = llm(
        f"Question: {question}\n\nPage title: {src['title']}\nPage content:\n{src['text']}\n\n"
        "Rate how directly this page helps answer the question, 0-10 (0 = unrelated, "
        "10 = directly answers). Pages about a different subject, or that mention the "
        "topic only in passing, must score 3 or lower. Then list 3-6 concise facts from "
        "the page that answer the question (keep names, numbers, dates). Use only the page.\n"
        'Return JSON: {"relevance": <number>, "facts": ["...", "..."]}',
        model, json_mode=True)
    r = parse_json(out, {})
    try:
        score = float(r.get("relevance", 0))
    except (TypeError, ValueError):
        score = 0.0
    facts = [f for f in r.get("facts", []) if isinstance(f, str) and f.strip()]
    return score, facts


def reflect(question, sources, model):
    notes = "\n".join(f"- " + "; ".join(s["facts"])[:500] for s in sources)
    out = llm(
        f"Question: {question}\n\nFacts gathered:\n{notes}\n\n"
        "Is this enough for a complete, well-supported answer? If not, give up to 2 "
        'follow-up search queries for what is missing. Return JSON: '
        '{"sufficient": true|false, "followup_queries": ["..."]}',
        model, json_mode=True)
    r = parse_json(out, {"sufficient": True})
    fq = [q for q in r.get("followup_queries", []) if isinstance(q, str) and q.strip()]
    return bool(r.get("sufficient", True)), fq[:2]


def research(question, model, depth="Standard", log_fn=log):
    """Return a list of relevant sources: {id, title, url, score, facts}."""
    n_q, n_res, rounds, max_pages = DEPTH[depth]
    topic, queries = plan(question, n_q, model)
    kept, seen = [], set()

    for rnd in range(1, rounds + 1):
        log_fn("Searching: " + " · ".join(queries))
        hits = []
        for q in queries:
            for h in search(q, n_res):
                if h["url"] not in seen:
                    seen.add(h["url"])
                    hits.append(h)
        hits.sort(key=rank)  # stable: trusted domains first

        with ThreadPoolExecutor(max_workers=6) as ex:
            pages = [p for p in ex.map(fetch, hits) if p]
        pages = [p for p in pages if mentions_topic(p["text"], topic)][:max_pages]
        log_fn(f"Reading {len(pages)} pages")

        for p in pages:
            score, facts = assess(question, p, model)
            if score >= MIN_SCORE and facts:
                kept.append({**p, "score": score, "facts": facts})
        log_fn(f"{len(kept)} relevant sources so far")

        if rnd == rounds:
            break
        if not kept:
            queries = [topic, f"{topic} facts"]
            continue
        ok, follow = reflect(question, kept, model)
        if ok or not follow:
            break
        queries = [q if topic.lower() in q.lower() else f"{topic} {q}" for q in follow]

    kept.sort(key=lambda s: -s["score"])
    kept = kept[:MAX_SOURCES]
    for i, s in enumerate(kept, 1):
        s["id"] = i
    return kept


def synthesize_stream(question, sources, model):
    """Yield the final answer token by token."""
    material = "\n\n".join(
        f"[{s['id']}] {s['title']}\n" + "\n".join(f"- {f}" for f in s["facts"])
        for s in sources)
    prompt = (
        f"Question: {question}\n\nFacts from sources:\n{material}\n\n"
        "Answer the question directly as a clear, well-organized brief. Start with a "
        "2-3 sentence direct answer, then short sections or bullets with the key details. "
        "Use ONLY the facts above and ignore any fact that is not about the question's "
        "subject. Cite inline as [n] using only the source numbers above. If evidence is "
        "thin or sources conflict, say so in one closing line. No title, and do not "
        "mention 'the sources provided'.")
    stream = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": "You write concise, accurate, well-cited answers."},
                  {"role": "user", "content": prompt}],
        options={"temperature": 0.3, "num_ctx": 8192}, stream=True)
    for chunk in stream:
        yield chunk["message"]["content"]


def sources_md(sources):
    return "\n".join(f"{s['id']}. [{s['title'] or s['url']}]({s['url']})" for s in sources)


def main():
    ap = argparse.ArgumentParser(description="Local web research agent (Ollama)")
    ap.add_argument("question")
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--depth", choices=list(DEPTH), default="Standard")
    ap.add_argument("--out", help="save answer to this .md file")
    a = ap.parse_args()

    sources = research(a.question, a.model, a.depth)
    if not sources:
        print("No relevant sources found. Try rephrasing the question.")
        return
    answer = "".join(synthesize_stream(a.question, sources, a.model))
    report = f"# {a.question}\n\n{answer}\n\n## Sources\n{sources_md(sources)}\n"
    print(report)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()