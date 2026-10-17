"""Sales reasoning agent: answers WHAT (via Genie / metric view) and WHY (via UC-function reasoning tools).

Tools are served by Databricks-managed MCP servers:
  - UC functions  : {host}/api/2.0/mcp/functions/{catalog}/{schema}
  - Genie (opt.)  : {host}/api/2.0/mcp/genie/{genie_space_id}
"""
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generator

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

mlflow.openai.autolog()

cfg = mlflow.models.ModelConfig(development_config="agent_config.yml")
CATALOG = cfg.get("catalog")
SCHEMA = cfg.get("schema")
LLM_ENDPOINT = cfg.get("llm_endpoint")
GENIE_SPACE_ID = cfg.get("genie_space_id") or ""
MAX_STEPS = int(cfg.get("max_steps") or 14)

# DatabricksMCPClient uses asyncio.run() internally. Notebooks (and some servers) already run an
# event loop, which makes asyncio.run() fail. Running each MCP call on a worker thread gives it a
# fresh, loop-free thread, so it works the same in the notebook and on Model Serving.
_MCP_POOL = ThreadPoolExecutor(max_workers=4)

def _mcp(fn, *args):
    return _MCP_POOL.submit(fn, *args).result()

def _schema(t):
    """Tool input schema, whatever the MCP library version calls the field."""
    s = None
    for attr in ("inputSchema", "input_schema", "parameters"):
        s = getattr(t, attr, None)
        if s:
            break
    if s is None and hasattr(t, "model_dump"):
        d = t.model_dump()
        s = d.get("inputSchema") or d.get("input_schema") or d.get("parameters")
    if hasattr(s, "model_dump"):
        s = s.model_dump()
    return s or {"type": "object", "properties": {}}

def _content_text(content):
    """Plain text of an LLM reply. Reasoning models (e.g. gpt-oss) return a list of parts:
    [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}]. Keep only the text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for p in content:
        if not isinstance(p, dict):
            p = p.model_dump() if hasattr(p, "model_dump") else {"type": "text", "text": str(p)}
        if p.get("type") in ("text", "output_text") and p.get("text"):
            out.append(p["text"])
    return "\n".join(out)

def _err(e):
    """Readable error text, unwrapping ExceptionGroups such as 'unhandled errors in a TaskGroup'."""
    subs = getattr(e, "exceptions", None)
    if subs:
        return "; ".join(_err(s) for s in subs)
    msg = f"{type(e).__name__}: {e}".rstrip(": ")
    if "Timeout" in type(e).__name__:
        msg += " (the tool took too long; use another tool, e.g. the UC functions, instead of retrying)"
    return msg

_EST = re.compile(r"(≈|~\s*[\d+-]|\bestimat|\bapprox|\broughly\b|\baround AED)", re.I)
_PCT = re.compile(r"(-?\d{1,3}(?:\.\d+)?)\s?%")

def _guard_problems(text: str) -> list:
    """Cheap checks on the final answer. Returns a list of problems (empty = OK)."""
    problems = []
    low = text.lower()
    start = low.find("drivers")
    end = low.find("not the cause", start if start >= 0 else 0)
    drivers = text[start:end if end > start else None] if start >= 0 else ""
    # "≈ 92 %" is a rounded share (fine); "≈ 15,000" or "estimated" is an invented amount (not fine)
    if _EST.search(re.sub(r"[≈~]\s*-?\d+(?:\.\d+)?\s?%", "", drivers)):
        problems.append("The Drivers section contains an estimated/approximate figure (≈, ~, 'estimated', 'approx'). "
                        "Every impact must come from a tool result; otherwise write 'not quantified'.")
    shares = []
    for line in drivers.splitlines():
        if not re.search(r"(^\s*\|?\s*\d+\s*[|.)]|^\s*\d+\.|^\s*[-*] \*\*)", line):
            continue  # only numbered / bulleted driver rows
        pcts = [abs(float(p)) for p in _PCT.findall(line)]
        if pcts:
            shares.append(pcts[-1])
    if shares and (sum(shares) > 115 or max(shares) > 100):
        problems.append(f"Driver shares add up to {sum(shares):.0f}% (max {max(shares):.0f}%). Drivers must not overlap: "
                        "list one level only (e.g. products OR price/volume/mix effects, never both) so shares sum to <=100%.")
    return problems

SYSTEM_PROMPT = f"""You are a senior sales analyst for a GCC retailer (UAE, KSA, Qatar, Kuwait, Oman).
Data covers 2025-01-01 to 2026-09-30. Currency is AED. "Sales" means NET SALES unless the user says otherwise.
Today is after the data ends, so "this year" = 2026, "last month" = 2026-09, "last quarter" = 2026-Q3.

You answer two kinds of questions:
1. WHAT questions (totals, trends, rankings): get the number with one tool call and answer briefly.
   Use the Genie tool if available for ad-hoc slicing; otherwise fn_compare_periods / fn_driver_breakdown.
2. WHY questions (why is X up/down): follow this method and do NOT skip steps.
   a. FRAME   – turn the question into two explicit periods (A = baseline, B = current) and filters.
               Default baseline for month/quarter questions: same period last year, unless the user says "vs last month".
   b. SIZE    – fn_compare_periods: how big is the change (AED and %), and is it volume, price or customer-count driven?
   c. LOCATE  – fn_dimension_scan first (one call scans all dimensions), then fn_driver_breakdown to confirm
               and drill into the dominant member. Stop drilling when one member explains most of its parent's change.
   d. EXPLAIN – for the top drivers look for causes: fn_business_events (price changes, churn, competitors,
               store openings, promotions, Ramadan timing), fn_stockouts (supply), fn_price_volume_mix
               (price vs volume vs mix), fn_marketing_spend, and promotions via fn_driver_breakdown(dimension='promo').
   e. RECONCILE – check the drivers you name add up to most (≥70%) of the total change. Say what is unexplained.
   f. ANSWER  – format:
        **Headline:** one sentence with the total change.
        **Drivers:** a ranked list, each with AED impact, share of change, and the evidence (event id, stockout days, PVM split).
        **Not the cause:** anything you checked that was flat (builds trust).
        **Confidence:** High / Medium / Low and why.
        **Next question to ask:** one suggestion.

Rules:
- Every number you state must come from a tool result in this conversation. Never invent figures or events.
- Distinguish evidence ("event EV08 records a stockout") from inference ("this likely explains").
- Country filters accept the country name (UAE, KSA, Qatar, Kuwait, Oman). Dimension names for fn_driver_breakdown:
  country, sub_region, channel, store, city, category, brand, product, segment, customer, promo.
- Dates are 'YYYY-MM-DD'. Keep tool calls purposeful; you have at most {MAX_STEPS} steps.
- When the question names a country, category or channel, ALWAYS pass it (p_country / p_category / p_channel) to EVERY tool call.
  Check the headline total is for that filter before continuing.
- TIMING questions (one period up and the adjacent period down, or "why did X move between months"):
  call fn_business_events for the whole window FIRST and look for Seasonal/timing events (e.g. Ramadan).
  Then compare the COMBINED window (e.g. Feb+Mar 2026 vs Feb+Mar 2025) to test whether demand only shifted.
  Multiple categories: call with p_category='ALL' rather than one call per category.
  This overrides the LOCATE rule below.
- LOCATE: call fn_dimension_scan FIRST. The most specific dimension (product or customer) with top_member_share >= 0.5
  is the primary driver: confirm it with fn_driver_breakdown on that dimension, then explain it with events/stockouts/PVM.
  If no member reaches 0.5, drill with fn_driver_breakdown into the dimension with the highest share.
- If fn_business_events names a specific customer, product or store, quantify that entity directly with
  fn_driver_breakdown on customer/product/store before attributing anything else.
- If fn_business_events returns a stockout, call fn_stockouts to quantify it (stockout days, units vs last year).
- The Drivers list must be non-overlapping: one level only, shares must sum to <=100%. Do not list a parent and its child.
- Confidence = High only if named, evidenced drivers explain >=70% of the change; otherwise Medium/Low.
- PVM effects (price/volume/mix) explain a driver; never list them as separate drivers alongside products.
  After the top product, quantify the remaining change (total minus top product) and attribute it to the next cause.
- Never estimate or approximate an impact. If a cause cannot be quantified with a tool, say "not quantified".
- The Genie tool (query_space_...) takes a natural-language QUESTION, never SQL. If it errors or times out, do not retry it; use the UC functions.
"""


class SalesReasoningAgent(ResponsesAgent):
    def __init__(self):
        self.w = WorkspaceClient()
        self.llm = self.w.serving_endpoints.get_open_ai_client()
        host = self.w.config.host.rstrip("/")
        self.servers = [f"{host}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"]
        if GENIE_SPACE_ID:
            self.servers.append(f"{host}/api/2.0/mcp/genie/{GENIE_SPACE_ID}")
        self._tools = None  # lazy: name -> (server_url, openai_spec)

    # ---------- tools ----------
    def tools(self):
        if self._tools is None:
            self._tools = {}
            for url in self.servers:
                client = DatabricksMCPClient(server_url=url, workspace_client=self.w)
                for t in _mcp(client.list_tools):
                    self._tools[t.name] = (url, {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": (t.description or "")[:1000],
                            "parameters": _schema(t),
                        },
                    })
        return self._tools

    def resolve_tool(self, name: str):
        """Map what the model called (e.g. 'fn_compare_periods') to the MCP tool name
        (e.g. 'agentpoc__sales_poc__fn_compare_periods')."""
        tools = self.tools()
        if name in tools:
            return name
        short = name.split("__")[-1].split(".")[-1]
        for full in tools:
            if full.split("__")[-1] == short:
                return full
        return None

    @mlflow.trace(span_type=SpanType.TOOL)
    def call_tool(self, name: str, args: dict) -> str:
        full = self.resolve_tool(name)
        if full is None:
            short_names = [n.split("__")[-1] for n in self.tools()]
            return f"TOOL ERROR: unknown tool '{name}'. Available tools: {short_names}"
        url, spec = self.tools()[full]
        name = full
        # Reject misspelled / unknown arguments instead of letting the function silently fall back to defaults
        # (e.g. 'country' instead of 'p_country' would otherwise run for ALL countries).
        props = (spec.get("function", {}).get("parameters") or {}).get("properties") or {}
        if props:
            unknown = [k for k in args if k not in props]
            if unknown:
                return (f"TOOL ERROR: unknown argument(s) {unknown} for {name.split('__')[-1]}. "
                        f"Valid arguments: {list(props)}. Call again with the correct names.")
        client = DatabricksMCPClient(server_url=url, workspace_client=self.w)
        try:
            res = _mcp(client.call_tool, name, args)
            text = "\n".join(getattr(c, "text", str(c)) for c in res.content)
        except Exception as e:  # return errors to the model so it can correct its call
            text = f"TOOL ERROR: {_err(e)}"
        return text[:20000]

    @mlflow.trace(name="answer_guard")
    def check_answer(self, text: str) -> list:
        return _guard_problems(text)

    # ---------- message conversion ----------
    @staticmethod
    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for p in content or []:
            p = p if isinstance(p, dict) else p.model_dump()
            if p.get("text"):
                parts.append(p["text"])
        return "\n".join(parts)

    def to_chat(self, items) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for it in items:
            it = it if isinstance(it, dict) else it.model_dump()
            typ = it.get("type", "message")
            if typ == "function_call":
                msgs.append({"role": "assistant", "content": None, "tool_calls": [{
                    "id": it["call_id"], "type": "function",
                    "function": {"name": it["name"], "arguments": it["arguments"]}}]})
            elif typ == "function_call_output":
                msgs.append({"role": "tool", "tool_call_id": it["call_id"], "content": it["output"]})
            elif it.get("role") in ("user", "assistant", "system"):
                msgs.append({"role": it["role"], "content": self._text(it.get("content"))})
        return msgs

    # ---------- agent loop ----------
    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        msgs = self.to_chat(request.input)
        specs = [spec for _, spec in self.tools().values()]
        guard_used = False
        for _ in range(MAX_STEPS):
            resp = self.llm.chat.completions.create(model=LLM_ENDPOINT, messages=msgs, tools=specs, temperature=0)
            m = resp.choices[0].message
            if not m.tool_calls:
                text = _content_text(m.content)
                if not text.strip():
                    # Reasoning models sometimes stop after "thinking" without calling a tool or answering.
                    msgs.append({"role": "user", "content": "Continue. Call the next tool you need now, "
                                 "or if you have enough evidence, write the final answer in the required format."})
                    continue
                problems = [] if guard_used else self.check_answer(text)
                if problems:
                    # One automatic rewrite when the answer breaks the evidence rules.
                    guard_used = True
                    msgs.append({"role": "assistant", "content": text})
                    msgs.append({"role": "user", "content": "Your answer breaks these rules:\n- " + "\n- ".join(problems)
                                 + "\nRewrite the full answer in the same format. Use only numbers from tool results; "
                                   "call a tool if you need a missing number."})
                    continue
                yield ResponsesAgentStreamEvent(type="response.output_item.done",
                                                item=self.create_text_output_item(text=text, id=str(uuid.uuid4())))
                return
            msgs.append({"role": "assistant", "content": _content_text(m.content) or None, "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
            for tc in m.tool_calls:
                yield ResponsesAgentStreamEvent(type="response.output_item.done", item=self.create_function_call_item(
                    id=str(uuid.uuid4()), call_id=tc.id, name=tc.function.name, arguments=tc.function.arguments))
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = self.call_tool(tc.function.name, args)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out})
                yield ResponsesAgentStreamEvent(type="response.output_item.done",
                                                item=self.create_function_call_output_item(call_id=tc.id, output=out))
        yield ResponsesAgentStreamEvent(type="response.output_item.done", item=self.create_text_output_item(
            text="I hit the step limit before finishing. Try narrowing the question (one country or one period).",
            id=str(uuid.uuid4())))

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [e.item for e in self.predict_stream(request) if e.type == "response.output_item.done"]
        return ResponsesAgentResponse(output=outputs)


AGENT = SalesReasoningAgent()
mlflow.models.set_model(AGENT)
