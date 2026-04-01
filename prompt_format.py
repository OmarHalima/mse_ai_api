"""
Single-string prompt builder for OpenAI-compatible /v1/chat/completions (LangChain, n8n).
Ported from another_app_code/another_app_code.py — tool results, assistant tool_calls, etc.
"""
import json
import re
import uuid
from typing import Optional


def _strip_degenerate_tool_json(text: str) -> str:
    if not text or not text.strip():
        return text
    raw = text.strip()
    t = raw
    if "```" in t:
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", t, re.DOTALL)
        if m:
            t = m.group(1).strip()
    try:
        o = json.loads(t)
        if isinstance(o, dict) and "tool_calls" in o and o.get("tool_calls") == []:
            if set(o.keys()) <= {"tool_calls"}:
                return ""
    except (json.JSONDecodeError, TypeError):
        pass
    if re.fullmatch(r'\{\s*"tool_calls"\s*:\s*\[\s*\]\s*\}', t):
        return ""
    return raw


def sanitize_assistant_content(content: str) -> str:
    """Avoid echoing {\"tool_calls\": []} into the linear prompt."""
    if not content or not str(content).strip():
        return content
    s = _strip_degenerate_tool_json(str(content))
    if not s.strip():
        return "[no assistant text]"
    return s


def format_tools_instruction(tools: list, user_question: str = "") -> str:
    instruction = "\n=== MANDATORY TOOL USAGE ===\n"
    instruction += "You MUST use one of the tools below to answer this question.\n"
    instruction += "Do NOT answer directly. Do NOT say you don't have information.\n"
    instruction += "You MUST respond with ONLY a JSON object to call the tool.\n\n"
    instruction += "RESPONSE FORMAT - respond with ONLY this JSON, nothing else:\n"
    instruction += '{"tool_calls": [{"name": "TOOL_NAME", "arguments": {"param": "value"}}]}\n\n'
    instruction += "RULES:\n"
    instruction += "- Your ENTIRE response must be valid JSON only\n"
    instruction += "- No markdown, no code blocks, no explanation\n"
    instruction += "- No text before or after the JSON\n\n"
    instruction += "Available tools:\n\n"
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")
        params = func.get("parameters", {})
        instruction += f"Tool: {name}\n"
        instruction += f"Description: {desc}\n"
        if params.get("properties"):
            instruction += "Parameters:\n"
            required_params = params.get("required", [])
            for param_name, param_info in params["properties"].items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                is_required = "required" if param_name in required_params else "optional"
                instruction += f"  - {param_name} ({param_type}, {is_required}): {param_desc}\n"
        instruction += "\n"
    instruction += "=== END OF TOOLS ===\n\n"
    first_tool = tools[0] if tools else {}
    first_func = first_tool.get("function", first_tool)
    first_name = first_func.get("name", "tool")
    instruction += "EXAMPLE: If the user asks a question, respond with:\n"
    instruction += '{"tool_calls": [{"name": "' + first_name + '", "arguments": {"input": "the user question here"}}]}\n\n'
    instruction += "Now respond with the JSON to call the appropriate tool:\n\n"
    return instruction


def format_prompt(messages: list, tools: Optional[list] = None) -> str:
    parts: list[str] = []
    system_parts: list[str] = []
    has_tool_results = False
    user_question = ""

    for msg in messages:
        role = msg.get("role", "")
        msg_type = msg.get("type", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text_parts.append(item.get("text", item.get("content", str(item))))
                else:
                    text_parts.append(str(item))
            content = "\n".join(text_parts)

        if role == "system":
            system_parts.append(content)
        elif role == "tool":
            has_tool_results = True
            tool_name = msg.get("name", "tool")
            parts.append(f"[TOOL RESULT from '{tool_name}']:\n{content}")
        elif msg_type == "function_call_output":
            has_tool_results = True
            call_id = msg.get("call_id", "")
            output_content = msg.get("output", content)
            parts.append(f"[TOOL RESULT (call_id: {call_id})]:\n{output_content}")
        elif msg_type == "function_call":
            func_name = msg.get("name", "?")
            func_args = msg.get("arguments", "{}")
            parts.append(f"[PREVIOUS TOOL CALL: Called '{func_name}' with arguments: {func_args}]")
        elif role == "assistant":
            assistant_content = content if content else ""
            assistant_content = sanitize_assistant_content(assistant_content)
            tool_calls_in_msg = msg.get("tool_calls") or []
            if tool_calls_in_msg:
                tc_descriptions = []
                for tc in tool_calls_in_msg:
                    func = tc.get("function", {})
                    tc_descriptions.append(
                        f"Called '{func.get('name', '?')}' with: {func.get('arguments', '{}')}"
                    )
                assistant_content += "\n[Previous tool calls: " + "; ".join(tc_descriptions) + "]"
            if assistant_content.strip():
                parts.append(f"[Assistant]: {assistant_content}")
        elif role == "user" or (msg_type == "message" and role != "system"):
            user_question = content
            parts.append(content)
            has_tool_results = False
        elif content:
            parts.append(content)

    final = ""

    if system_parts:
        if tools and not has_tool_results:
            final += "=== YOUR ROLE ===\n"
            final += "\n\n".join(system_parts)
            final += "\n=== END OF ROLE ===\n\n"
        else:
            final += "=== SYSTEM INSTRUCTIONS (FOLLOW STRICTLY) ===\n"
            final += "\n\n".join(system_parts)
            final += "\n=== END OF INSTRUCTIONS ===\n\n"

    if tools and not has_tool_results:
        final += format_tools_instruction(tools, user_question)

    if has_tool_results:
        final += "=== CONTEXT FROM TOOLS ===\n"
        final += "The following information was retrieved by the tools you requested.\n"
        final += "Use ONLY this information to answer the user's question.\n\n"

    if parts:
        final += "\n".join(parts)

    if has_tool_results:
        final += "\n\n=== INSTRUCTION ===\n"
        final += "Now answer the user's question based ONLY on the tool results above.\n"

    return final


def parse_tool_calls_lite(response_text: str) -> Optional[list]:
    """Same as another_app parse_tool_calls — OpenAI-shaped tool_calls list."""
    cleaned = response_text.strip()
    if "```" in cleaned:
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
        if code_block_match:
            cleaned = code_block_match.group(1).strip()

    json_candidates = [cleaned]
    json_match = re.search(r'\{[\s\S]*"tool_calls"[\s\S]*\}', cleaned)
    if json_match:
        json_candidates.append(json_match.group(0))

    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                raw_calls = parsed["tool_calls"]
                if isinstance(raw_calls, list) and len(raw_calls) > 0:
                    formatted_calls = []
                    for call in raw_calls:
                        tool_name = call.get("name", "")
                        arguments = call.get("arguments", {})
                        if isinstance(arguments, dict):
                            arguments_str = json.dumps(arguments, ensure_ascii=False)
                        else:
                            arguments_str = str(arguments)
                        formatted_calls.append(
                            {
                                "id": f"call_{uuid.uuid4().hex[:24]}",
                                "type": "function",
                                "function": {"name": tool_name, "arguments": arguments_str},
                            }
                        )
                    return formatted_calls
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return None
