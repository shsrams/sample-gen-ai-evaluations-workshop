"""Pure helpers of the Converse wrapper, driven with content blocks shaped like Bedrock responses.

Tool results go through the real gateway; no Bedrock call is made here.
"""

from fsi_eval.converse_agent import (
    greedy_decoding_fields,
    map_in_order,
    messages_in_window,
    scripted_exchange,
    tool_results_for,
    visible_text,
)
from fsi_eval.systems_of_record import SystemsOfRecord


def test_visible_text_strips_thinking():
    blocks = [{"text": "<thinking>Look up accounts first.</thinking>\nYou have two accounts."}]
    visible, raw = visible_text(blocks)
    assert visible == "You have two accounts."
    assert raw.startswith("<thinking>")


def test_visible_text_empty_when_only_tool_use():
    blocks = [{"toolUse": {"toolUseId": "t1", "name": "getAccounts", "input": {}}}]
    assert visible_text(blocks) == ("", "")


def test_visible_text_joins_text_blocks():
    blocks = [{"text": "First."}, {"toolUse": {"toolUseId": "t1", "name": "x", "input": {}}}, {"text": "Second."}]
    assert visible_text(blocks)[0] == "First.\nSecond."


def test_tool_results_for_executes_through_gateway(fixtures, session):
    systems = SystemsOfRecord(fixtures)
    blocks = [{"text": "<thinking>x</thinking>"},
              {"toolUse": {"toolUseId": "tool-abc", "name": "getAccounts", "input": {}}}]
    results = tool_results_for(blocks, systems, session, turn_index=2)
    assert len(results) == 1
    block = results[0]["toolResult"]
    assert block["toolUseId"] == "tool-abc"
    assert block["status"] == "success"
    assert block["content"][0]["json"]["source"] == "Core Banking"
    call = systems.store.tool_calls[-1]
    assert call.tool == "getAccounts" and call.turn_index == 2


def test_tool_results_for_marks_error_status(fixtures, session):
    systems = SystemsOfRecord(fixtures)
    blocks = [{"toolUse": {"toolUseId": "tool-x", "name": "noSuchTool", "input": {}}}]
    block = tool_results_for(blocks, systems, session, turn_index=0)[0]["toolResult"]
    assert block["status"] == "error"
    assert block["content"][0]["json"]["error"] == "unknown_tool"


def test_scripted_exchange_shape():
    messages = scripted_exchange("How's the weather?", "Sunny where you are, I hope.")
    assert messages == [
        {"role": "user", "content": [{"text": "How's the weather?"}]},
        {"role": "assistant", "content": [{"text": "Sunny where you are, I hope."}]},
    ]


def test_greedy_decoding_fields_only_for_nova():
    # Nova 2 Lite (the agent under test) reads topK from the nested inferenceConfig; a top-level topK is rejected.
    assert greedy_decoding_fields("us.amazon.nova-2-lite-v1:0") == {"inferenceConfig": {"topK": 1}}
    assert greedy_decoding_fields("us.amazon.nova-micro-v1:0") == {"inferenceConfig": {"topK": 1}}
    assert greedy_decoding_fields("us.anthropic.claude-sonnet-5") is None


def test_map_in_order_keeps_input_order():
    assert map_in_order(lambda x: x * x, [3, 1, 2], max_workers=3) == [9, 1, 4]


def test_messages_in_window_keeps_whole_turns():
    messages = [
        {"role": "user", "content": [{"text": "turn 0"}]},
        {"role": "assistant", "content": [{"text": "reply 0"}]},
        {"role": "user", "content": [{"text": "turn 1"}]},
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t", "name": "getCards", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t", "content": [{"json": {}}]}}]},
    ]
    turn_starts = [0, 2]
    assert messages_in_window(messages, turn_starts, None) == messages
    assert messages_in_window(messages, turn_starts, 0) == messages[2:]
    assert messages_in_window(messages, turn_starts, 1) == messages
    assert messages_in_window(messages, turn_starts, 5) == messages
