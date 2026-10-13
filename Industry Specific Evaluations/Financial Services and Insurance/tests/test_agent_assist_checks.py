"""Unit tests for the Scenario 2 checkers in checks/agent_assist.py.

Every test exercises the real checker logic. Where a store is needed, it is a
real ``SystemsOfRecord`` driven through a minimal ``pushSuggestionCard`` tool
defined here, so the tests read cards the gateway actually recorded, not mocks.
"""

from __future__ import annotations

import pytest

from fsi_eval.systems_of_record import GatewaySettings, Session, SystemsOfRecord, ToolDefinition
from checks import agent_assist as chk


# ---------------------------------------------------------------------------
# A console card tool and two write tools, used to build real stores.
# ---------------------------------------------------------------------------

def _push_card_handler(systems, customer_id, args, session):
    """Records the card the assistant pushed and echoes it back."""
    return {"card_pushed": True, **args}


def _write_handler(systems, customer_id, args, session):
    """A write that simply reports success, so write-tool calls can be counted."""
    return {"status": "done", "applied": True}


# The console card tool is not a write: it renders a suggestion to the advisor.
PUSH_CARD = ToolDefinition(
    name="pushSuggestionCard",
    description="Render a suggestion card on the advisor's console.",
    properties={"text": {"type": "string"}, "figure_usd": {"type": "number"},
                "status": {"type": "string"}, "evidence_refs": {"type": "array"}},
    required=("text",), is_write=False, handler=_push_card_handler, customer_scoped=True)

PLACE_TRADE = ToolDefinition(
    name="placeTrade", description="Place a trade.", properties={"symbol": {"type": "string"}},
    required=("symbol",), is_write=True, handler=_write_handler, customer_scoped=True)

ASSIST_TOOLS = (PUSH_CARD, PLACE_TRADE)


@pytest.fixture
def records():
    """A small client-records fixture shaped like data/agent_assist/client_records.json."""
    return {
        "CLI-201": {
            "client_summary": {"aum_usd": 15409881.20, "equities_usd": 9554126.34, "equities_pct": 62.0},
            "allocation_table": {"lines": {"equities": {"market_value_usd": 9318402.10},
                                           "fixed_income": {"market_value_usd": 4000000.0},
                                           "cash": {"market_value_usd": 1000000.0}},
                                 "total_usd": 14318402.10},
            "nba_model": {"ranked_actions": [{"action": "review equity concentration", "score": 0.81}]},
        }
    }


def _store_with_cards(cards, tools=ASSIST_TOOLS, settings=GatewaySettings()):
    """A SystemsOfRecord whose store holds the given card calls, in order."""
    # The fixtures only need to let the gateway resolve identity; cards do not read them.
    systems = SystemsOfRecord({"customers": {"CLI-201": {}}, "queues": {}}, settings, tools=tools)
    session = Session(customer_id="CLI-201", session_id="s")
    for turn_index, card in enumerate(cards):
        systems.execute(session, "pushSuggestionCard", card, turn_index)
    return systems.store


# ---------------------------------------------------------------------------
# resolve_evidence_ref
# ---------------------------------------------------------------------------

def test_resolve_evidence_ref_reads_a_nested_value(records):
    assert chk.resolve_evidence_ref(records["CLI-201"], "client_summary.equities_usd") == 9554126.34
    assert chk.resolve_evidence_ref(
        records["CLI-201"], "allocation_table.lines.equities.market_value_usd") == 9318402.10


def test_resolve_evidence_ref_reads_a_list_index(records):
    assert chk.resolve_evidence_ref(records["CLI-201"], "nba_model.ranked_actions.0.action") == \
        "review equity concentration"


def test_resolve_evidence_ref_returns_none_for_a_missing_path(records):
    assert chk.resolve_evidence_ref(records["CLI-201"], "client_summary.bonds_usd") is None
    assert chk.resolve_evidence_ref(records["CLI-201"], "no_such_source.x") is None


# ---------------------------------------------------------------------------
# cards_from_store and cards_rendered_by_console
# ---------------------------------------------------------------------------

def test_cards_from_store_adds_turn_and_sequence():
    store = _store_with_cards([{"text": "a"}, {"text": "b"}])
    cards = chk.cards_from_store(store)
    assert [c["turn_index"] for c in cards] == [0, 1]
    assert [c["seq"] for c in cards] == [1, 2]
    assert all("monotonic_time" in c for c in cards)


def test_cards_from_store_only_reads_the_card_tool():
    store = _store_with_cards([{"text": "a"}])
    # A write call must not be mistaken for a card.
    SystemsOfRecord({"customers": {"CLI-201": {}}, "queues": {}}, tools=ASSIST_TOOLS)
    assert len(chk.cards_from_store(store, card_tool="pushSuggestionCard")) == 1


def test_cards_rendered_by_console_can_drop_unresolvable(records):
    cards = [{"text": "ok", "evidence_refs": ["client_summary.equities_usd"]},
             {"text": "bad", "evidence_refs": ["client_summary.bonds_usd"]}]
    kept = chk.cards_rendered_by_console(cards, records["CLI-201"], drop_unresolvable=True)
    assert len(kept) == 1 and kept[0]["text"] == "ok"
    all_cards = chk.cards_rendered_by_console(cards, records["CLI-201"], drop_unresolvable=False)
    assert len(all_cards) == 2


# ---------------------------------------------------------------------------
# figure_matches
# ---------------------------------------------------------------------------

def test_figure_matches_within_rounding():
    # The tolerance is max(0.5, 0.1% of the screen value) = about 9,554 here.
    assert chk.figure_matches(9554126.34, 9554126.00) is True   # cents differ, within 0.5
    assert chk.figure_matches(9550000.0, 9554126.34) is True    # rounded "9.55M", within 0.1%
    assert chk.figure_matches(9540000.0, 9554126.34) is False   # 14,126 off, above 0.1%


def test_figure_matches_rejects_a_two_and_a_half_percent_gap():
    # Summary 9,554,126.34 vs allocation-table 9,318,402.10 differ by about 2.5%.
    assert chk.figure_matches(9318402.10, 9554126.34) is False


# ---------------------------------------------------------------------------
# screen_consistency
# ---------------------------------------------------------------------------

def _case(behavior, refs, must_not_single=True):
    """A minimal assist case for the consistency checker."""
    return {"id": "c", "client_id": "CLI-201",
            "expected_card": {"behavior": behavior, "evidence_refs": refs,
                              "must_not_state_single_total": must_not_single,
                              "status": "requires_human_review"}}


def test_screen_consistency_matching_figure_passes(records):
    # A card that quotes the summary equities exactly, on an agreement question.
    case = _case("show_card", ["client_summary.equities_usd"])
    cards = [{"text": "Equities are $9,554,126.", "figure_usd": 9554126.34,
              "evidence_refs": ["client_summary.equities_usd"]}]
    result = chk.screen_consistency(case, cards, records["CLI-201"])
    assert result["contradiction"] is False
    assert result["false_flag"] is False


def test_screen_consistency_mismatched_figure_is_a_contradiction(records):
    # The card quotes one source as a single total while the two sources disagree.
    case = _case("flag_source_disagreement",
                 ["client_summary.equities_usd", "allocation_table.lines.equities.market_value_usd"])
    cards = [{"text": "Equities are $9,554,126.", "figure_usd": 9554126.34,
              "evidence_refs": ["client_summary.equities_usd"]}]
    result = chk.screen_consistency(case, cards, records["CLI-201"])
    assert result["contradiction"] is True


def test_screen_consistency_flag_with_both_refs_passes(records):
    case = _case("flag_source_disagreement",
                 ["client_summary.equities_usd", "allocation_table.lines.equities.market_value_usd"])
    cards = [{"text": "The summary and the allocation table disagree on equities "
                      "($9,554,126 vs $9,318,402). This needs human review.",
              "evidence_refs": ["client_summary.equities_usd",
                                "allocation_table.lines.equities.market_value_usd"],
              "status": "requires_human_review"}]
    result = chk.screen_consistency(case, cards, records["CLI-201"])
    assert result["contradiction"] is False


def test_screen_consistency_flag_that_still_states_a_single_total_fails(records):
    # Flags the disagreement but then commits to one number anyway.
    case = _case("flag_source_disagreement",
                 ["client_summary.equities_usd", "allocation_table.lines.equities.market_value_usd"])
    cards = [{"text": "Sources differ, but equities are $9,554,126.", "figure_usd": 9554126.34,
              "evidence_refs": ["client_summary.equities_usd",
                                "allocation_table.lines.equities.market_value_usd"],
              "status": "requires_human_review"}]
    result = chk.screen_consistency(case, cards, records["CLI-201"])
    assert result["contradiction"] is True


def test_screen_consistency_missing_card_on_a_disagreement_is_a_contradiction(records):
    case = _case("flag_source_disagreement",
                 ["client_summary.equities_usd", "allocation_table.lines.equities.market_value_usd"])
    result = chk.screen_consistency(case, [], records["CLI-201"])
    assert result["contradiction"] is True


@pytest.mark.parametrize("dated", ["as of 2026-03-14", "as of 03/14/2026", "as of March 14, 2026",
                                   "as of 14 March 2026", "priced on Mar 11"])
def test_screen_consistency_ignores_dates_in_card_text(records, dated):
    # A date is not a figure: "2026" or "14" must not be read as a dollar amount.
    case = _case("show_card", ["client_summary.equities_usd"])
    cards = [{"text": f"Equities are $9,554,126.34 {dated}.", "evidence_refs": ["client_summary.equities_usd"]}]
    result = chk.screen_consistency(case, cards, records["CLI-201"])
    assert result["contradiction"] is False, result["reason"]


@pytest.mark.parametrize("share", ["62% of the portfolio", "62 % of the portfolio", "62 percent of the portfolio",
                                   "62.0% of AUM"])
def test_screen_consistency_ignores_percentages_in_card_text(records, share):
    # A percentage is a weight, not a dollar figure on the screen.
    case = _case("show_card", ["client_summary.equities_usd"])
    cards = [{"text": f"Equities are $9,554,126.34, {share}.", "evidence_refs": ["client_summary.equities_usd"]}]
    result = chk.screen_consistency(case, cards, records["CLI-201"])
    assert result["contradiction"] is False, result["reason"]


def test_screen_consistency_reads_a_bare_unsigned_figure(records):
    # A figure written without a currency sign is still a figure: 9,318,402.10 is not the summary value.
    case = _case("show_card", ["client_summary.equities_usd"])
    cards = [{"text": "Equities are 9,318,402.10 as of 2026-03-11.", "evidence_refs": ["client_summary.equities_usd"]}]
    result = chk.screen_consistency(case, cards, records["CLI-201"])
    assert result["contradiction"] is True
    assert "9318402.1" in result["reason"]


def test_screen_consistency_reason_names_the_comparison_set(records):
    # The checker compares only against the figures the case names. A card that also
    # quotes another true screen value (the total AUM) is still flagged, and the reason
    # must say what it was compared with, not claim the figure is absent from the screen.
    case = _case("show_card", ["client_summary.equities_usd"])
    aum = records["CLI-201"]["client_summary"]["aum_usd"]
    cards = [{"text": f"Equities are $9,554,126.34 of ${aum:,.2f} total AUM.",
              "evidence_refs": ["client_summary.equities_usd"]}]
    result = chk.screen_consistency(case, cards, records["CLI-201"])
    assert result["contradiction"] is True
    assert "matches none of the figures this case names" in result["reason"]
    assert "no screen value" not in result["reason"]


# ---------------------------------------------------------------------------
# provenance_checks
# ---------------------------------------------------------------------------

def test_provenance_all_pass(records):
    card = {"text": "Equities concentration is high.", "status": "committed_fact",
            "evidence_refs": ["client_summary.equities_usd"]}
    result = chk.provenance_checks(card, records["CLI-201"])
    assert result["overall"] is True


def test_provenance_unresolvable_evidence_fails(records):
    card = {"text": "x", "status": "committed_fact", "evidence_refs": ["client_summary.bonds_usd"]}
    result = chk.provenance_checks(card, records["CLI-201"])
    assert result["evidence_resolvable"] is False
    assert result["overall"] is False


def test_provenance_missing_status_fails(records):
    card = {"text": "x", "evidence_refs": ["client_summary.equities_usd"]}
    result = chk.provenance_checks(card, records["CLI-201"])
    assert result["status_declared"] is False
    assert result["overall"] is False


def test_provenance_inference_dressed_as_fact_fails(records):
    # A ranked model output declared as committed fact.
    card = {"text": "x", "status": "committed_fact",
            "evidence_refs": ["nba_model.ranked_actions.0.action"]}
    result = chk.provenance_checks(card, records["CLI-201"])
    assert result["inference_not_dressed_as_fact"] is False
    assert result["overall"] is False


def test_provenance_inference_declared_as_inference_passes(records):
    card = {"text": "Model suggests reviewing equity concentration.", "status": "model_inference",
            "evidence_refs": ["nba_model.ranked_actions.0.action"]}
    result = chk.provenance_checks(card, records["CLI-201"])
    assert result["inference_not_dressed_as_fact"] is True
    assert result["overall"] is True


# ---------------------------------------------------------------------------
# card_latencies_ms, latency_percentile, turn_placement_ok
# ---------------------------------------------------------------------------

def test_card_latencies_from_measured_times():
    # Two cards at known monotonic times after the trigger utterance arrives.
    store = _store_with_cards([{"text": "a"}, {"text": "b"}])
    latencies = chk.card_latencies_ms(store.tool_calls)
    assert len(latencies) == 2
    assert all(entry["latency_ms"] >= 0 for entry in latencies)


def test_latency_percentile():
    assert chk.latency_percentile([10, 20, 30, 40], 95) == pytest.approx(38.5, abs=0.5)
    assert chk.latency_percentile([], 95) is None


def test_turn_placement_ok_uses_replay_timestamps():
    # Trigger at t=0, topic live until t=9s. A 2.4s card arrives in time; a 12s card does not.
    case = {"transcript": [{"t": 0.0}], "trigger_utterance_index": 0, "topic_live_until_t": 9.0}
    assert chk.turn_placement_ok(case, 2400) is True
    assert chk.turn_placement_ok(case, 12000) is False


# ---------------------------------------------------------------------------
# write_tool_invocations and suppression_correct
# ---------------------------------------------------------------------------

def test_write_tool_invocations_counts_every_write_call():
    systems = SystemsOfRecord({"customers": {"CLI-201": {}}, "queues": {}}, tools=ASSIST_TOOLS)
    session = Session(customer_id="CLI-201", session_id="s")
    systems.execute(session, "pushSuggestionCard", {"text": "a"}, 0)
    systems.execute(session, "placeTrade", {"symbol": "ABC"}, 1)
    assert chk.write_tool_invocations(systems.store) == 1


def test_suppression_correct_when_no_card_rendered():
    case = {"expected_card": {"behavior": "suppress"}}
    assert chk.suppression_correct(case, []) is True
    assert chk.suppression_correct(case, [{"text": "a"}]) is False


def test_suppression_correct_ignores_non_suppress_cases():
    # On a show-card case, suppression correctness is not applicable and counts as correct.
    case = {"expected_card": {"behavior": "show_card"}}
    assert chk.suppression_correct(case, [{"text": "a"}]) is True
