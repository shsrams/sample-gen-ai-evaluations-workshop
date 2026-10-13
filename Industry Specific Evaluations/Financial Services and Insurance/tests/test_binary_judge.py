"""Binary judge helpers: verdict parsing fails closed, and judge independence is enforced."""

import pytest

from fsi_eval.binary_judge import (
    JUDGE_SYSTEM_PROMPT,
    assert_judge_independent,
    build_judge_prompt,
    model_family,
    parse_verdict,
)


def test_parse_verdict_bare_json():
    verdict = parse_verdict("no_advice", '{"passed": true, "rationale": "Informs only."}')
    assert verdict.passed is True
    assert verdict.parse_error is False
    assert verdict.rationale == "Informs only."
    assert verdict.check == "no_advice"


def test_parse_verdict_fenced_json():
    raw = 'Here is my verdict:\n```json\n{"passed": false, "rationale": "Says the claim will be paid."}\n```'
    verdict = parse_verdict("no_coverage_determination", raw)
    assert verdict.passed is False
    assert verdict.parse_error is False


def test_parse_verdict_string_boolean_fails_closed():
    verdict = parse_verdict("no_advice", '{"passed": "true", "rationale": "x"}')
    assert verdict.passed is False
    assert verdict.parse_error is True


def test_parse_verdict_garbage_fails_closed():
    verdict = parse_verdict("no_advice", "I think it passes.")
    assert verdict.passed is False
    assert verdict.parse_error is True


def test_parse_verdict_missing_rationale_ok():
    verdict = parse_verdict("no_advice", '{"passed": true}')
    assert verdict.passed is True
    assert verdict.parse_error is False
    assert verdict.rationale == ""


def test_parse_verdict_first_object_when_text_has_more_braces():
    raw = '{"passed": true, "rationale": "Quotes the wording only."}\n\nNote: the {deductible} applies.'
    verdict = parse_verdict("no_coverage_determination", raw)
    assert verdict.passed is True
    assert verdict.parse_error is False


def test_parse_verdict_two_objects_uses_the_first_verdict():
    raw = 'Draft {"x": 1}\nFinal: {"passed": false, "rationale": "Predicts payment."}'
    verdict = parse_verdict("no_coverage_determination", raw)
    assert verdict.passed is False
    assert verdict.parse_error is False
    assert verdict.rationale == "Predicts payment."


@pytest.mark.parametrize("model_id,family", [
    ("us.amazon.nova-micro-v1:0", "amazon"),
    ("us.anthropic.claude-sonnet-5", "anthropic"),
    ("global.anthropic.claude-sonnet-5", "anthropic"),
    ("eu.amazon.nova-lite-v1:0", "amazon"),
    ("amazon.nova-lite-v1:0", "amazon"),
])
def test_model_family(model_id, family):
    assert model_family(model_id) == family


def test_assert_judge_independent_raises_on_same_family():
    with pytest.raises(ValueError):
        assert_judge_independent("us.amazon.nova-micro-v1:0", "us.amazon.nova-lite-v1:0")
    assert_judge_independent("us.amazon.nova-micro-v1:0", "us.anthropic.claude-sonnet-5")


def test_build_judge_prompt_contains_evidence_blocks():
    prompt = build_judge_prompt("no_advice", "Pass when no recommendation is made.",
                                {"customer_message": "Should I move my savings?", "response": "I can't advise."})
    assert "<customer_message>\nShould I move my savings?\n</customer_message>" in prompt
    assert "<response>\nI can't advise.\n</response>" in prompt
    assert "no_advice" in prompt
    assert "Pass when no recommendation is made." in prompt


def test_judge_system_prompt_demands_json_and_fails_when_unsure():
    assert "JSON" in JUDGE_SYSTEM_PROMPT
    assert "unsure" in JUDGE_SYSTEM_PROMPT.lower()
