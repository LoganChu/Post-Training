from datasets import Dataset

from posttrain.data import DAPO_PREFIX, DAPO_SUFFIX, EvalIndex, decontaminate, gold_is_gradable, is_binary_answer, to_grpo_dataset


def test_dapo_wrapper_constants_round_trip():
    problem = "Compute $1+1$."
    wrapped = DAPO_PREFIX + problem + DAPO_SUFFIX
    assert wrapped[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)] == problem


def test_gold_gradability():
    for ok in ["34", "-3", "\\frac{1}{2}", "2\\sqrt{2}", "(x+1)^2"]:
        assert gold_is_gradable(ok), ok
    assert not gold_is_gradable("")


def test_binary_answers():
    for a in ["Yes", "no", "\\text{True}", "False.", " yes "]:
        assert is_binary_answer(a), a
    for a in ["34", "\\frac{1}{2}", "Yesterday", "p < 1", "\\text{No solution}"]:
        assert not is_binary_answer(a), a


BOILERPLATE = "can be written as m n where m and n are relatively prime positive integers find m n"


def test_decontaminate_drops_copies_not_boilerplate():
    eval_problem = (
        "When rolling a certain unfair six sided die with faces numbered 1 2 3 4 5 and 6 the probability "
        "of obtaining face F is greater than one sixth and the probability of the opposite face is less. " + BOILERPLATE
    )
    index = EvalIndex([eval_problem])
    ds = Dataset.from_list(
        [
            {"problem": "Restated: " + eval_problem.replace("six sided", "six-sided"), "solution": "copy"},
            {"problem": "A circle is inscribed in a right triangle with legs 6 and 8. Its area " + BOILERPLATE, "solution": "boilerplate"},
            {"problem": "Unrelated problem about counting lattice points inside a circle of radius ten", "solution": "unrelated"},
        ]
    )
    assert decontaminate(ds, index)["solution"] == ["boilerplate", "unrelated"]


def test_grpo_formats():
    ds = Dataset.from_list([{"problem": "What is 1+1?", "solution": "2", "source": "t", "id": "t-0"}])
    zero = to_grpo_dataset(ds, "zero")[0]
    assert isinstance(zero["prompt"], str) and zero["prompt"].endswith("Assistant:") and zero["solution"] == "2"
    assert "problem" not in zero
    think = to_grpo_dataset(ds, "think")[0]
    assert think["prompt"][0]["role"] == "user" and "\\boxed{}" in think["prompt"][0]["content"]
