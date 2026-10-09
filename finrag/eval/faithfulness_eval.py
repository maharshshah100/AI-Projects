"""
faithfulness_eval.py — LLM-as-judge faithfulness scoring, validated against
human labels before being trusted.

Faithfulness answers a different question than retrieval or the gate: given
the context that WAS retrieved and an answer that WAS generated, does the
answer actually stick to what the context supports, or does it add
unsupported claims? This is exactly the layer gate_eval.py's results proved
is necessary -- retrieval confidence can't catch "the right company was
mentioned but this specific claim isn't supported," only a reader that
understands meaning can.

CRITICAL METHODOLOGY NOTE: an LLM-as-judge is only trustworthy once you've
checked it against real human judgments on a sample and confirmed they
agree. Skipping that validation step and just trusting the judge is the
single most common way people fake evaluation rigor without actually having
it -- so this file has two parts, not one:

  1. score_faithfulness() -- the actual judge, requires a live API call.
  2. HUMAN_VALIDATION_SET -- a small set of (answer, expected_human_verdict)
     pairs labeled BY HAND, used to check the judge's agreement rate with a
     real person before trusting its scores on the full golden dataset.

Both require GROQ_API_KEY and network -- neither can run in this
sandbox. Run this locally.
"""
from dataclasses import dataclass
from typing import List


JUDGE_PROMPT = """You are a strict fact-checker. You will be shown a QUESTION, the SOURCE \
material that was retrieved to answer it, and a generated ANSWER. Your job is ONLY to check \
whether every factual claim in the ANSWER is actually supported by the SOURCE material.

Score the answer as one of:
- "FAITHFUL" -- every factual claim in the answer is directly supported by the source.
- "UNSUPPORTED" -- the answer contains at least one factual claim (a number, a fact, an \
event) that is NOT present in the source material, even if it sounds plausible.
- "APPROPRIATELY_DECLINED" -- the answer correctly states that the source doesn't contain \
enough information, rather than guessing.

Respond with ONLY the label, then a dash, then a one-sentence reason. Example:
UNSUPPORTED - The answer states Q3 revenue was $12M, but no Q3 figure appears in the source.

QUESTION: {question}

SOURCE:
{source}

ANSWER:
{answer}
"""


@dataclass
class HumanLabel:
    question: str
    source: str
    answer: str
    human_verdict: str  # "FAITHFUL" | "UNSUPPORTED" | "APPROPRIATELY_DECLINED"
    note: str = ""


# Hand-labeled by a human (not generated), specifically including one
# deliberately unfaithful answer and one deliberately over-cautious decline,
# so the validation set can actually catch a judge that's too lenient or too
# strict -- a validation set of only "obviously faithful" examples wouldn't
# test anything.
HUMAN_VALIDATION_SET = [
    HumanLabel(
        question="What was Apple's total net sales in fiscal year 2024?",
        source="Total net sales: $391,035 (2024), +2% change, $383,285 (2023)",
        answer="Apple's total net sales in fiscal year 2024 were $391,035 million [1].",
        human_verdict="FAITHFUL",
        note="Directly supported, correctly cited.",
    ),
    HumanLabel(
        question="What was Apple's total net sales in fiscal year 2024?",
        source="Total net sales: $391,035 (2024), +2% change, $383,285 (2023)",
        answer="Apple's total net sales in fiscal year 2024 were approximately $400 billion, "
               "reflecting continued strong iPhone demand [1].",
        human_verdict="UNSUPPORTED",
        note="The $400B figure is wrong (real figure is $391,035M) and the 'iPhone demand' "
             "causal claim isn't in the source at all -- a human fact-checker would flag both.",
    ),
    HumanLabel(
        question="How many unicorns does Tesla have in its stable?",
        source="Tesla annual revenue: 2023 $96.77B, 2024 $97.69B (roughly flat, +0.9%)",
        answer="I don't have information in the sources about Tesla having any unicorns -- "
               "this appears to be a nonsensical question for an automotive company.",
        human_verdict="APPROPRIATELY_DECLINED",
        note="Correctly recognized the source doesn't support this claim, declined rather than guessing.",
    ),
    HumanLabel(
        question="What is Microsoft's relationship with OpenAI?",
        source="Microsoft has a long-term partnership with OpenAI... Azure powers all of "
               "OpenAI's workloads.",
        answer="Microsoft owns a 49% equity stake in OpenAI and Azure exclusively powers "
               "all of OpenAI's compute workloads [1].",
        human_verdict="UNSUPPORTED",
        note="The '49% equity stake' figure appears nowhere in the source -- this is a "
             "plausible-sounding but fabricated specific number, exactly the failure mode "
             "faithfulness scoring exists to catch.",
    ),
]


def score_faithfulness(question: str, source: str, answer: str, api_key: str = None) -> str:
    """Requires `pip install groq` and network access -- run locally. Free
    tier available at console.groq.com."""
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=150,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, source=source, answer=answer)}],
    )
    return response.choices[0].message.content.strip()


def validate_judge_against_humans(api_key: str = None):
    """
    Run this BEFORE trusting the judge on the full golden dataset. If
    agreement with human labels is low, the judge prompt needs work --
    don't proceed to scoring real pipeline outputs with an unvalidated judge.
    """
    agreements = 0
    for label in HUMAN_VALIDATION_SET:
        judge_output = score_faithfulness(label.question, label.source, label.answer, api_key=api_key)
        judge_verdict = judge_output.split("-")[0].strip()
        agrees = judge_verdict == label.human_verdict
        agreements += agrees
        print(f"{'✅' if agrees else '❌'} Human: {label.human_verdict:22s} | Judge: {judge_output[:80]}")
        if not agrees:
            print(f"    Human note: {label.note}")

    agreement_rate = agreements / len(HUMAN_VALIDATION_SET)
    print(f"\nJudge-human agreement: {agreement_rate:.1%} ({agreements}/{len(HUMAN_VALIDATION_SET)})")
    if agreement_rate < 0.75:
        print("⚠️  Agreement below 75% -- do not trust this judge on the full dataset yet. "
              "Revise JUDGE_PROMPT and re-validate before using it for real scoring.")
    return agreement_rate


if __name__ == "__main__":
    print("This requires GROQ_API_KEY and network access -- run locally:")
    print("  export GROQ_API_KEY=gsk_...")
    print("  python3 faithfulness_eval.py")
    print()
    import os
    if os.environ.get("GROQ_API_KEY"):
        validate_judge_against_humans(api_key=os.environ["GROQ_API_KEY"])
    else:
        print("No API key found in environment -- skipping live validation.")
        print(f"Human validation set ready with {len(HUMAN_VALIDATION_SET)} labeled examples.")
