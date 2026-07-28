#!/usr/bin/env python3
"""
eval_harness.py — measured accuracy for JurisDocs AI.

This does NOT prove the tool is correct on every question; that is not
achievable for an open-ended natural language system. It measures accuracy on a
fixed, attorney-verified question bank so you can state a defensible number with
a known sample size, and detect regressions when you change the pipeline.

Two layers:

  RETRIEVAL (deterministic, free, no API calls)
      Did the governing section text actually reach the model's context?
      Every retrieval failure is a guaranteed answer failure, so this layer
      catches the majority of problems at zero cost. Run it on every commit.

  ANSWER (requires HF_TOKEN, costs API calls, non-deterministic)
      Given correct context, did the model state the right facts, cite the right
      subsection, and refuse when it should have?

Usage:
    # fast, free, deterministic — run this constantly
    python eval_harness.py --pdf ch24.pdf ch92.pdf --retrieval-only

    # full evaluation, 3 runs per question to measure variance
    export HF_TOKEN=hf_...
    python eval_harness.py --pdf ch24.pdf ch92.pdf --runs 3 --out results.csv
"""

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict

import yaml

import rag_core


REFUSAL_MARKERS = [
    r"does not (?:contain|cover|address|govern|include|mention|provide|specify)",
    r"do not (?:contain|cover|address|govern|include|mention|provide|specify)",
    r"no (?:direct )?(?:statutory )?(?:basis|provision|section|mention|reference)",
    r"not (?:explicitly )?(?:stated|mentioned|addressed|found|present|included)",
    r"is not (?:in|part of|within|governed by)",
    r"there is no ",
    r"I (?:don't|do not) know",
    r"cannot (?:determine|answer|be determined)",
    r"unable to (?:determine|answer|find)",
    r"outside the scope",
    r"not covered by",
    r"appears? to be (?:no|incorrect|a misstatement)",
    r"cannot provide legal advice",
    r"(?:can't|cannot|not able to) (?:provide|give|offer) (?:legal )?advice",
    r"(?:consult|speak with|contact) (?:a|an|your) (?:lawyer|attorney|legal)",
    r"tenant rights organization",
]

# The model's most dangerous habit: correctly noting the document is silent, then
# continuing anyway with general legal knowledge anchored to a real-but-unrelated
# section number. The refusal reads as compliant, so refusal-detection alone
# scores it as a pass. These patterns catch the continuation.
EXTRADOC_MARKERS = [
    r"general principles",
    r"generally speaking",
    r"in general,",
    r"typically,?\s",
    r"usually,?\s",
    r"it can be inferred",
    r"can be inferred that",
    r"common law",
    r"under (?:Texas|state|federal) law(?:,| generally)",
    r"other relevant (?:statutes|laws|provisions)",
    r"may be considered",
]


def is_rate_limit(exc) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "ratelimit" in type(exc).__name__.lower()


def suggested_wait(exc, fallback: float) -> float:
    """Providers usually say how long to wait. Use their number when present."""
    match = re.search(r"try again in ([\d.]+)\s*(m|s)", str(exc), re.IGNORECASE)
    if match:
        value = float(match.group(1))
        return value * 60 if match.group(2).lower() == "m" else value + 1
    return fallback


def call_with_retry(fn, retries: int = 5, base_delay: float = 8.0):
    """
    Free tiers rate-limit aggressively. Without backoff a 24-call run dies after
    the first handful and reports the rest as errors, which tells you nothing
    about the model.
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            if not is_rate_limit(exc) or attempt == retries - 1:
                raise
            wait = suggested_wait(exc, base_delay * (2 ** attempt))
            print(f"         rate limited — waiting {wait:.0f}s "
                  f"(attempt {attempt + 1}/{retries})")
            time.sleep(wait)


def matches_any(patterns, text):
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def all_match(patterns, text):
    missing = [p for p in patterns if not re.search(p, text, re.IGNORECASE)]
    return (not missing), missing


def score_retrieval(question, result):
    """
    Deterministic checks on what reached the context window.
    Returns (passed, list_of_failure_reasons).
    """
    failures = []
    context = result["context"]
    retrieved = set(s for s in result["retrieved_sections"] if s)

    for section in question.get("expect_sections", []):
        if section not in retrieved:
            failures.append(f"section {section} not retrieved")

    ok, missing = all_match(question.get("expect_context_regex", []), context)
    for pattern in missing:
        failures.append(f"context missing /{pattern}/")

    return (not failures), failures


def score_answer(question, result):
    """Checks on the generated text. Returns (passed, failures, cited_sections)."""
    failures = []
    answer = result["answer"]
    cited = set(rag_core.extract_section_numbers(answer))

    refused = matches_any(REFUSAL_MARKERS, answer)

    if question.get("expect_refusal"):
        if not refused:
            failures.append("expected a refusal, model answered instead")
        # A refusal that keeps talking is not a refusal. This is the check that
        # catches "the document doesn't say, but generally... (Sec. 24.0062)".
        for pattern in EXTRADOC_MARKERS:
            if re.search(pattern, answer, re.IGNORECASE):
                failures.append(
                    f"refused then supplied extra-document knowledge (/{pattern}/)"
                )
                break
    else:
        if refused:
            failures.append("model refused on an answerable question")

        ok, missing = all_match(question.get("require_answer_regex", []), answer)
        for pattern in missing:
            failures.append(f"answer missing required /{pattern}/")

    for pattern in question.get("forbid_answer_regex", []):
        if re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"answer contains forbidden /{pattern}/")

    # Citation precision: every section the model cites must be one we expected,
    # one that was actually retrieved, or one the user named in the question.
    # A refusal that says "Section 24.999 does not appear in this document" is
    # correct behavior, not a fabricated citation.
    allowed = set(question.get("expect_sections", []))
    allowed |= set(s for s in result["retrieved_sections"] if s)
    allowed |= set(rag_core.extract_section_numbers(question["question"]))
    fabricated = cited - allowed
    if fabricated:
        failures.append(f"fabricated citation(s): {sorted(fabricated)}")

    # Subsection-level citation, e.g. "24.005(b)". Soft by default because it is
    # the hardest thing to get right; set strict_subsection: true to enforce.
    expected_sub = question.get("expect_subsection")
    if expected_sub:
        # Accept either "24.006(b)" or the prose form "Subsection (b)". Both
        # point the reader to the right provision; only the format differs.
        parts = re.match(r"([\d.]+)\(([a-z])\)", expected_sub)
        if parts:
            section, letter = parts.groups()
            pattern = (rf"(?:{re.escape(section)}\s*\(\s*{letter}\s*\)"
                       rf"|[Ss]ubsection\s*\(\s*{letter}\s*\))")
        else:
            pattern = re.escape(expected_sub)
        if not re.search(pattern, answer):
            msg = f"did not cite subsection {expected_sub}"
            if question.get("strict_subsection"):
                failures.append(msg)
            else:
                failures.append(f"[soft] {msg}")

    hard_failures = [f for f in failures if not f.startswith("[soft]")]
    return (not hard_failures), failures, cited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", nargs="+", required=True, help="one or more statute PDFs")
    parser.add_argument("--questions", default="eval_questions.yaml")
    parser.add_argument("--runs", type=int, default=1, help="repeats per question (variance)")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--out", default=None, help="write per-run results to CSV")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="seconds between calls; raise if rate limited")
    args = parser.parse_args()

    for path in args.pdf:
        if not os.path.exists(path):
            sys.exit(f"PDF not found: {path}")

    with open(args.questions) as fh:
        questions = yaml.safe_load(fh)

    print(f"Indexing {len(args.pdf)} PDF(s)...")
    index = rag_core.build_index_from_paths(args.pdf)
    print(f"  {index['pages']} pages, {len(index['chunks'])} chunks, "
          f"{index['total_chars']:,} chars\n")

    if index["total_chars"] < 500:
        sys.exit("Corpus has almost no extractable text — the PDFs are likely scans. "
                 "Run OCR first.")

    if not rag_core.BM25_AVAILABLE:
        print("WARNING: rank_bm25 missing — keyword retrieval disabled. "
              "Results will not reflect production.\n")

    chain = None
    if not args.retrieval_only:
        provider = rag_core.detect_provider()
        if provider is None:
            sys.exit(
                "No LLM credential found. Set one of:\n"
                "  export GROQ_API_KEY=...    # free tier, console.groq.com\n"
                "  export GOOGLE_API_KEY=...  # free tier, aistudio.google.com\n"
                "  export OLLAMA_MODEL=qwen2.5:7b   # local, offline, unlimited\n"
                "  export HF_TOKEN=...\n"
                "Or pass --retrieval-only to skip the answer layer."
            )

        placeholders = ("hf_your_token", "hf_...", "your_key_here", "hf_your_token_here")
        for var in ("GROQ_API_KEY", "GOOGLE_API_KEY", "HF_TOKEN"):
            if os.environ.get(var, "") in placeholders and os.environ.get(var):
                sys.exit(f"{var} is set to a placeholder value. Use your real key.")

        if provider == "huggingface":
            os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", os.environ.get("HF_TOKEN", ""))

        model = os.environ.get("LLM_MODEL", rag_core.DEFAULT_MODELS.get(provider))
        print(f"Provider: {provider} / {model}")
        chain = rag_core.build_chain()

        # Preflight: one throwaway call. Without this, an auth or billing failure
        # produces 100% "failures" that look like model errors, and you spend an
        # evening debugging a pipeline that was fine.
        print("Preflight: testing inference endpoint...")
        try:
            call_with_retry(lambda: chain.invoke(
                {"input": "Reply with OK.", "context": "none", "chat_history": "none"}))
            print("  endpoint reachable\n")
        except Exception as exc:
            sys.exit(f"\nInference endpoint unreachable — aborting before scoring.\n"
                     f"  {type(exc).__name__}: {exc}\n\n"
                     "Check the credential is valid and has quota remaining.")

    rows = []
    errors = 0
    tally = defaultdict(lambda: {"retrieval": 0, "answer": 0, "runs": 0})

    for question in questions:
        qid = question["id"]
        for run in range(1, args.runs + 1):
            if args.retrieval_only:
                docs, query_sections = rag_core.hybrid_retrieve(index, question["question"])
                result = {
                    "answer": "",
                    "context": rag_core.format_context(docs),
                    "retrieved_sections": [d.metadata.get("section", "") for d in docs],
                    "query_sections": query_sections,
                }
            else:
                try:
                    result = call_with_retry(
                        lambda: rag_core.answer_question(index, chain, question["question"])
                    )
                    result["error"] = None
                    if args.delay:
                        time.sleep(args.delay)
                except Exception as exc:
                    result = {
                        "answer": "",
                        "context": "",
                        "retrieved_sections": [],
                        "query_sections": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }

            if result.get("error"):
                # An infrastructure failure is not an accuracy result. Scoring it
                # as one would understate the pipeline and send you chasing the
                # wrong bug.
                errors += 1
                rows.append({
                    "id": qid, "run": run,
                    "retrieval_pass": "ERROR", "answer_pass": "ERROR",
                    "retrieval_failures": "", "answer_failures": result["error"],
                    "cited": "", "retrieved_sections": "", "answer": "",
                })
                print(f"[ERROR] {qid} (run {run}): {result['error'][:120]}")
                continue

            r_pass, r_fail = score_retrieval(question, result)
            if args.retrieval_only:
                a_pass, a_fail, cited = True, [], set()
            else:
                a_pass, a_fail, cited = score_answer(question, result)

            tally[qid]["runs"] += 1
            tally[qid]["retrieval"] += int(r_pass)
            tally[qid]["answer"] += int(a_pass)

            rows.append({
                "id": qid,
                "run": run,
                "retrieval_pass": r_pass,
                "answer_pass": a_pass,
                "retrieval_failures": "; ".join(r_fail),
                "answer_failures": "; ".join(a_fail),
                "cited": ", ".join(sorted(cited)),
                "retrieved_sections": ", ".join(
                    dict.fromkeys(s for s in result["retrieved_sections"] if s)
                ),
                "answer": result["answer"][:1500],
            })

            status = "PASS" if (r_pass and a_pass) else "FAIL"
            print(f"[{status}] {qid} (run {run})")
            for failure in r_fail + a_fail:
                print(f"         - {failure}")
            if args.verbose and result["answer"]:
                print(f"         > {result['answer'][:300]}")

    print("\n" + "=" * 64)
    if errors:
        print(f"ERRORS: {errors} run(s) failed before scoring. "
              "Accuracy below excludes them and is NOT a full result.")
    scored = {q: t for q, t in tally.items() if t["runs"]}
    if not scored:
        sys.exit("No runs completed successfully — nothing to score.")
    n = len(scored)
    retrieval_rate = sum(t["retrieval"] / t["runs"] for t in scored.values()) / n
    print(f"Questions: {n}   Runs each: {args.runs}")
    print(f"Retrieval accuracy: {retrieval_rate:.1%}")
    if not args.retrieval_only:
        answer_rate = sum(t["answer"] / t["runs"] for t in scored.values()) / n
        print(f"Answer accuracy:    {answer_rate:.1%}")
        flaky = [q for q, t in scored.items() if 0 < t["answer"] < t["runs"]]
        if flaky:
            print(f"\nFLAKY (inconsistent across runs): {', '.join(flaky)}")
            print("Non-deterministic questions are the ones to fix first — an "
                  "intermittently wrong legal answer is worse than a consistently "
                  "wrong one, because spot-checking will miss it.")
    print("=" * 64)
    print(f"\nThese figures describe {n} questions only. They say nothing about "
          "questions outside the bank.")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()