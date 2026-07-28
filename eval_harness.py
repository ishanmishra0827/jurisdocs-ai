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
]


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
    else:
        if refused:
            failures.append("model refused on an answerable question")

        ok, missing = all_match(question.get("require_answer_regex", []), answer)
        for pattern in missing:
            failures.append(f"answer missing required /{pattern}/")

    for pattern in question.get("forbid_answer_regex", []):
        if re.search(pattern, answer, re.IGNORECASE):
            failures.append(f"answer contains forbidden /{pattern}/")

    # Citation precision: every section the model cites must be one we expected
    # or one that was actually retrieved. Anything else is fabricated.
    allowed = set(question.get("expect_sections", [])) | set(
        s for s in result["retrieved_sections"] if s
    )
    fabricated = cited - allowed
    if fabricated:
        failures.append(f"fabricated citation(s): {sorted(fabricated)}")

    # Subsection-level citation, e.g. "24.005(b)". Soft by default because it is
    # the hardest thing to get right; set strict_subsection: true to enforce.
    expected_sub = question.get("expect_subsection")
    if expected_sub:
        escaped = re.escape(expected_sub)
        if not re.search(escaped, answer):
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
        token = os.environ.get("HF_TOKEN", "")
        if not token:
            sys.exit("HF_TOKEN not set. Export it, or pass --retrieval-only.")
        if token in ("hf_your_token", "hf_...", "hf_your_token_here"):
            sys.exit(f"HF_TOKEN is set to the placeholder '{token}'. "
                     "Export your real HuggingFace token.")
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
        chain = rag_core.build_chain()

        # Preflight: one throwaway call. Without this, an auth failure produces
        # 100% "failures" that look like model errors, and you spend an evening
        # debugging a pipeline that was fine.
        print("Preflight: testing inference endpoint...")
        try:
            chain.invoke({"input": "Reply with OK.", "context": "none", "chat_history": "none"})
            print("  endpoint reachable\n")
        except Exception as exc:
            sys.exit(f"\nInference endpoint unreachable — aborting before scoring.\n"
                     f"  {type(exc).__name__}: {exc}\n\n"
                     "Check that HF_TOKEN is valid and has Inference API permission.")

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
                    result = rag_core.answer_question(index, chain, question["question"])
                    result["error"] = None
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
