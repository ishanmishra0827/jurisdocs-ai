#!/usr/bin/env python3
"""
parse_catalog.py — turn a Prosper ISD course catalog PDF into structured records.

Why parse instead of chunk:

Character-based chunking splits wherever the counter runs out, which routinely
separates a course from its prerequisites. In the statute version of this
project that exact failure made the model announce that a rule did not exist
when it was sitting one chunk away. Here the equivalent mistake is telling a
student a course has no prerequisite when it does — and they enroll, and it is
wrong in a way that costs them a semester.

Each course in this catalog is already a record with labelled fields, so one
course becomes one chunk and that failure is impossible by construction.

Run:
    python parse_catalog.py documents/pisd/2026-2027-High-School-Academic-Course-Guide-and-Catalog.pdf
"""

import re
import sys
import json
from langchain_community.document_loaders import PyPDFLoader


# Course codes: EN3210, EN431A, PASSA/B, E5105A/B, MA2100
# Must contain a digit (EN3210) or an /X suffix (PASSA/B). Without this,
# roman numerals ("Honors Chinese III") and prefixes ("ENGL 1301") are
# mistaken for course codes.
CODE = r"(?:[A-Z][A-Z0-9]{2,8}/[A-Z]|[A-Z]{1,6}\d[A-Z0-9]{0,6}(?:/[A-Z])?)"

HEADER = re.compile(
    r"(?P<name>[A-Z][^\n]{2,170}?)\s+"
    r"(?P<codes>(?:" + CODE + r"\s+){1,5})"
    r"Grade:?\s*(?P<grade>\d{1,2}(?:\s*[-–]\s*\d{1,2})?)"
)

FIELD_CREDIT = re.compile(r"Credits?\s*:?\s*(?P<v>[\d.]+(?:\s*/\s*[\d.]+)?)", re.I)
FIELD_GPA = re.compile(r"GPA\s*:?\s*(?P<v>[A-Za-z/\-+ ]{2,20}?)(?=\s+(?:Prereq|Coreq|Grade|Credit)|\s*$)", re.I)
# Field labels vary across the catalog: "Credit : 1", "Credits: 2",
# "Pre-requisite or Corequisite:". Match all of them.
_STOP = (
    r"(?=\s+Co-?requisites?\s*:|\s+Pre-?requisites?\s*:|\s+Recommended\b|"
    r"\s+Grade\s*:|\s+Credits?\s*:|\s+GPA\s*:|$)"
)
FIELD_PREREQ = re.compile(
    r"Pre-?requisites?(?:\s+or\s+Co-?requisites?)?\s*:?\s*(?P<v>.*?)" + _STOP,
    re.I | re.S,
)
FIELD_COREQ = re.compile(
    r"Co-?requisites?\s*:?\s*(?P<v>.*?)" + _STOP, re.I | re.S
)


def cut_at_description(value: str) -> str:
    """
    Separate a field value from the course description that follows it.

    There is no delimiter between them, and two PDF quirks make a naive cut
    wrong in opposite directions. Some values are split across lines one word at
    a time ("Prerequisite:\n\nDance\n\nI") so stopping at the first newline
    yields "Dance" instead of "Dance I". Others sit on one line with the
    description immediately after ("Algebra I, Geometry and Algebra II\nAP
    Precalculus centers on...") so NOT stopping swallows the paragraph.

    What separates them is length. Prerequisites are short — a course name or a
    short list. Description sentences are long. So keep newline-separated
    segments until one is clearly prose.

    Truncation here is worse than failure: "Dance" reads as a complete
    prerequisite, and a student cannot tell the "I" is missing.
    """
    segments = [s.strip() for s in re.split(r"\n+", value) if s.strip()]
    kept = []
    for seg in segments:
        # A bare number is a page footer; everything after it belongs to the
        # next page, including section headings like "Orchestra".
        if re.fullmatch(r"\d{1,3}", seg):
            break
        if kept and len(seg) > 40:
            break
        kept.append(seg)
        if len(" ".join(kept)) > 200:
            break
    return re.sub(r"\s+", " ", " ".join(kept)).strip(" .;,")


def normalize(text: str) -> str:
    """
    The PDF extracts with doubled spaces between every word and hard-wraps
    mid-value ("(ENGL \\n1302)"). Both wreck keyword matching and make field
    regexes unreliable.
    """
    text = text.replace("\u00a0", " ").replace("\u2019", "'").replace("\u2013", "-")
    text = re.sub(r"-\n(\w)", r"-\1", text)          # rejoin hyphen line breaks
    text = re.sub(r"\n(?=[a-z0-9)])", " ", text)      # rejoin mid-value wraps
    text = re.sub(r"[ \t]{2,}", " ", text)            # collapse doubled spaces
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_fields(tail: str):
    """Pull labelled fields out of the text following a course header."""
    out = {}

    m = FIELD_CREDIT.search(tail)
    out["credit"] = m.group("v").strip() if m else ""

    m = FIELD_GPA.search(tail)
    out["gpa"] = m.group("v").strip() if m else ""

    m = FIELD_PREREQ.search(tail)
    out["prerequisite"] = cut_at_description(m.group("v")) if m else ""

    m = FIELD_COREQ.search(tail)
    out["corequisite"] = cut_at_description(m.group("v")) if m else ""

    # Description is whatever remains once the labelled fields are removed.
    # Cutting at the last field's end loses text when a field regex stopped
    # early, so strip the fields out instead of slicing past them.
    # Remove each field's LABEL and its trimmed VALUE only. The prereq/coreq
    # patterns run to end-of-tail, so deleting the whole match would take the
    # description with it.
    body = tail
    for pattern, key in ((FIELD_CREDIT, "credit"), (FIELD_GPA, "gpa"),
                         (FIELD_PREREQ, "prerequisite"), (FIELD_COREQ, "corequisite")):
        m = pattern.search(body)
        if not m:
            continue
        value = out.get(key, "")
        consumed = m.end()
        if key in ("prerequisite", "corequisite") and value:
            vstart = body.find(value, m.start())
            if vstart > -1:
                consumed = vstart + len(value)
        body = body[:m.start()] + " " + body[consumed:]
    body = re.sub(r"\b(Grade|Credit|GPA):?\s*[\w./\- ]{0,20}", " ", body)
    out["description"] = re.sub(r"\s+", " ", body).strip()[:1500]
    return out




# A course code has a digit (EN3210, E7005A/B) or an A/B suffix (PASSA/B).
# Requiring one of those keeps "ENGL" in "ENGL 1301" from being read as a code.
COURSE_CODE = re.compile(r"(?:[A-Z][A-Z0-9]{2,8}/[A-Z]|[A-Z]{1,6}\d[A-Z0-9]{0,6}(?:/[A-Z])?)")
NAME_CODE_PAIR = re.compile(
    r"(?P<n>[A-Za-z][^\n]*?)\s+(?P<c>(?:[A-Z][A-Z0-9]{2,8}/[A-Z]|[A-Z]{1,6}\d[A-Z0-9]{0,6}(?:/[A-Z])?))(?=\s|$)"
)


def split_inline_courses(name: str, codes_blob: str):
    """
    Ensembles are listed as a single run sharing one field block:

        Concert Band I E7005A/B Concert Band II E7006A/B
        Concert Band III E7007A/B Concert Band IV E7008A/B  Grade: 9-12 ...

    Parsed as one record, the name becomes a wall of text and only the last
    code survives — so a student searching "Concert Band II" gets nothing.
    Split into one record per name/code pair, each inheriting the shared fields.
    """
    combined = (name + " " + codes_blob).strip()
    pairs = [(m.group("n").strip(" :-"), m.group("c")) for m in NAME_CODE_PAIR.finditer(combined)]
    if len(pairs) < 2:
        return None
    cleaned = [(n, c) for n, c in pairs if n and len(n) < 70]
    return cleaned if len(cleaned) >= 2 else None


def looks_like_course(name: str) -> bool:
    """
    Reject description prose that happens to match the header pattern, e.g.
    "AP Computer Science A introduces students to computer science ... 50".
    Those end in a stray page number and run far longer than a real title.
    """
    if len(name) > 90:
        return False
    if re.search(r"\s\d{1,3}$", name):
        return False
    return True


def inherit_grouped_fields(records):
    """
    Catalogs list related courses as a run of headers sharing one field block:

        Debate II  DEBA2A/B  Grade: 10-12
        Debate III DEBA3A/B  Grade: 11-12
        Debate IV  DEBA4A/B  Grade: 12
        Credit: 1  GPA: Honors  Prerequisite: Debate I

    Parsed naively, Debate II and III come out with no prerequisite at all —
    which would tell a student the course has none when it requires Debate I.
    Each field block applies backwards to every header in its run.
    """
    for i, rec in enumerate(records):
        incomplete = not rec["credit"] and not rec["prerequisite"]
        if not (incomplete and len(rec["description"]) < 60):
            continue
        for j in range(i + 1, min(i + 5, len(records))):
            donor = records[j]
            if donor["credit"] or donor["prerequisite"]:
                for field in ("credit", "gpa", "prerequisite", "corequisite", "description"):
                    if not rec[field]:
                        rec[field] = donor[field]
                rec["grouped_with"] = donor["name"]
                break
    return records


def parse_catalog(pdf_path: str):
    pages = PyPDFLoader(pdf_path).load()
    page_starts = []
    buf = []
    pos = 0
    for page in pages:
        text = normalize(page.page_content)
        page_starts.append((pos, page.metadata.get("page", 0)))
        buf.append(text)
        pos += len(text) + 1
    full = "\n".join(buf)

    def page_for(offset):
        num = 0
        for start, p in page_starts:
            if start <= offset:
                num = p
            else:
                break
        return num + 1

    matches = list(HEADER.finditer(full))
    records = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        tail = full[m.end():end]

        codes = [c for c in re.split(r"\s+", m.group("codes").strip()) if c]
        name = re.sub(r"\s+", " ", m.group("name")).strip(" :-")

        rec = {
            "name": name,
            "codes": codes,
            "grade": re.sub(r"\s*", "", m.group("grade")),
            "page": page_for(m.start()),
        }
        if not looks_like_course(name):
            continue

        rec.update(split_fields(tail))

        inline = split_inline_courses(m.group("name"), m.group("codes"))
        if inline:
            for sub_name, sub_code in inline:
                sub = dict(rec)
                sub["name"] = sub_name
                sub["codes"] = [sub_code]
                sub["listed_with"] = [n for n, _ in inline]
                records.append(sub)
        else:
            records.append(rec)

    records = inherit_grouped_fields(records)
    return records, full


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python parse_catalog.py <catalog.pdf>")
    path = sys.argv[1]

    records, full = parse_catalog(path)
    print(f"characters after normalization : {len(full):,}")
    print(f"course records parsed          : {len(records)}")

    missing_prereq = sum(1 for r in records if not r["prerequisite"])
    missing_credit = sum(1 for r in records if not r["credit"])
    no_desc = sum(1 for r in records if len(r["description"]) < 40)
    print(f"  missing prerequisite field   : {missing_prereq}")
    print(f"  missing credit field         : {missing_credit}")
    print(f"  very short/absent description: {no_desc}")
    grouped = sum(1 for r in records if r.get("grouped_with"))
    print(f"  fields inherited from a group : {grouped}")

    print("\n" + "=" * 70)
    print("SAMPLE RECORDS")
    print("=" * 70)
    step = max(1, len(records) // 5)
    for rec in records[::step][:5]:
        print(f"\nNAME  : {rec['name']}")
        print(f"CODES : {rec['codes']}")
        print(f"GRADE : {rec['grade']}   CREDIT: {rec['credit']}   GPA: {rec['gpa']}")
        print(f"PREREQ: {rec['prerequisite'][:160]}")
        print(f"COREQ : {rec['corequisite'][:120]}")
        print(f"PAGE  : {rec['page']}")
        print(f"DESC  : {rec['description'][:180]}...")

    with open("catalog_records.json", "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"\nWrote {len(records)} records to catalog_records.json")
    print("\nCheck a few against the PDF before trusting any of it.")


if __name__ == "__main__":
    main()