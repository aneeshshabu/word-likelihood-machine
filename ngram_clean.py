#!/usr/bin/env python3
"""
ngram_clean.py - remove junk from text before ngram.py learns from it.

    python ngram_clean.py pol_500K4chan.csv -o pol_clean.txt
    python ngram_clean.py "wheelers np.txt" "wheelers dadh.pdf" -o wheeler_clean.txt
    python ngram_clean.py book.pdf --preview          # before/after samples, nothing written

Two kinds of cleaning (picked automatically, or with --mode):

  forum  (.csv files, or --mode forum)
     only the text column of the CSV; HTML tags and codes (&#039; -> ');
     reply links (>>467552285); greentext/quoted lines (>like this);
     "X replied to", "Quote from", "Originally posted by", [quote] blocks;
     links, emails, @mentions, "Sent from my iPhone", "Edit:", timestamps

  book   (.txt / .pdf files, or --mode book)
     figure/table captions; page headers and footers (lines that repeat);
     page numbers and number-only lines; headings and broken table cells;
     "(see Figure 10-17)", "[12]", "(Smith et al., 2003)", footnote numbers;
     words split across lines with a hyphen; then the text is re-joined
     and split into sentences

Both then keep only lines that look like real sentences.

Output: one sentence per line, a blank line between posts/documents.
(ngram.py treats a blank line as "new document": no patterns across it.)
"""
import argparse
import csv
import html
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

WORD_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")
ALPHA_RE = re.compile(r"^[^\W\d_]+(?:['’][^\W\d_]+)*$")     # a word made of letters


# --------------------------------------------------------------------------
# Reading files
# --------------------------------------------------------------------------
def pick_text_column(header, rows):
    """The column whose values are longest and most word-like."""
    best, best_score = 0, -1
    for c in range(len(header)):
        vals = [r[c] for r in rows if c < len(r)]
        if not vals:
            continue
        score = sum(len(re.findall(r"[A-Za-z]{2,}", v)) for v in vals) / len(vals)
        if score > best_score:
            best, best_score = c, score
    return best


def read_csv_texts(path, column=None):
    """Every row's text from the CSV's text column (auto-detected)."""
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(1 << 16)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        header = next(reader, [])
        head_rows = []
        for row in reader:
            head_rows.append(row)
            if len(head_rows) >= 300:
                break
        if column is not None:
            col = header.index(column) if column in header else int(column)
        else:
            col = pick_text_column(header, head_rows)
        # was the first line really a header? (a header has no long text)
        texts = []
        if header and col < len(header) and len(header[col]) > 60:
            texts.append(header[col])
        for row in head_rows:
            if col < len(row):
                texts.append(row[col])
        for row in reader:
            if col < len(row):
                texts.append(row[col])
    return texts, (header[col] if col < len(header) else f"column {col}")


def read_parts(path, column=None):
    """(kind, [texts]) for a file. CSV -> one text per row; others -> one text."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        texts, _ = read_csv_texts(p, column)
        return "forum", texts
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(p))
        return "book", ["\n".join((page.extract_text() or "") for page in reader.pages)]
    return "book", [p.read_text(encoding="utf-8", errors="replace")]


# --------------------------------------------------------------------------
# Shared: does this look like a real sentence?
# --------------------------------------------------------------------------
def load_blocklist(path):
    if not path:
        return set()
    words = set()
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    return words


def sentence_ok(s, stats, min_words, need_capital, blocklist, grammar=False):
    toks = WORD_RE.findall(s)
    words = [t for t in toks if t[0].isalnum() or t[0] == "_"]
    if len(words) < min_words:
        stats["dropped: too short"] += 1
        return False
    alpha = sum(1 for w in words if ALPHA_RE.match(w))
    if alpha / len(words) < 0.7:
        stats["dropped: mostly numbers/codes"] += 1
        return False
    if max(len(w) for w in words) > 30 or re.search(r"(\w)\1{4,}", s):
        stats["dropped: gibberish/spam"] += 1
        return False
    if len(words) >= 6 and len({w.lower() for w in words}) / len(words) < 0.4:
        stats["dropped: repetitive"] += 1
        return False
    punct = len(toks) - len(words)
    if punct > len(words):
        stats["dropped: mostly symbols"] += 1
        return False
    if need_capital and not re.match(r"[\"'(“‘]?[A-Z0-9]", s):
        stats["dropped: not a sentence start"] += 1
        return False
    if blocklist and any(w.lower() in blocklist for w in words):
        stats["dropped: blocklist"] += 1
        return False
    if grammar:
        import grammatizator
        if any(i.severity == "error" for i in grammatizator.check(s, warnings=False)):
            stats["dropped: grammar errors"] += 1
            return False
    return True


# protect abbreviations so "e.g. the" doesn't count as a sentence end
ABBREV = ["e.g.", "i.e.", "etc.", "vs.", "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.", "Fig.", "Figs.",
          "No.", "approx.", "al.", "cf.", "Jr.", "Sr.", "Inc.", "Ltd.", "Co.", "U.S.", "a.m.", "p.m."]
SENT_SPLIT = re.compile(r"(?<=[.!?])[\"'”’)]*\s+(?=[\"'“‘(]?[A-Z0-9])")


def split_sentences(text):
    t = text
    for a in ABBREV:
        t = t.replace(a, a.replace(".", "\x00"))
    return [s.replace("\x00", ".").strip() for s in SENT_SPLIT.split(t) if s.strip()]


# --------------------------------------------------------------------------
# Forum posts
# --------------------------------------------------------------------------
FORUM_DROP_LINE = re.compile(
    r"""^\s*(?:
        .{0,60}\b(?:replied\ to|wrote|said|posted)\s*:?\s*$     # "John replied to" / "Bob wrote:"
      | (?:quote|quoting)\b.*$                                  # "Quote from: ...", "Quote:"
      | originally\ posted\ by\b.*$
      | (?:last\ )?edit(?:ed)?(?:\ by\b.*|\s*\d*\s*:.*)$          # "Edit:", "Edited by ..."
      | sent\ from\ my\b.*$                                     # phone signatures
      | (?:click\ to\ expand|show\ more|reply|share|report|like|quote|permalink)\W*$
      | reply\ \#?\d+\b.*$
      | posted\ (?:on|by|at)\b.*$
      | (?:anonymous|anon)\s*\d*\s*(?:no\.?\s*\d+)?.*\d{1,2}:\d{2}.*$
      | (?:[-_=*~#]{3,}).*$                                     # separators / signature lines
    )""", re.I | re.X)
TIMESTAMP = re.compile(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}(?:\s*\(\w{2,3}\))?(?:\s*\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?)?",
                       re.I)
URL = re.compile(r"(?:https?://|www\.)\S+|\b\S+\.(?:com|org|net|io|gov|edu|co|uk|in|de|ru)(?:/\S*)?\b", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
MENTION = re.compile(r"(?<!\w)@\w+")
REPLY_REF = re.compile(r">>\s*\d+|>>>/\w+/\d*")
BBQUOTE = re.compile(r"\[quote[^\]]*\].*?\[/quote\]", re.I | re.S)
BBTAG = re.compile(r"\[/?(?:b|i|u|s|url|img|color|size|spoiler|code|font|center|list|\*)[^\]]*\]", re.I)
IMG_NAME = re.compile(r"\b\S+\.(?:jpe?g|png|gif|webm|mp4|webp)\b", re.I)


def clean_forum_post(raw, stats, keep_quotes=False, blocklist=frozenset(), grammar=False):
    """One post (may contain HTML) -> cleaned sentences joined by newlines ('' if nothing left)."""
    s = raw
    if "<" in s:
        n = len(re.findall(r'class="(?:quotelink|deadlink)"', s))
        if n:
            stats["removed: reply links (>>123)"] += n
        s = re.sub(r'<a [^>]*class="quotelink"[^>]*>.*?</a>', " ", s, flags=re.S)
        s = re.sub(r'<span class="deadlink">.*?</span>', " ", s, flags=re.S)
        s = re.sub(r"<wbr\s*/?>", "", s)
        s = re.sub(r"<br\s*/?>|</p>|</div>|</li>", "\n", s, flags=re.I)
        s = re.sub(r"<[^>]+>", " ", s)
        stats["cleaned: posts with HTML"] += 1
    if "&" in s:
        s = html.unescape(s)
    if "[quote" in s.lower():
        s = BBQUOTE.sub(" ", s)
        stats["removed: [quote] blocks"] += 1
    s = BBTAG.sub(" ", s)
    s = REPLY_REF.sub(" ", s)

    out = []
    for line in s.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">") and not keep_quotes:
            stats["removed: quoted/greentext lines"] += 1
            continue
        line = line.lstrip(">").strip()
        if FORUM_DROP_LINE.match(line):
            stats["removed: forum boilerplate lines"] += 1
            continue
        for pat, name in ((URL, "links"), (EMAIL, "emails"), (MENTION, "@mentions"),
                          (TIMESTAMP, "dates/times"), (IMG_NAME, "file names")):
            line, n = pat.subn(" ", line)
            if n:
                stats[f"removed: {name}"] += n
        line = re.sub(r"\s+", " ", line).strip()
        for sent in split_sentences(line):
            if sentence_ok(sent, stats, 3, False, blocklist, grammar):
                out.append(sent)
                stats["kept: sentences"] += 1
    return "\n".join(out)


# --------------------------------------------------------------------------
# Books / textbooks / PDFs
# --------------------------------------------------------------------------
CAPTION = re.compile(r"^\s*(?:[A-Z]{1,4}\s*(?:\(\d+\))?,?\s*[A-Z]?\s+)?"
                     r"(?:Figure|Fig\.|Figs?\.|Table|Box|Plate|Chart|Diagram|Exhibit|Graph|Illustration|Map|Photo)"
                     r"\s*[A-Z]?\d+(?:[-.–:]\d+)*[A-Za-z]?\b", re.I)
SOURCE_LINE = re.compile(r"^\s*(?:Source|From|Courtesy of|Reprinted|Adapted|Modified|Redrawn)\b.{0,200}$", re.I)
HEADER = re.compile(r"^\s*(?:(?:chapter|section|part|unit|appendix)\s+[\dIVXLC]+\b.*|\d{1,4}\s+[A-Z][^.!?]{0,70}"
                    r"|[A-Z][^.!?]{0,70}\s+\d{1,4}|page\s+\d+(?:\s+of\s+\d+)?)\s*$", re.I)
INLINE_REF = re.compile(r"\s*\((?:see\s+(?:also\s+)?|cf\.\s+|as\s+in\s+|e\.g\.,?\s+)?"
                        r"(?:Figures?|Figs?\.|Tables?|Boxes?|Chapters?|Plates?|Sections?|Appendix|Eq\.|Equation)"
                        r"[^()]{0,80}\)", re.I)
CITATION = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]"
                      r"|\s*\((?:[A-Z][A-Za-z'-]+(?:\s+(?:et\s+al\.|and|&)\s*[A-Z]?[A-Za-z'-]*)?,?\s+\d{4}[a-z]?;?\s*)+\)")
FOOTNOTE = re.compile(r"(?<=[a-z\)][.,;:])\d{1,3}(?=\s|$)|(?<=[a-z]{3})\d{1,2}(?=[.,;:]?\s+[A-Z])")


BULLET = re.compile(r"^\s*(?:[•▪●◦■►✓✔*–—-]|\(?\d{1,2}[.)]|\(?[a-z][.)])\s+")
ENUM = re.compile(r"(?<![\w(])\((?:\d{1,2}|[a-z]|[ivx]{1,4})\)\s*")
FIG_MENTION = re.compile(r"\b(?:Figures?|Figs?\.|Tables?|Plates?)\s+\d", re.I)
# reference-list entries: "Lyons H: Title, J Am Dent Assoc 34:647, 1947."
BIBLIO = re.compile(r"\b\d{1,4}\s*(?:\(\d+\))?:\s*\d{1,5}(?:\s*[-–]\s*\d+)?,?\s*(?:19|20)\d{2}\b"
                    r"|^\(?(?:From |Modified from |Adapted from |In )?[A-Z][A-Za-z'-]+(?: [A-Z][A-Za-z'-]+)? "
                    r"[A-Z]{1,3}(?:,| and |:| et al)"
                    r"|\bet al\b.*\b(?:19|20)\d{2}\b|\b(?:ed|eds|editor|editors)\b[.:,].*\b(?:19|20)\d{2}\b")
SMALL_WORDS = {"a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "from", "with", "at", "by", "vs"}


def is_heading(line):
    """ALL CAPS lines, or short Title Case lines with no full stop."""
    if re.search(r"[.!?,;:]$", line):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", line)
    if not words:
        return False
    letters = "".join(words)
    if len(words) >= 2 and sum(c.isupper() for c in letters) > 0.8 * len(letters):
        return True
    big = [w for w in words if w.lower() not in SMALL_WORDS]
    return len(words) <= 10 and len(big) >= 2 and all(w[0].isupper() for w in big)


def book_profile(text):
    """Facts about the whole document that line-cleaning needs."""
    lines = [l.strip() for l in text.split("\n")]
    nonempty = [l for l in lines if l]
    lengths = sorted(len(l) for l in nonempty) or [60]
    counts = Counter(l for l in nonempty if len(l) < 90)
    repeated = {l for l, c in counts.items() if c >= 3}
    words = set(re.findall(r"[a-z]{3,}", text.lower()))
    return {"median": lengths[len(lengths) // 2], "repeated": repeated, "words": words}


def normalize(text):
    text = unicodedata.normalize("NFKC", text)          # ligatures (ﬁ), odd spaces, fullwidth
    text = text.replace("­", "")                    # soft hyphen
    text = re.sub(r"[ -​  　\xa0]", " ", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_book_text(text, stats, profile=None, blocklist=frozenset(), grammar=False):
    text = normalize(text)
    profile = profile or book_profile(text)
    short = max(18, 0.45 * profile["median"])
    lines = text.split("\n")

    kept, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            kept.append("")                              # paragraph break
            continue
        if CAPTION.match(line) or SOURCE_LINE.match(line):
            stats["removed: figure/table captions"] += 1
            extra = 0                                    # caption continues until a line ends a sentence
            while (extra < 6 and i < len(lines) and lines[i].strip()
                   and not re.search(r"[.!?)]\s*$", (lines[i - 1] if extra else line).strip())):
                i += 1
                extra += 1
            continue
        if line in profile["repeated"] and not re.search(r"[a-z][.!?,;:]$", line):
            stats["removed: repeated headers/footers"] += 1
            continue
        if not re.search(r"[A-Za-z]{3,}", line):
            stats["removed: number/symbol lines"] += 1
            continue
        if is_heading(line):
            stats["removed: page headers/headings"] += 1
            continue
        if HEADER.match(line) and len(line) < 80 and not re.search(r"[.!?,;:]$", line):
            stats["removed: page headers/headings"] += 1
            continue
        if len(line) < short and not re.search(r"[.!?:;,\"'”)\]]$", line) and not line.endswith("-"):
            stats["removed: headings/table cells"] += 1
            continue
        if BULLET.match(line):                           # bullets / "1." start a new paragraph
            line = BULLET.sub("", line)
            kept.append("")
            stats["cleaned: bullets/numbering"] += 1
        kept.append(line)

    # re-join lines into running text; mend words broken with a hyphen
    paras, cur = [], ""
    for line in kept:
        if not line:
            if cur:
                paras.append(cur)
            cur = ""
            continue
        if cur.endswith("-") and line[:1].islower():
            head = re.search(r"(\w+)-$", cur)
            tail = re.match(r"\w+", line)
            if head and tail and (head.group(1) + tail.group(0)).lower() in profile["words"]:
                cur = cur[:-1] + line                     # "cervico-" + "incisal" -> one word
                stats["mended: hyphenated line breaks"] += 1
            else:
                cur = cur + line                          # keep real hyphen: "three-" + "cusp"
        else:
            cur = (cur + " " + line) if cur else line
    if cur:
        paras.append(cur)

    out = []
    for p in paras:
        for pat, name in ((INLINE_REF, "(see Figure ...) references"), (CITATION, "citations"),
                          (URL, "links"), (EMAIL, "emails"), (FOOTNOTE, "footnote numbers")):
            p, n = pat.subn("", p)
            if n:
                stats[f"removed: {name}"] += n
        p = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"\s+", " ", p)).strip()
        p, n = ENUM.subn("", p)
        if n:
            stats["removed: (1) (a) enumerators"] += n
        sents = []
        for s in split_sentences(p):
            if FIG_MENTION.search(s):
                stats["dropped: refers to a figure/table"] += 1
            elif BIBLIO.search(s):
                stats["dropped: bibliography entries"] += 1
            elif s[-1:] in ".!?\"'”)" and sentence_ok(s, stats, 4, True, blocklist, grammar):
                sents.append(s)
        stats["kept: sentences"] += len(sents)
        if sents:
            out.append("\n".join(sents))
    return "\n".join(out)


# --------------------------------------------------------------------------
# Jobs (also used by ngram.py build, in worker processes)
# --------------------------------------------------------------------------
def clean_job(job):
    """job = (kind, [texts], options) -> ([cleaned texts], Counter)"""
    kind, texts, opts = job
    stats = Counter()
    blocklist = opts.get("blocklist") or frozenset()
    if kind == "forum":
        res = [clean_forum_post(t, stats, opts.get("keep_quotes", False), blocklist, opts.get("grammar"))
               for t in texts]
        stats["posts read"] += len(texts)
    else:
        res = [clean_book_text(t, stats, blocklist=blocklist, grammar=opts.get("grammar")) for t in texts]
        stats["documents read"] += len(texts)
    return [r for r in res if r], stats


def make_jobs(kind, texts, opts, target=1 << 21):
    """Group many small posts into ~2 MB jobs so all CPU threads share the work."""
    if kind != "forum":
        return [(kind, [t], opts) for t in texts]
    jobs, cur, size = [], [], 0
    for t in texts:
        cur.append(t)
        size += len(t)
        if size >= target:
            jobs.append((kind, cur, opts))
            cur, size = [], 0
    if cur:
        jobs.append((kind, cur, opts))
    return jobs


def print_report(stats, chars_in, chars_out, out=sys.stdout):
    print("\nWhat the cleaner did:", file=out)
    for k in sorted(stats, key=lambda k: (not k.startswith("posts") and not k.startswith("doc"),
                                          not k.startswith("kept"), k)):
        print(f"  {k:<42} {stats[k]:>12,}", file=out)
    if chars_in:
        print(f"  {'text kept':<42} {chars_out / 1e6:>10.1f} M chars of {chars_in / 1e6:.1f} M "
              f"({100 * chars_out / chars_in:.0f}%)", file=out)


# --------------------------------------------------------------------------
# Preview: show before/after so you can trust (or tune) the cleaner
# --------------------------------------------------------------------------
def preview(path, mode="auto", n=6, keep_quotes=False, seed=None, column=None, grammar=False, blocklist=frozenset()):
    """List of (before, after) samples."""
    kind, texts = read_parts(path, column)
    if mode != "auto":
        kind = mode
    rng = random.Random(seed)
    stats = Counter()
    pairs = []
    if kind == "forum":
        picks = [t for t in rng.sample(texts, min(len(texts), n * 6)) if t.strip()]
        picks.sort(key=lambda t: -(("<" in t) + (">" in t)))      # show interesting ones first
        for t in picks[:n]:
            pairs.append((t, clean_forum_post(t, stats, keep_quotes, blocklist, grammar)))
    else:
        full = normalize(texts[0])
        prof = book_profile(full)
        lines = full.split("\n")
        for _ in range(n):
            start = rng.randrange(max(1, len(lines) - 22))
            window = "\n".join(lines[start:start + 22])
            pairs.append((window, clean_book_text(window, stats, prof, blocklist, grammar)))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Remove junk from text before building a model.")
    ap.add_argument("inputs", nargs="+", help=".csv / .txt / .pdf files")
    ap.add_argument("-o", "--output", default="cleaned.txt")
    ap.add_argument("--mode", choices=["auto", "forum", "book"], default="auto",
                    help="auto: .csv = forum, .txt/.pdf = book")
    ap.add_argument("--column", help="CSV column name or number holding the text (default: guess)")
    ap.add_argument("--keep-quotes", action="store_true", help="keep >quoted / greentext lines")
    ap.add_argument("--blocklist", help="text file, one word per line: drop sentences containing them")
    ap.add_argument("--grammar-filter", action="store_true",
                    help="also drop sentences the grammatizator finds errors in")
    ap.add_argument("--preview", action="store_true", help="show before/after samples, write nothing")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    if args.preview:
        for path in args.inputs:
            for before, after in preview(path, args.mode, keep_quotes=args.keep_quotes, column=args.column,
                                         grammar=args.grammar_filter):
                print("=" * 70 + "\nBEFORE:\n" + before[:1500] + "\n\nAFTER:\n" + (after or "(nothing kept)"))
        return

    from multiprocessing import Pool
    import os
    opts = {"keep_quotes": args.keep_quotes, "blocklist": frozenset(load_blocklist(args.blocklist)),
            "grammar": args.grammar_filter}
    stats, chars_in, chars_out = Counter(), 0, 0
    with open(args.output, "w", encoding="utf-8") as out, Pool(args.workers or os.cpu_count()) as pool:
        for path in args.inputs:
            kind, texts = read_parts(path, args.column)
            if args.mode != "auto":
                kind = args.mode
            print(f"{path}: {len(texts):,} {'posts' if kind == 'forum' else 'document(s)'}, cleaning as {kind}")
            chars_in += sum(len(t) for t in texts)
            for res, st in pool.imap(clean_job, make_jobs(kind, texts, opts)):
                stats.update(st)
                for r in res:
                    out.write(r + "\n\n")
                    chars_out += len(r)
    print_report(stats, chars_in, chars_out)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
