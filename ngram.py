#!/usr/bin/env python3
"""
ngram.py - build your own word-pattern (n-gram) model from text, then use it as
a next-word predictor (keyboard prototype), a text generator, and a compressor.

    python ngram.py build    books/ chats.txt notes.pdf -o model.db
    python ngram.py lookup   model.db "how" --next are may
    python ngram.py predict  model.db
    python ngram.py generate model.db "i think that"
    python ngram.py compress model.db "hey, how are you doing today?"
    python ngram.py decompress model.db <code>

Levels (your naming):
    level 0 = single words and how common they are   (the "vocab" table)
    level 1 = 2-word patterns    "how"      -> "are"
    level 2 = 3-word patterns    "how are"  -> "you"
    level k = (k+1)-word patterns: k words of context, then the next word

The likelihood stored for each pattern is a conditional probability:
    P(next | context) = count(context + next) / count(context followed by anything)
"""
import argparse
import base64
import math
import os
import random
import re
import sqlite3
import sys
import time
import zlib
from collections import Counter
from multiprocessing import Pool, shared_memory
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Tokenizing: turning text into a list of words
# --------------------------------------------------------------------------
# A token is either a word (letters/digits, may contain apostrophes like
# "don't") or a single punctuation mark. Everything is lowercased so that
# "How" and "how" count as the same word.
TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")

UNK, BOUNDARY = 0, 1       # special ids: unknown word, document boundary
FIRST_WORD = 2             # real words start at id 2, most frequent first
TEXT_EXTS = {".txt", ".md", ".text", ".csv", ".srt", ".log"}


def norm(token):
    return token.lower().replace("’", "'")   # curly apostrophe -> straight


def tokenize(text):
    return [norm(m.group()) for m in TOKEN_RE.finditer(text)]


# --------------------------------------------------------------------------
# Reading input files
# --------------------------------------------------------------------------
def collect_files(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*")
                            if f.is_file() and f.suffix.lower() in TEXT_EXTS | {".pdf"})
        elif p.is_file():
            files.append(p)
        else:
            print(f"  ! skipping {item}: not found")
    return files


def _read_document(job):
    """Runs in a worker process. Returns (path, kind, [texts], error).
    A CSV gives one text per row (its text column); other files give one text."""
    path, column = job
    try:
        import ngram_clean
        kind, parts = ngram_clean.read_parts(path, column)
        return str(path), kind, parts, None
    except ImportError as e:
        if "pypdf" in str(e):
            return str(path), "book", [], "reading PDFs needs pypdf:  pip install pypdf"
        return str(path), "book", [], f"{e} (keep ngram_clean.py next to ngram.py)"
    except Exception as e:  # a broken file should not kill the whole build
        return str(path), "book", [], str(e)


def split_chunks(text, size=1 << 21):
    """Cut text into ~2 MB pieces at line/space breaks so workers can share the job."""
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind("\n", start, end)
            if cut <= start:
                cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut + 1
        chunks.append(text[start:end])
        start = end
    return chunks


# Worker functions for the two tokenizing passes ---------------------------
BLANK_LINE = re.compile(r"\n[ \t\r\f\v]*\n")      # a blank line separates posts/paragraphs


def _count_words(chunk):
    c = Counter()
    for piece in BLANK_LINE.split(chunk):
        c.update(tokenize(piece))
    return c


_VOCAB = None


def _init_encoder(vocab):
    global _VOCAB
    _VOCAB = vocab


def _encode(chunk):
    get = _VOCAB.get
    ids = []
    for piece in BLANK_LINE.split(chunk):
        ids.extend(get(t, UNK) for t in tokenize(piece))
        ids.append(BOUNDARY)               # patterns never cross a blank line
    return np.array(ids, dtype=np.int32)


# --------------------------------------------------------------------------
# Counting n-grams
# --------------------------------------------------------------------------
# The whole text becomes one long array of word ids. An n-gram is any n
# consecutive ids. Counting them = sorting all windows and counting runs of
# identical rows. That sort is the heavy part, and it is what runs on all CPU
# cores (or on the GPU).
#
# The work is split into "partitions" by hashing the first word(s) of each
# window. Every context lands in exactly one partition, so each partition can
# be counted, turned into probabilities and pruned completely on its own,
# with no merging step afterwards.

BLOCK = 1 << 22


def _select_rows(tok, cs, n, part, nparts):
    """All n-word windows belonging to one partition, as an (rows, n) int32 array.
    cs is a running count of special tokens, used to skip windows that touch an
    unknown word or a document boundary."""
    W = len(tok) - n + 1
    out = []
    for s in range(0, W, BLOCK):
        e = min(W, s + BLOCK)
        ok = (cs[s + n:e + n] - cs[s:e]) == 0
        if nparts > 1:
            h = tok[s:e].astype(np.int64)
            if n >= 3:                       # hash first two words (both are context)
                h = h * 1000003 + tok[s + 1:e + 1]
            ok &= (h % nparts) == part
        idx = np.nonzero(ok)[0]
        if idx.size:
            idx += s
            out.append(np.stack([tok[idx + j] for j in range(n)], axis=1))
    if not out:
        return np.zeros((0, n), np.int32)
    return np.concatenate(out)


def _count_rows_cpu(rows, V):
    """Unique rows + how often each occurs."""
    n = rows.shape[1]
    if len(rows) == 0:
        return rows, np.zeros(0, np.int64)
    if V ** n < 2 ** 62:
        # Fast path: pack each n-gram into one 64-bit number, then a plain sort.
        key = np.zeros(len(rows), np.int64)
        for j in range(n):
            key = key * V + rows[:, j]
        u, c = np.unique(key, return_counts=True)
        out = np.empty((len(u), n), np.int32)
        for j in range(n - 1, -1, -1):
            out[:, j] = u % V
            u //= V
        return out, c.astype(np.int64)
    # Too many words to pack: sort row-by-row instead.
    order = np.lexsort(rows.T[::-1])
    r = rows[order]
    new = np.ones(len(r), bool)
    new[1:] = np.any(r[1:] != r[:-1], axis=1)
    starts = np.nonzero(new)[0]
    counts = np.diff(np.append(starts, len(r)))
    return r[starts], counts.astype(np.int64)


def _count_rows_torch(rows, device):
    import torch
    t = torch.from_numpy(np.ascontiguousarray(rows)).to(device)
    u, c = torch.unique(t, dim=0, return_counts=True)
    return u.cpu().numpy().astype(np.int32), c.cpu().numpy().astype(np.int64)


def _finalize(uniq, counts, min_count, top_k):
    """Group n-grams by context, compute P(next | context), rank the next-words
    inside each context (most likely first) and drop what we don't keep."""
    M, n = uniq.shape
    k = n - 1
    if M == 0:
        return uniq, counts, np.zeros(0, np.int32), np.zeros(0, np.float32), 0
    # sort: context first, then count (descending), then word id (ties -> more common word)
    keys = [uniq[:, -1], -counts] + [uniq[:, j] for j in range(k - 1, -1, -1)]
    order = np.lexsort(keys)
    uniq, counts = uniq[order], counts[order]
    ctx = uniq[:, :k]
    new_ctx = np.ones(M, bool)
    new_ctx[1:] = np.any(ctx[1:] != ctx[:-1], axis=1)
    starts = np.nonzero(new_ctx)[0]
    group = np.cumsum(new_ctx) - 1
    totals = np.add.reduceat(counts, starts)        # how often each context was followed by anything
    rank = np.arange(M) - starts[group]
    prob = counts / totals[group]
    keep = (counts >= min_count) & (rank < top_k)
    return (uniq[keep], counts[keep], rank[keep].astype(np.int32),
            prob[keep].astype(np.float32), len(starts))


# Worker side of the CPU counting pool. The token array lives in shared memory
# so 16+ workers don't each need their own copy.
_SHM, _TOK, _CS = [], None, None


def _attach(name):
    try:
        return shared_memory.SharedMemory(name=name, track=False)   # Python 3.13+
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def _init_counter(tok_name, cs_name, length):
    global _TOK, _CS
    a, b = _attach(tok_name), _attach(cs_name)
    _SHM.extend([a, b])
    _TOK = np.ndarray((length,), np.int32, a.buf)
    _CS = np.ndarray((length + 1,), np.int32, b.buf)


def _count_task(task):
    n, part, nparts, min_count, top_k, V = task
    rows = _select_rows(_TOK, _CS, n, part, nparts)
    uniq, counts = _count_rows_cpu(rows, V)
    return (n,) + _finalize(uniq, counts, min_count, top_k)


def pick_device(choice):
    """Returns a torch device for counting, or None for the CPU pool."""
    if choice == "cpu":
        return None
    try:
        import torch
    except ImportError:
        if choice == "cuda":
            sys.exit("--device cuda needs PyTorch with CUDA (see README).")
        return None
    if choice == "torch-cpu":                 # testing aid: torch code path, no GPU
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "cuda":
        sys.exit("PyTorch is installed but can't see a CUDA GPU.")
    return None


# --------------------------------------------------------------------------
# The lookup-table file (SQLite)
# --------------------------------------------------------------------------
# One table per level, because each level has a different context length and
# the predictor asks them in order: longest context first, then shorter.
#   level_k(context TEXT, rank INT, next TEXT, count INT, prob REAL)
# Open model.db in "DB Browser for SQLite" to look around.

def create_db(path, levels):
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE vocab (id INTEGER PRIMARY KEY, word TEXT NOT NULL, "
                "count INTEGER NOT NULL, prob REAL NOT NULL)")
    for k in range(1, levels + 1):
        con.execute(f"CREATE TABLE level_{k} (context TEXT NOT NULL, rank INTEGER NOT NULL, "
                    f"next TEXT NOT NULL, count INTEGER NOT NULL, prob REAL NOT NULL, "
                    f"PRIMARY KEY (context, rank)) WITHOUT ROWID")
    return con


def write_level(con, words_arr, n, uniq, counts, rank, prob):
    k = n - 1
    if len(uniq) == 0:
        return
    cols = [words_arr[uniq[:, j]] for j in range(k)]
    ctx = cols[0] if k == 1 else [" ".join(t) for t in zip(*cols)]
    nxt = words_arr[uniq[:, k]]
    con.executemany(f"INSERT INTO level_{k} VALUES (?,?,?,?,?)",
                    zip(ctx, rank.tolist(), nxt, counts.tolist(), prob.tolist()))


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def cmd_build(args):
    t0 = time.time()
    workers = args.workers or os.cpu_count() or 1
    device = pick_device(args.device)

    def step(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    files = collect_files(args.inputs)
    if not files:
        sys.exit("No .txt/.pdf files found.")
    step(f"reading {len(files)} file(s) with {workers} worker processes")
    with Pool(workers) as pool:
        docs = pool.map(_read_document, [(f, args.column) for f in files], chunksize=1)
    for path, kind, parts, err in docs:
        if err:
            print(f"  ! {path}: {err}")
        elif kind == "forum":
            print(f"  {Path(path).name}: {len(parts):,} rows from the CSV's text column")

    if args.clean:
        import ngram_clean
        opts = {"keep_quotes": args.keep_quotes, "grammar": args.grammar_filter,
                "blocklist": frozenset(ngram_clean.load_blocklist(args.blocklist))}
        step("cleaning junk" + (" (+ grammar filter)" if args.grammar_filter else ""))
        stats, chars_in, chars_out, cleaned = Counter(), 0, 0, []
        with Pool(workers) as pool:
            for path, kind, parts, err in docs:
                kind = kind if args.clean_mode == "auto" else args.clean_mode
                chars_in += sum(len(t) for t in parts)
                out = []
                for res, st in pool.imap(ngram_clean.clean_job, ngram_clean.make_jobs(kind, parts, opts)):
                    stats.update(st)
                    out += res
                chars_out += sum(len(t) for t in out)
                cleaned.append((path, kind, out, err))
        docs = cleaned
        ngram_clean.print_report(stats, chars_in, chars_out)

    doc_chunks, n_chars = [], 0
    for path, kind, parts, err in docs:
        text = "\n\n".join(parts)            # blank line between posts = no patterns across them
        doc_chunks.append(split_chunks(text))
        n_chars += len(text)
    del docs
    chunks = [c for dc in doc_chunks for c in dc]
    step(f"{n_chars / 1e6:.1f} M characters in {len(chunks)} chunks")

    # Pass 1: count single words (level 0) to build the vocabulary.
    with Pool(workers) as pool:
        word_counts = Counter()
        for c in pool.imap_unordered(_count_words, chunks):
            word_counts.update(c)
    total_tokens = sum(word_counts.values())
    items = [(w, c) for w, c in word_counts.items() if c >= args.min_word_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    if args.max_vocab:
        items = items[:args.max_vocab]
    words = ["<unk>", "<b>"] + [w for w, _ in items]
    kept = sum(c for _, c in items)
    vocab = {w: i for i, w in enumerate(words)}
    del word_counts
    step(f"{total_tokens:,} words total, {len(items):,} distinct words kept "
         f"({100 * kept / max(total_tokens, 1):.1f}% of text covered)")

    # Pass 2: turn the text into one long array of word ids.
    with Pool(workers, initializer=_init_encoder, initargs=(vocab,)) as pool:
        arrays = pool.map(_encode, chunks, chunksize=1)
    del chunks
    pieces, i = [], 0
    for dc in doc_chunks:
        pieces += arrays[i:i + len(dc)]
        pieces.append(np.array([BOUNDARY], np.int32))   # no patterns across files
        i += len(dc)
    tokens = np.concatenate(pieces)
    del arrays, pieces, doc_chunks
    cs = np.zeros(len(tokens) + 1, np.int32)
    cs[1:] = np.cumsum(tokens < FIRST_WORD, dtype=np.int32)
    V = len(words)
    step(f"encoded {len(tokens):,} tokens")

    con = create_db(args.output, args.levels)
    con.executemany("INSERT INTO vocab VALUES (?,?,?,?)",
                    [(UNK, "<unk>", total_tokens - kept, 0.0), (BOUNDARY, "<b>", 0, 0.0)] +
                    [(i + FIRST_WORD, w, c, c / max(kept, 1)) for i, (w, c) in enumerate(items)])
    words_arr = np.array(words, dtype=object)

    tasks = []
    for k in range(1, args.levels + 1):
        n = k + 1
        W = len(tokens) - n + 1
        if W <= 0:
            continue
        if device is not None:
            nparts = max(1, math.ceil(W / args.gpu_rows))
        else:
            nparts = max(workers * 2, math.ceil(W / 4_000_000))
        tasks += [(n, p, nparts, args.min_count, args.top_k, V) for p in range(nparts)]

    stats = {k: [0, 0] for k in range(1, args.levels + 1)}   # rows kept, contexts seen

    def handle(res, done):
        n, uniq, counts, rank, prob, n_ctx = res
        write_level(con, words_arr, n, uniq, counts, rank, prob)
        stats[n - 1][0] += len(uniq)
        stats[n - 1][1] += n_ctx
        if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
            step(f"counting patterns: {done}/{len(tasks)} pieces done")

    if device is not None:
        step(f"counting on {device} (sorting happens on the device)")
        warned = False
        for done, (n, p, nparts, mc, tk, V_) in enumerate(tasks, 1):
            rows = _select_rows(tokens, cs, n, p, nparts)
            try:
                uniq, counts = _count_rows_torch(rows, device)
            except Exception as e:      # out of GPU memory etc. -> do this piece on CPU
                if not warned:
                    print(f"  ! GPU failed ({type(e).__name__}); using CPU for such pieces")
                    warned = True
                uniq, counts = _count_rows_cpu(rows, V_)
            handle((n,) + _finalize(uniq, counts, mc, tk), done)
    else:
        step(f"counting on CPU with {workers} worker processes")
        shm_t = shared_memory.SharedMemory(create=True, size=max(tokens.nbytes, 4))
        shm_c = shared_memory.SharedMemory(create=True, size=cs.nbytes)
        try:
            view = np.ndarray(tokens.shape, np.int32, shm_t.buf)
            view[:] = tokens
            view2 = np.ndarray(cs.shape, np.int32, shm_c.buf)
            view2[:] = cs
            del view, view2
            with Pool(workers, initializer=_init_counter,
                      initargs=(shm_t.name, shm_c.name, len(tokens))) as pool:
                for done, res in enumerate(pool.imap_unordered(_count_task, tasks), 1):
                    handle(res, done)
        finally:
            shm_t.close()
            shm_t.unlink()
            shm_c.close()
            shm_c.unlink()

    meta = {"levels": args.levels, "model_id": os.urandom(4).hex(), "tokens": len(tokens),
            "cleaned": (args.clean_mode if args.clean else "no") + ("+grammar" if args.clean and args.grammar_filter else ""),
            "vocab": len(items), "min_count": args.min_count, "top_k": args.top_k,
            "files": len(files), "built": time.strftime("%Y-%m-%d %H:%M")}
    con.executemany("INSERT INTO meta VALUES (?,?)", [(k, str(v)) for k, v in meta.items()])
    con.commit()

    print()
    for k, (rows, n_ctx) in stats.items():
        print(f"  level {k} ({k + 1}-word patterns): {rows:>12,} stored   "
              f"({n_ctx:,} different {k}-word contexts seen)")
    print("\n  most common 2-word patterns:")
    for ctx, nxt, c, p in con.execute(
            "SELECT context, next, count, prob FROM level_1 ORDER BY count DESC LIMIT 10"):
        print(f"    {ctx!r:>10} -> {nxt!r:<10} seen {c:>8,}x   P = {p:.3f}")
    con.close()
    size = os.path.getsize(args.output) / 1e6
    step(f"saved {args.output} ({size:.1f} MB)")


# --------------------------------------------------------------------------
# Using the model
# --------------------------------------------------------------------------
class NGramModel:
    BACKOFF = 0.4   # "stupid backoff": each level we drop costs a factor 0.4

    def __init__(self, path):
        if not os.path.exists(path):
            sys.exit(f"Model not found: {path}")
        self.con = sqlite3.connect(path)
        meta = dict(self.con.execute("SELECT key, value FROM meta"))
        self.levels = int(meta["levels"])
        self.model_id = meta["model_id"]
        self.meta = meta
        self.words, self.counts = [], []
        for _, w, c in self.con.execute("SELECT id, word, count FROM vocab ORDER BY id"):
            self.words.append(w)
            self.counts.append(c)
        self.word_id = {w: i for i, w in enumerate(self.words)}
        self.total = max(1, sum(self.counts[FIRST_WORD:]))
        self._cache = {}

    def nexts(self, level, context):
        """[(next_word, count, prob), ...] most likely first, for a context of `level` words."""
        key = (level, context)
        hit = self._cache.get(key)
        if hit is None:
            hit = self.con.execute(
                f"SELECT next, count, prob FROM level_{level} WHERE context=? ORDER BY rank",
                (context,)).fetchall()
            if len(self._cache) > 200_000:
                self._cache.clear()
            self._cache[key] = hit
        return hit

    def predict(self, history, k=5, prefix=""):
        """Top-k next words. Ask the longest context first; a word found there keeps
        that score. Words only found at shorter contexts are scored lower."""
        top = min(self.levels, len(history))
        best = {}
        for level in range(top, 0, -1):
            weight = self.BACKOFF ** (top - level)
            for w, c, p in self.nexts(level, " ".join(history[-level:])):
                if w not in best and w.startswith(prefix):
                    best[w] = (weight * p, level)
        if len(best) < k:             # level 0: just common words
            weight, added = self.BACKOFF ** top, 0
            for wid in range(FIRST_WORD, len(self.words)):
                w = self.words[wid]
                if w not in best and w.startswith(prefix):
                    best[w] = (weight * self.counts[wid] / self.total, 0)
                    added += 1
                    if added >= k:
                        break
        return sorted(best.items(), key=lambda x: -x[1][0])[:k]

    def sample_next(self, history, temperature, rng):
        cands = []
        for level in range(min(self.levels, len(history)), 0, -1):
            cands = self.nexts(level, " ".join(history[-level:]))
            if cands:
                break
        if not cands:
            cands = [(self.words[i], self.counts[i], 0)
                     for i in range(FIRST_WORD, min(len(self.words), FIRST_WORD + 300))]
        weights = [c ** (1.0 / temperature) for _, c, _ in cands]
        return rng.choices([w for w, _, _ in cands], weights)[0]

    # ---- ranking, used by the compressor ---------------------------------
    # For a given history, all words are put in one fixed order: candidates from
    # the longest context, then from shorter contexts, then every other word by
    # overall frequency. A word's "rank" is its position in that order. Good
    # predictions -> small ranks -> few bits.
    def _candidate_lists(self, history):
        for level in range(min(self.levels, len(history)), 0, -1):
            yield [w for w, _, _ in self.nexts(level, " ".join(history[-level:]))]

    def rank_of(self, history, word):
        wid = self.word_id.get(word)
        if wid is None or wid < FIRST_WORD:
            return None                                    # not in vocabulary
        seen, r = set(), 0
        for lst in self._candidate_lists(history):
            for cand in lst:
                if cand in seen:
                    continue
                if cand == word:
                    return r
                seen.add(cand)
                r += 1
        pos = wid - FIRST_WORD
        return r + pos - sum(1 for s in seen if self.word_id[s] - FIRST_WORD < pos)

    def word_at(self, history, rank):
        seen, i = set(), 0
        for lst in self._candidate_lists(history):
            for cand in lst:
                if cand in seen:
                    continue
                if i == rank:
                    return cand
                seen.add(cand)
                i += 1
        pos = rank - i
        for s in sorted(self.word_id[x] - FIRST_WORD for x in seen):
            if s <= pos:
                pos += 1
            else:
                break
        return self.words[pos + FIRST_WORD]


NO_SPACE_BEFORE = set(".,!?;:%)]}'")


def detokenize(tokens):
    out, cap = [], True
    for t in tokens:
        if t == "i":
            t = "I"
        if cap and t[:1].isalpha():
            t = t[:1].upper() + t[1:]
            cap = False
        if out and not (t in NO_SPACE_BEFORE or t.startswith("'")) and out[-1] not in "([{":
            out.append(" ")
        out.append(t)
        if t in ".!?":
            cap = True
    return "".join(out)


def cmd_lookup(args):
    m = NGramModel(args.model)
    history = tokenize(args.words)
    if not history:
        sys.exit("Give at least one word.")
    for level in range(min(m.levels, len(history)), 0, -1):
        ctx = " ".join(history[-level:])
        rows = m.nexts(level, ctx)
        print(f"\nlevel {level}: after {ctx!r}  ({len(rows)} stored)")
        for w, c, p in rows[:args.k]:
            print(f"   {w!r:<16} P = {p:.4f}   (seen {c:,}x)")
        for w in args.next or []:
            w = norm(w)
            hit = [(c, p) for nw, c, p in rows if nw == w]
            if hit:
                print(f"   -> P({w!r} | {ctx!r}) = {hit[0][1]:.5f}   (seen {hit[0][0]:,}x)")
            else:
                print(f"   -> {w!r} after {ctx!r}: not stored "
                      f"(seen fewer than min-count times, or outside the top-k)")
    w = history[-1]
    if w in m.word_id:
        c = m.counts[m.word_id[w]]
        print(f"\nlevel 0: {w!r} appears {c:,}x  (P = {c / m.total:.6f} of all words)")
    else:
        print(f"\nlevel 0: {w!r} is not in the vocabulary")


def cmd_predict(args):
    m = NGramModel(args.model)
    print("Type text and press Enter to see the next-word suggestions.\n"
          "Leave the last word unfinished (no trailing space) to get completions.\n"
          "Empty line quits.\n")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not line.strip():
            break
        toks = tokenize(line)
        prefix = ""
        if toks and not line[-1].isspace() and re.fullmatch(r"\w[\w']*", toks[-1]):
            prefix = toks.pop()
        for w, (score, level) in m.predict(toks, args.k, prefix):
            src = f"level {level}: after {' '.join(toks[-level:])!r}" if level else "level 0: common word"
            print(f"   {w:<16} score {score:.4f}   ({src})")
        print()


def cmd_generate(args):
    m = NGramModel(args.model)
    rng = random.Random(args.seed)
    if args.grammar:
        import grammatizator
        toks, vetoes = grammatizator.generate(m, args.start, args.words, args.temperature, rng,
                                              no_junk=not args.allow_junk)
        print(detokenize(toks))
        print(f"\n[grammar vetoed {len(vetoes)} word choices]", file=sys.stderr)
        for ctx, w, why in vetoes[:10]:
            print(f"   after '{ctx}': not '{w}'  ({why})", file=sys.stderr)
        return
    history = tokenize(args.start)
    out = list(history)
    for _ in range(args.words):
        w = m.sample_next(history, args.temperature, rng)
        out.append(w)
        history.append(w)
    print(detokenize(out))


# --------------------------------------------------------------------------
# Compression
# --------------------------------------------------------------------------
# Each word is replaced by its rank in the model's prediction list (0 = the
# model's first guess). Casing and spacing go in two small side streams, and
# anything the model doesn't know is stored literally. Then deflate squeezes
# the streams. The receiver needs the exact same model.db to decode.
GAPS = ["", " ", "\n", "\r\n", "\n\n", "\r\n\r\n"]
GAP_LITERAL = 255


def _apply_case(w, code):
    return w if code == 0 else (w[:1].upper() + w[1:] if code == 1 else w.upper())


def _put_varint(buf, v):
    while v >= 0x80:
        buf.append((v & 0x7F) | 0x80)
        v >>= 7
    buf.append(v)


def _get_varint(data, pos):
    v = shift = 0
    while True:
        b = data[pos]
        pos += 1
        v |= (b & 0x7F) << shift
        if b < 0x80:
            return v, pos
        shift += 7


def compress_text(m, text):
    gaps, cases, ranks, literals = bytearray(), bytearray(), bytearray(), []
    history, pos, n_tok = [], 0, 0

    def put_gap(g):
        if g in GAPS:
            gaps.append(GAPS.index(g))
        else:
            gaps.append(GAP_LITERAL)
            literals.append(g)

    for match in TOKEN_RE.finditer(text):
        put_gap(text[pos:match.start()])
        orig = match.group()
        w = norm(orig)
        r = m.rank_of(history, w)
        if r is None:
            _put_varint(ranks, 0)            # 0 = "not in vocabulary, read a literal"
            literals.append(w)
        else:
            _put_varint(ranks, r + 1)
        code = next((c for c in (0, 1, 2) if _apply_case(w, c) == orig), 3)
        cases.append(code)
        if code == 3:
            literals.append(orig)
        history.append(w)
        pos = match.end()
        n_tok += 1
    put_gap(text[pos:])

    payload = bytearray()
    _put_varint(payload, n_tok)
    payload += gaps + cases + ranks
    for lit in literals:
        b = lit.encode("utf-8")
        _put_varint(payload, len(b))
        payload += b
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    packed = co.compress(bytes(payload)) + co.flush()
    flag, body = (1, packed) if len(packed) < len(payload) else (0, bytes(payload))
    return bytes([0xA0 | flag]) + bytes.fromhex(m.model_id)[:2] + body


def decompress_bytes(m, data):
    if len(data) < 3 or data[0] >> 4 != 0xA:
        sys.exit("Not ngram-compressed data.")
    if data[1:3] != bytes.fromhex(m.model_id)[:2]:
        sys.exit("This was compressed with a different model.db.")
    body = data[3:]
    if data[0] & 1:
        body = zlib.decompressobj(-15).decompress(body)
    n_tok, pos = _get_varint(body, 0)
    gaps = body[pos:pos + n_tok + 1]
    pos += n_tok + 1
    cases = body[pos:pos + n_tok]
    pos += n_tok
    ranks = []
    for _ in range(n_tok):
        r, pos = _get_varint(body, pos)
        ranks.append(r)
    literals = []
    while pos < len(body):
        ln, pos = _get_varint(body, pos)
        literals.append(body[pos:pos + ln].decode("utf-8"))
        pos += ln
    lit = iter(literals)

    def gap(code):
        return next(lit) if code == GAP_LITERAL else GAPS[code]

    out, history = [], []
    for i in range(n_tok):
        out.append(gap(gaps[i]))
        w = next(lit) if ranks[i] == 0 else m.word_at(history, ranks[i] - 1)
        out.append(next(lit) if cases[i] == 3 else _apply_case(w, cases[i]))
        history.append(w)
    out.append(gap(gaps[n_tok]))
    return "".join(out)


def _deflate(b):
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    return co.compress(b) + co.flush()


def cmd_compress(args):
    m = NGramModel(args.model)
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if text is None:
        sys.exit("Give text or --file.")
    data = compress_text(m, text)
    if decompress_bytes(m, data) != text:        # never hand out something that won't decode
        sys.exit("Internal error: round-trip check failed.")
    raw = len(text.encode("utf-8"))
    print(f"original:     {raw:>8,} bytes", file=sys.stderr)
    print(f"plain zip:    {len(_deflate(text.encode('utf-8'))):>8,} bytes", file=sys.stderr)
    print(f"n-gram model: {len(data):>8,} bytes   ({100 * len(data) / max(raw, 1):.1f}% of original)",
          file=sys.stderr)
    if args.out:
        Path(args.out).write_bytes(data)
        print(f"written to {args.out}", file=sys.stderr)
    else:
        print(base64.urlsafe_b64encode(data).decode().rstrip("="))


def cmd_decompress(args):
    m = NGramModel(args.model)
    if args.file:
        data = Path(args.file).read_bytes()
    else:
        code = args.code.strip()
        data = base64.urlsafe_b64decode(code + "=" * (-len(code) % 4))
    text = decompress_bytes(m, data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text + ("" if text.endswith("\n") else "\n"))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Word-pattern (n-gram) model builder and tools.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="count word patterns in .txt/.pdf files")
    b.add_argument("inputs", nargs="+", help="files or folders (searched recursively)")
    b.add_argument("-o", "--output", default="model.db")
    b.add_argument("--levels", type=int, default=4,
                   help="highest level; 4 = patterns up to 5 words (default 4)")
    b.add_argument("--min-count", type=int, default=2,
                   help="drop patterns seen fewer times than this (default 2)")
    b.add_argument("--top-k", type=int, default=50,
                   help="keep at most this many next-words per context (default 50)")
    b.add_argument("--min-word-count", type=int, default=2,
                   help="words seen fewer times become <unk> (default 2)")
    b.add_argument("--max-vocab", type=int, default=0, help="cap vocabulary size (0 = no cap)")
    b.add_argument("--workers", type=int, default=0, help="CPU processes (default: all threads)")
    b.add_argument("--device", choices=["auto", "cpu", "cuda", "torch-cpu"], default="auto",
                   help="where to count: auto uses a CUDA GPU if PyTorch can see one")
    b.add_argument("--clean", action="store_true",
                   help="remove junk first (HTML, reply links, captions, headers...; see ngram_clean.py)")
    b.add_argument("--clean-mode", choices=["auto", "forum", "book"], default="auto",
                   help="auto: .csv = forum posts, .txt/.pdf = book")
    b.add_argument("--grammar-filter", action="store_true",
                   help="with --clean: also drop sentences the grammatizator finds errors in")
    b.add_argument("--keep-quotes", action="store_true", help="with --clean: keep >quoted lines")
    b.add_argument("--blocklist", help="with --clean: file of words; sentences containing them are dropped")
    b.add_argument("--column", help="CSV column (name or number) holding the text; default: guess")
    b.add_argument("--gpu-rows", type=int, default=20_000_000,
                   help="rows per GPU batch; lower it if the GPU runs out of memory")
    b.set_defaults(func=cmd_build)

    lk = sub.add_parser("lookup", help="show what follows some words, with likelihoods")
    lk.add_argument("model")
    lk.add_argument("words")
    lk.add_argument("--next", nargs="*", help="specific next words to look up")
    lk.add_argument("-k", type=int, default=10)
    lk.set_defaults(func=cmd_lookup)

    p = sub.add_parser("predict", help="interactive next-word predictor")
    p.add_argument("model")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_predict)

    g = sub.add_parser("generate", help="let the model write text")
    g.add_argument("model")
    g.add_argument("start", nargs="?", default="")
    g.add_argument("--words", type=int, default=60)
    g.add_argument("--temperature", type=float, default=0.8,
                   help="lower = safer/more repetitive, higher = wilder")
    g.add_argument("--seed", type=int)
    g.add_argument("--grammar", action="store_true",
                   help="every word must pass the grammatizator's rules (needs grammatizator.py)")
    g.add_argument("--allow-junk", action="store_true", help="with --grammar: allow numbers/symbols")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("compress", help="compress text with the model")
    c.add_argument("model")
    c.add_argument("text", nargs="?")
    c.add_argument("--file")
    c.add_argument("--out", help="write binary here instead of printing a text code")
    c.set_defaults(func=cmd_compress)

    d = sub.add_parser("decompress", help="decompress a code or file")
    d.add_argument("model")
    d.add_argument("code", nargs="?")
    d.add_argument("--file")
    d.add_argument("--out")
    d.set_defaults(func=cmd_decompress)

    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):     # Windows consoles + odd characters
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass
    args.func(args)


if __name__ == "__main__":     # required on Windows for multiprocessing
    main()
