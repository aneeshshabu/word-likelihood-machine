# ngram.py: a word-pattern model you build yourself

You feed it text (`.txt`, `.md`, `.pdf`). It counts which word follows which, and how often. It saves those counts as lookup tables which can be visualised and experimented on. Then you can use the tables to:

- look up likelihoods (`lookup`)
- suggest the next word like a phone keyboard (`predict`)
- let it write text (`generate`)
- compress messages (`compress` / `decompress`)
- Visualise using ngram-map

## Setup (Windows, macOS or Linux)

```
pip install numpy pypdf
```

That's enough to use every CPU thread. **For the GPU (optional):** install PyTorch with CUDA using the command that https://pytorch.org/get-started/locally/ gives you (choose Pip, your OS and a CUDA version). Then `--device auto`, the default, uses the GPU automatically. Check with:

```
python -c "import torch; print(torch.cuda.is_available())"
```

## The easy way: the window

```
python ngram_gui.py
```

It has tabs for Build, Look up, Predict (a live keyboard: click a suggestion or press Tab), Generate and Compress. Keep it in the same folder as `ngram.py`. If you rename it to `ngram_gui.pyw`, double-clicking it opens the window without a console.

## Cleaning junk out of your data (`ngram_clean.py`)

Junk in, junk out. The cleaner runs before counting. In the window, use the **Cleaning** box on the Build tab; it's on by default. From the command line:

```
python ngram.py build pol_500K4chan.csv -o pol.db --clean --grammar-filter
python ngram_clean.py pol_500K4chan.csv -o pol_clean.txt          # standalone: writes a clean .txt
python ngram_clean.py "wheelers dadh.pdf" --preview                # before/after samples
```

- **Forum mode** (`.csv` files): reads only the CSV's text column. It removes HTML, `>>12345` reply links, `>greentext` and quoted lines, "X replied to", "Quote from", `[quote]` blocks, links, emails, `@mentions`, timestamps, "Sent from my iPhone" and similar.
- **Book mode** (`.txt`/`.pdf`): removes figure and table captions, page headers and footers (any short line that repeats), page numbers, headings, broken table cells, "(see Figure 10-17)", `[12]`, citations, footnote numbers and bibliography entries. It also mends words split by a hyphen at the end of a line, then keeps only proper sentences.
- **`--grammar-filter`** also drops sentences the grammatizator finds errors in.
- **`--blocklist words.txt`** (one word per line) drops sentences containing those words. Useful if you build a keyboard from forum data.

At the end, it prints a report of exactly what it removed. A blank line in any text file now counts as "new document", so patterns never run across two posts.

## The grammatizator (`grammatizator.py`)

English grammar as plain logic: 19 rules built on hand-written word lists (122 irregular verbs with all their forms, 273 common verbs, articles, pronouns, modals, be/have/do). There's no machine learning in it.

```
python grammatizator.py check "i has went to a apple store"
python grammatizator.py fix "your a genius, you should of known"
python grammatizator.py rules
python grammatizator.py generate model.db "it was"
python ngram.py generate model.db "it was" --grammar
```

In the window, it's in three places:

- the **Grammar** tab: check or fix any text, with problems underlined
- **Generate**: the model proposes a word, the rules can veto it, and the log shows every veto
- **Predict**: ungrammatical suggestions are blocked (shown in red in the details list)

**What it can't do:** judge meaning. "The ship ate a happy idea." passes every rule. It also only checks words its lists know. Tested on 13,000 sentences from published books, it flagged 0.8%, and nearly all of those were genuinely old grammar (King James Bible "an hundred", Shakespeare "stronger then").

## 3D word map

```
python ngram_map.py model.db
python ngram_map.py model.db --words 800 --links 6 --skip-top 20
```

**No terminal needed:** use the **Word map** tab in `ngram_gui.py`, or double-click `ngram_map.py` and pick a model in the little window. It writes `<model>_map.html` next to the model and opens it in your browser. The first map downloads the 3D library once into a `map_assets` folder. After that, maps work offline, and each map is a single .html file you can share.

- **Size** = how common the word is.
- **Colour** = a group of words that often appear together.
- **Lines** = word pairs that often sit side by side. The strength is a "Dice score", which keeps everything from just linking to "the".
- **Moving dots** = which word usually comes first.

Click a word to see what usually comes next and before it, with likelihood bars. `--skip-top 20` hides the 20 most common words. 

## Commands

```
python ngram.py build my_texts/ chats.txt book.pdf -o model.db
python ngram.py lookup model.db "how" --next are may
python ngram.py predict model.db
python ngram.py generate model.db "i think that" --words 80
python ngram.py compress model.db "hey, how are you doing today?"
python ngram.py decompress model.db <the code it printed>
```

A folder is searched recursively for `.txt/.md/.pdf/.csv/.srt/.log` files.

## How it works

**Levels** (your naming):

| level | pattern | example | table |
|---|---|---|---|
| 0 | 1 word | how common "are" is | `vocab` |
| 1 | 2 words | "how" → "are" | `level_1` |
| 2 | 3 words | "how are" → "you" | `level_2` |
| k | k+1 words | k words of context → next word | `level_k` |

**The likelihood number** is a conditional probability:

```
P(are | how) = (times "how are" appeared) / (times "how" was followed by any word)
```

So all the next-word probabilities for one context add up to 1. On the test books, P(are | how) = 0.008, and "how may" was too rare to be stored at all.

**Why one table per level:** each level has a different context length, and the predictor checks them in order. It tries the longest context first (`level_4`: the last 4 words). If that context was never seen, it falls back to 3 words, then 2, and so on. A word found only at a shorter level gets its score multiplied by 0.4 for every level it dropped (this is called "stupid backoff", and it really is the name). The tables are in an SQLite file. You can open `model.db` in *DB Browser for SQLite* and browse it, and later an Android keyboard app can read the same file.

**How the counting uses your hardware:**

1. All CPU threads read files and tokenize the text (lowercase it and split it into words and punctuation).
2. The text becomes one long array of word ids.
3. The array is split into pieces by hashing each pattern's first words. Every piece is counted (a big sort), turned into probabilities and pruned on its own. On CPU, that runs on every core. With `--device cuda`, the sort runs on the GPU.

To be honest about the GPU: counting n-grams is mostly sorting and memory traffic, not heavy maths. A GPU helps, but much less than it does for neural networks. A 16-thread CPU can be close. The build prints timings, so try both and compare:

```
python ngram.py build texts/ --device cpu
python ngram.py build texts/ --device cuda
```

## Settings worth knowing

| option | default | meaning |
|---|---|---|
| `--levels` | 4 | highest level (4 = patterns up to 5 words) |
| `--min-count` | 2 | drop patterns seen only once. This keeps the file *much* smaller. Use 1 for tiny inputs |
| `--top-k` | 50 | keep at most 50 next-words per context |
| `--min-word-count` | 2 | words seen once become `<unk>` (unknown) |
| `--workers` | all threads | CPU processes |
| `--gpu-rows` | 20M | lower this if the GPU runs out of memory |

**RAM:** plan for roughly 3–4× the size of your text (1 GB of text ≈ 180M words). The file grows with `--levels`. Most 5-word patterns occur only once, which is why `--min-count 2` matters.

## Compression: what's happening

Sender and receiver both have the same `model.db`. For each word, the sender asks the model for its ranked list of guesses and writes down the word's **position in that list** (0 = the model guessed right first try). Good predictions give tiny numbers, and tiny numbers take few bits. Capital letters, spacing and unknown words are stored separately, so the result decodes to exactly the original text. On a normal paragraph it came out at about 40% of the original size, versus about 64% for plain zip. Very short messages save less, because there's a 3-byte header.

The "proper" version of this replaces ranks with **arithmetic coding**. That spends exactly −log₂(P) bits per word, so a word the model gave 50% costs 1 bit. Compression with arithmetic coding is literally the same thing as prediction, and it's why people say a better language model is a better compressor.

## Things you'll notice (and why they matter for your LLM)

- **`generate` copies whole sentences from the books** once it's in a 4-word context. Most 4-word contexts were seen only once, with one continuation, so the model memorizes rather than generalizes.
- **It knows nothing about words it hasn't seen together.** If "how are" was never followed by "ya", the model has no idea that "ya" ≈ "you". The table has no notion of similar words.
- **The context is fixed.** It sees 4 words back and nothing before that.

These three limits are exactly what neural language models fix. Word embeddings fix "similar words". Attention fixes "fixed context": every word can look back at every earlier word and decide which ones matter. Your n-gram table is the baseline your LLM has to beat.

