#!/usr/bin/env python3
"""
grammatizator.py - English grammar rules written as plain logic.

    python grammatizator.py check "he go to the the shop. i has a apple"
    python grammatizator.py fix   "your a idiot, you should of known"
    python grammatizator.py check --file essay.txt
    python grammatizator.py rules                      # every rule, with examples
    python grammatizator.py generate model.db "it was" # n-gram text, filtered by the rules

How it works
------------
1. WORD LISTS: hand-written lists say what each word can be: articles (a, an,
   the), pronouns (i, he, they...), forms of be/have/do, modal verbs (can,
   would...), ~130 irregular verbs with all their forms, ~200 common verbs.
2. RULES: each rule looks at a word and the words just before it, and says
   "this can't happen in English" (an error) or "this is probably wrong"
   (a warning). Example: after "he/she/it", a plain verb like "go" is wrong;
   it must be "goes".
3. During generation, the n-gram model proposes the next word and the rules
   can veto it. The model then picks its next-best word.

What it can't do (nobody's rule set can): judge meaning. "The ship ate a
happy idea." breaks no rule. And it only flags what its lists know: an
unknown verb can't be checked.
"""
import argparse
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass

TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")      # same as ngram.py


def norm(t):
    return t.lower().replace("’", "'")


# ==========================================================================
# 1. WORD LISTS
# ==========================================================================
ARTICLES = {"a", "an", "the"}
POSS_DET = {"my", "your", "our", "their", "its"}
DET_STRICT = ARTICLES | POSS_DET | {"every"}               # must be followed by a noun phrase
DET_SECOND = DET_STRICT | {"his", "her"}
SUBJ_ONLY = {"i", "he", "she", "we", "they"}               # can't come right after "the"
BE_FINITE = {"am", "is", "are", "was", "were"}
BE_NEG = {"isn't", "aren't", "wasn't", "weren't"}
HAVE_FORMS = {"have", "has", "had", "haven't", "hasn't", "hadn't", "i've", "you've", "we've", "they've"}
DO_NEG = {"don't", "doesn't", "didn't"}
MODALS_STRICT = {"would", "could", "should", "shall", "wouldn't", "couldn't", "shouldn't", "mustn't",
                 "can't", "cannot", "won't", "shan't", "mightn't"}
MODALS_AMBIG = {"can", "will", "may", "must", "might"}      # also nouns: "a can", "the will"
AUX_ALL = (BE_FINITE | BE_NEG | HAVE_FORMS | DO_NEG | MODALS_STRICT | MODALS_AMBIG |
           {"do", "does", "did"})
CAUSATIVE = {"let", "lets", "letting", "make", "makes", "made", "making", "have", "has", "had", "having",
             "help", "helps", "helped", "see", "saw", "seen", "watch", "watched", "hear", "heard",
             "feel", "felt", "bid"}
COORD = {"and", "or", "but", "nor"}
CLAUSE_START = COORD | {"so", "because", "if", "when", "while", "although", "though", "since", "unless",
                        "until", "where", "whereas", "then"}
PREP_STRICT = {"of", "to", "with", "from", "into", "onto", "upon", "at"}
NO_END = ARTICLES | POSS_DET | {"every", "and", "or", "nor", "because", "although", "whose"}
SENT_END = {".", "!", "?"}
SOFT_PUNCT = {",", ";", ":"}
ALL_PUNCT = SENT_END | SOFT_PUNCT
REPEAT_OK = {"that", "had", "is", "very", "really", "no", "so", "bye", "ha", "haha", "yes", "oh", "ah",
             "blah", "la", "far", "many", "much", "knock", "tick", "tut", "go", "come", "more", "now", "well"}

# irregular verbs: base past participle
IRREGULAR = """
arise arose arisen | awake awoke awoken | bear bore borne | beat beat beaten | become became become
begin began begun | bend bent bent | bet bet bet | bind bound bound | bite bit bitten | bleed bled bled
blow blew blown | break broke broken | breed bred bred | bring brought brought | build built built
burst burst burst | buy bought bought | catch caught caught | choose chose chosen | cling clung clung
come came come | cost cost cost | creep crept crept | cut cut cut | deal dealt dealt | dig dug dug
do did done | draw drew drawn | drink drank drunk | drive drove driven | eat ate eaten | fall fell fallen
feed fed fed | feel felt felt | fight fought fought | find found found | flee fled fled | fling flung flung
fly flew flown | forbid forbade forbidden | forget forgot forgotten | forgive forgave forgiven
freeze froze frozen | get got got | give gave given | go went gone | grind ground ground | grow grew grown
hang hung hung | have had had | hear heard heard | hide hid hidden | hit hit hit | hold held held
hurt hurt hurt | keep kept kept | kneel knelt knelt | know knew known | lay laid laid | lead led led
leave left left | lend lent lent | let let let | lie lay lain | light lit lit | lose lost lost
make made made | mean meant meant | meet met met | pay paid paid | put put put | quit quit quit
read read read | ride rode ridden | ring rang rung | rise rose risen | run ran run | say said said
see saw seen | seek sought sought | sell sold sold | send sent sent | set set set | shake shook shaken
shine shone shone | shoot shot shot | shrink shrank shrunk | shut shut shut | sing sang sung
sink sank sunk | sit sat sat | sleep slept slept | slide slid slid | speak spoke spoken
spend spent spent | spin spun spun | spread spread spread | spring sprang sprung | stand stood stood
steal stole stolen | stick stuck stuck | sting stung stung | stink stank stunk | strike struck struck
swear swore sworn | sweep swept swept | swim swam swum | swing swung swung | take took taken
teach taught taught | tear tore torn | tell told told | think thought thought | throw threw thrown
understand understood understood | wake woke woken | wear wore worn | weep wept wept | win won won
wind wound wound | write wrote written
"""
IRR = [tuple(e.split()) for e in re.split(r"[|\n]", IRREGULAR) if e.strip()]
BASE_OF = {}                                   # any form -> base
for b, p, pp in IRR:
    BASE_OF.setdefault(p, b)
    BASE_OF.setdefault(pp, b)
PARTICIPLE_OF = {p: pp for b, p, pp in IRR if p != pp}          # went -> gone
PARTICIPLE_OF.update({"was": "been", "were": "been"})
# past forms that are also other words (a saw, to fell trees, to found a company...)
AMBIGUOUS_FORMS = {"bore", "lay", "saw", "fell", "rose", "found", "ground", "wound", "bound", "left",
                   "lit", "shot", "struck", "stuck", "spun", "won", "bit", "spoke", "stole", "tore"}
PAST_NOT_PART = {p for p in PARTICIPLE_OF if p not in AMBIGUOUS_FORMS} - {"was", "were"}   # went, came...
IRR_BASES = {b for b, _, _ in IRR}
PAST_FORMS = ({p for _, p, _ in IRR} | {pp for _, _, pp in IRR}) - IRR_BASES - AMBIGUOUS_FORMS

REGULAR_VERBS = """
accept add agree allow answer appear apply argue arrive ask attack avoid believe belong borrow call care
carry change check clean close collect compare complain consider contain continue cook copy count cover
create cry dance decide deliver depend describe destroy die disappear discover discuss dislike drop earn
enjoy enter escape exist expect explain fail fill finish fix follow force gather guess hate happen help
hope hurry ignore imagine improve include increase intend introduce invite join jump kill kiss knock
laugh learn like listen live look love manage marry matter mention miss move notice obey offer open
order own pass pick plan play pray prefer prepare pretend prevent produce promise protect prove provide
pull push reach realize realise receive refuse remain remember remove repeat reply report require rest
return save scream seem serve settle share shout smell smile solve start stay stop study succeed suffer
suggest support suppose surprise talk taste thank touch travel try turn use visit vote wait walk want
warn wash watch wish wonder work worry exist deserve belong agree claim admit deny
"""
# verbs whose plain form after he/she/it is wrong ("he go" -> "he goes")
BASE_VERBS = ({b for b in IRR_BASES if b not in {b2 for b2, p, _ in IRR if p == b2}} |
              set(REGULAR_VERBS.split())) - {"have", "do", "be", "need", "dare", "lie", "lay", "wind", "bear", "like"}


def third_person(v):
    """go -> goes, try -> tries, watch -> watches."""
    if v == "have":
        return "has"
    if v.endswith("y") and len(v) > 2 and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    if v.endswith(("s", "sh", "ch", "x", "z", "o")):
        return v + "es"
    return v + "s"


THIRD_SG = {third_person(v): v for v in BASE_VERBS}            # goes -> go

COMPARATIVES = set("""bigger smaller better worse larger faster slower older younger easier harder higher
lower stronger weaker longer shorter greater richer poorer happier sadder smarter cheaper newer closer
deeper darker cleaner colder hotter warmer cooler quicker louder quieter safer wiser nicer heavier thinner
thicker wider narrower brighter tougher simpler busier funnier prettier uglier healthier wealthier
earlier later further fewer""".split())
SUPERLATIVES = {w[:-1] + "st" if w.endswith("er") else w for w in COMPARATIVES} | {"best", "worst", "least"}
SUPERLATIVES = {s.replace("betst", "best").replace("worsst", "worst") for s in SUPERLATIVES}

MISSPELLED = {
    "alot": "a lot", "irregardless": "regardless", "definately": "definitely", "seperate": "separate",
    "untill": "until", "recieve": "receive", "wierd": "weird", "beleive": "believe", "occured": "occurred",
    "thier": "their", "teh": "the", "dont": "don't", "doesnt": "doesn't", "didnt": "didn't",
    "isnt": "isn't", "wasnt": "wasn't", "arent": "aren't", "werent": "weren't", "couldnt": "couldn't",
    "shouldnt": "shouldn't", "wouldnt": "wouldn't", "havent": "haven't", "hasnt": "hasn't",
    "im": "I'm", "ive": "I've", "youre": "you're", "theyre": "they're", "thats": "that's",
    "whats": "what's", "theres": "there's", "lets": None, "u": "you", "ur": "your", "pls": "please",
    "probaly": "probably", "goverment": "government", "tommorow": "tomorrow", "begining": "beginning",
    "noone": "no one", "eachother": "each other", "infact": "in fact", "atleast": "at least",
}
MISSPELLED = {k: v for k, v in MISSPELLED.items() if v}

# "a" vs "an" goes by SOUND, not letter
AN_SOUND = ("hour", "honest", "honor", "honour", "heir", "herb")
A_SOUND = ("unic", "unif", "unil", "unio", "uniq", "unis", "unit", "univ", "unix", "usa", "use", "usu",
           "uti", "ute", "ura", "ure", "uri", "uro", "eu", "ewe", "one", "once", "ubiq", "ufo", "uga",
           "uku", "ukr")

VERBISH_ENDINGS = ("'s", "'re", "'m", "'ll", "'ve", "'d", "n't")


def wants_an(word):
    """True if the word should take 'an', False for 'a', None if we can't tell."""
    w = word.lower()
    if not w[:1].isalpha():
        return None
    if w.startswith(AN_SOUND):
        return True
    if w.startswith(A_SOUND):
        return False
    if len(w) <= 4 and not re.search(r"[aeiouy]", w):     # acronyms like "fbi": unknown
        return None
    return w[0] in "aeiou"


def is_verbish(w):
    return (w in BE_FINITE or w in BE_NEG or w in HAVE_FORMS or w in AUX_ALL or w in BASE_OF
            or w in BASE_VERBS or w in THIRD_SG or w in {"be", "been", "being"}
            or w.endswith(VERBISH_ENDINGS) or (len(w) > 4 and w.endswith(("ed", "ing"))))


def is_word(t):
    return bool(t) and (t[0].isalnum() or t[0] == "_")


def is_junk(t):
    """For generation only: numbers, odd symbols, very long tokens."""
    if any(ch.isdigit() for ch in t):
        return True
    if not is_word(t) and t not in ALL_PUNCT | {"'", '"', "(", ")", "-"}:
        return True
    return len(t) > 20


# ==========================================================================
# 2. RULES
# ==========================================================================
@dataclass
class Issue:
    rule: str
    severity: str          # "error" or "warning"
    message: str
    at: int                # index of the token to replace
    fix: object = None     # replacement text for that token ("" = delete), None = no fix
    first: int = 0         # first token to highlight
    last: int = 0          # last token to highlight


RULES = []                 # (id, severity, title, explanation, bad example, good example, function)


def rule(rid, severity, title, explain, bad, good):
    def deco(fn):
        RULES.append((rid, severity, title, explain, bad, good, fn))
        return fn
    return deco


class GrammarState:
    """Remembers the text so far, so rules can look back."""

    def __init__(self, vocab=None):
        self.toks = []
        self.origs = []
        self.sent_start = 0
        self.parens = 0
        self.sent_words = 0
        self.sent_verb = False
        self.vocab = vocab

    def prev(self, k=1):
        return self.toks[-k] if len(self.toks) >= k else None

    def prev_orig(self, k=1):
        return self.origs[-k] if len(self.origs) >= k else None

    def capitalised_mid_sentence(self, k=1):
        """Was the k-th previous word written with a capital letter in the middle of a sentence?
        ("May" the month, "A" as a label, initials like "E.")"""
        o = self.prev_orig(k)
        return bool(o) and o[:1].isupper() and len(self.toks) - k > self.sent_start

    def at_sentence_start(self):
        return len(self.toks) == self.sent_start

    def issues_for(self, tok, nxt=None, orig=None, check_warnings=True):
        """Problems that adding `tok` next would cause (doesn't change the state)."""
        out = []
        for rid, sev, title, explain, bad, good, fn in RULES:
            if sev == "warning" and not check_warnings:
                continue
            res = fn(self, tok, nxt, orig)
            if res:
                msg, at_back, fix, span_back = res
                i = len(self.toks)
                out.append(Issue(rid, sev, msg, i - at_back, fix, i - span_back, i))
        return out

    def push(self, tok, orig=None):
        self.toks.append(tok)
        self.origs.append(orig)
        if tok == "(":
            self.parens += 1
        elif tok == ")" and self.parens:
            self.parens -= 1
        if is_word(tok):
            self.sent_words += 1
            if is_verbish(tok):
                self.sent_verb = True
        if tok in SENT_END:
            self.sent_start = len(self.toks)
            self.sent_words = 0
            self.sent_verb = False


# Each rule returns None (fine) or (message, which token to fix [0 = this one,
# 1 = the one before], replacement or None, how far back to highlight).

@rule("repeated-word", "error", "Same word twice", "A word repeated by accident.",
      "I went to the the shop.", "I went to the shop.")
def r_repeat(s, t, nxt, orig):
    if t == s.prev() and is_word(t) and t not in REPEAT_OK and not t.isdigit():
        return f"'{t}' twice in a row", 0, "", 1


@rule("a-an", "error", "a / an", "'an' goes before a vowel SOUND (an apple, an hour), 'a' before a "
      "consonant sound (a car, a university).", "a apple, an university", "an apple, a university")
def r_a_an(s, t, nxt, orig):
    p = s.prev()
    if p in ("a", "an") and is_word(t) and s.prev_orig() not in ("A",):
        want = wants_an(t)
        if want is not None and (p == "an") != want:
            right = "an" if want else "a"
            return f"use '{right}' before '{t}'", 1, right, 1


@rule("two-determiners", "error", "Two determiners", "Words like the / a / my / your / its don't stack. "
      "Often it's a mix-up: your -> you're, its -> it's, their -> they're.",
      "your a genius, its a trap", "you're a genius, it's a trap")
def r_two_det(s, t, nxt, orig):
    p = s.prev()
    if p in DET_STRICT and t in DET_SECOND and p != t and "A" not in (s.prev_orig(), orig):
        fix = {"your": "you're", "its": "it's", "their": "they're"}.get(p, "")
        hint = f" (did you mean '{fix}'?)" if fix else ""
        return f"'{p} {t}' can't go together{hint}", 1, fix, 1


@rule("determiner-then", "error", "Nothing after 'the'", "After the / a / my / every there must be a noun "
      "(or an adjective and a noun), not a verb, pronoun, 'and', 'of' or punctuation. "
      "'their is' is usually 'there is'.", "the is, a of, their are, my .", "the dog is, there are")
def r_det_then(s, t, nxt, orig):
    p = s.prev()
    if p not in DET_STRICT or nxt == "-" or s.prev_orig() == "A":
        return None
    if p == "my" and t in {",", "!"}:                             # "my, my, how..."
        return None
    if t in BE_FINITE:
        fix = {"their": "there", "your": "you're", "its": "it's"}.get(p)
        if p == "your":
            return f"'your {t}' (did you mean 'you {t}' / 'you're'?)", 1, None, 1
        return f"'{p}' can't be followed by '{t}'" + (f" (did you mean '{fix}'?)" if fix else ""), 1, fix, 1
    if t in SUBJ_ONLY or t in COORD or t in PREP_STRICT or t in MODALS_STRICT or t in ALL_PUNCT or t == ")":
        return f"'{p}' needs a noun after it, not '{t}'", 1, None, 1


@rule("subject-verb", "error", "Subject–verb agreement", "The verb must match the subject: I am / he is / "
      "they are; I have / he has; he goes (not he go).", "he go, they was, I has, she don't",
      "he goes, they were, I have, she doesn't")
def r_agree(s, t, nxt, orig):
    p, pp = s.prev(), s.prev(2)
    inverted = pp in AUX_ALL or (pp == "not" and s.prev(3) in AUX_ALL)   # "does he have", "does not he"
    if pp in {"and", "or", "nor"} and p in {"i", "he", "she", "it"}:   # "you and I are", "he and she go"
        return None
    if p == "i":
        fix = {"is": "am", "are": "am", "has": "have", "does": "do", "doesn't": "don't",
               "isn't": "am not", "aren't": "am not", "hasn't": "haven't"}.get(t)
        if fix and not inverted:
            return f"'I {t}' should be 'I {fix}'", 0, fix, 1
    elif p in ("he", "she", "it"):
        if inverted or (p == "it" and pp in CAUSATIVE):           # "let it go", "make it work"
            return None
        fix = {"are": "is", "am": "is", "have": "has", "do": "does", "don't": "doesn't",
               "aren't": "isn't", "haven't": "hasn't"}.get(t)
        if fix:
            return f"'{p} {t}' should be '{p} {fix}'", 0, fix, 1
        it_is_subject = p != "it" or pp is None or pp in ALL_PUNCT or pp in CLAUSE_START | {"that"}
        if t in BASE_VERBS and it_is_subject:
            return f"'{p} {t}' should be '{p} {third_person(t)}'", 0, third_person(t), 1
    elif p in ("we", "they") or (p == "you" and (pp is None or pp in ALL_PUNCT or pp in CLAUSE_START)):
        fix = {"is": "are", "am": "are", "was": "were", "has": "have", "does": "do", "doesn't": "don't",
               "isn't": "aren't", "wasn't": "weren't", "hasn't": "haven't"}.get(t)
        if fix and not inverted:
            return f"'{p} {t}' should be '{p} {fix}'", 0, fix, 1


PREPOSITIONS = {"of", "in", "on", "at", "for", "with", "from", "by", "about", "into", "onto", "under", "over",
                "through", "after", "before", "between", "among", "behind", "beside", "near", "without", "within",
                "during", "against", "toward", "towards", "upon", "across", "around", "along", "beyond", "like",
                "than", "to", "as", "per", "via"}
NOUN_DETS = {"the", "a", "an", "this", "that", "these", "those", "every", "each", "my", "your", "his", "her",
             "its", "our", "their"}
NOT_SINGULAR_NOUNS = {"people", "police", "cattle", "staff", "team", "family", "crew", "government", "majority",
                      "public", "audience", "committee", "couple", "media", "youth", "clergy", "jury", "rich",
                      "poor", "good", "brave", "number", "rest", "lot", "half", "most", "none", "all"}
PLURAL_LOOKALIKES = {"news", "means", "series", "species", "physics", "mathematics", "politics", "economics",
                     "ethics", "athletics", "measles", "billiards", "gymnastics", "lens", "chaos", "bias",
                     "canvas", "gas", "bus", "plus", "thanks", "always", "perhaps", "whereas", "was", "has",
                     "is", "this", "us", "yes", "less", "unless", "its", "his", "hers", "ours", "yours", "theirs"}


@rule("noun-verb", "error", "Noun–verb agreement", "One thing takes is/was/has; several take are/were/have. "
      "(Uses your model's words to tell singular from plural: if 'roots' exists, 'root' is one.)",
      "the root are long, the roots is long", "the root is long, the roots are long")
def r_noun_agree(s, t, nxt, orig):
    p, pp = s.prev(), s.prev(2)
    if not s.vocab or pp not in NOUN_DETS or not p or not p.isalpha() or s.capitalised_mid_sentence():
        return None
    if s.prev(3) in PREPOSITIONS | COORD:     # "seats in the shade were": the subject is further back
        return None
    singular_verbs = {"is": "are", "was": "were", "has": "have", "isn't": "aren't", "wasn't": "weren't",
                      "hasn't": "haven't"}
    plural_verbs = {"are": "is", "aren't": "isn't"}      # not "were" (if the event were over) or "have"
    if (t in plural_verbs and pp not in {"these", "those"} and p not in NOT_SINGULAR_NOUNS
            and p not in BASE_VERBS and p not in IRR_BASES
            and not p.endswith("s") and p + "s" in s.vocab and p not in BASE_OF and p not in THIRD_SG):
        return f"'{p}' is one thing: '{p} {plural_verbs[t]}'", 0, plural_verbs[t], 1
    if (t in singular_verbs and pp not in {"a", "an", "this", "that", "every", "each"} and len(p) > 3
            and p.endswith("s") and not p.endswith(("ss", "us", "is", "ics", "'s")) and p not in PLURAL_LOOKALIKES
            and p[:-1] in s.vocab and p not in THIRD_SG):
        return f"'{p}' is more than one: '{p} {singular_verbs[t]}'", 0, singular_verbs[t], 1


def _base_of(t):
    if t in BE_FINITE:
        return "be"
    return {"has": "have", "does": "do", "did": "do", "been": "have been"}.get(t) or THIRD_SG.get(t) or BASE_OF.get(t)


@rule("modal-verb", "error", "Verb after can/would/should", "After a modal verb (can, could, would, "
      "should, will, must, might) comes the plain verb: could go, should be, will have.",
      "could went, should is, will goes, can been", "could go, should be, will go, can be")
def r_modal(s, t, nxt, orig):
    p, pp = s.prev(), s.prev(2)
    is_modal = p in MODALS_STRICT or (p in MODALS_AMBIG and pp not in DET_SECOND | PREP_STRICT |
                                       {"this", "that", "in", "on", "for", "by"})
    if not is_modal or pp == "-":                                   # "milk-can"
        return None
    if p in MODALS_AMBIG and (t in BE_FINITE or t in {"has", "does"} or s.capitalised_mid_sentence()):
        return None
    if t in BE_FINITE or t in {"has", "does", "did", "been"} or t in PAST_FORMS or t in THIRD_SG:
        fix = _base_of(t)
        return f"after '{p}' use the plain verb" + (f" '{fix}'" if fix else ""), 0, fix, 1


@rule("should-of", "error", "'should of' → 'should have'", "'should've' sounds like 'should of', "
      "but it's 'should have'.", "I could of won", "I could have won")
def r_should_of(s, t, nxt, orig):
    p, pp = s.prev(), s.prev(2)
    modal = p in {"would", "could", "should"} or (p in {"must", "might", "may"} and pp not in DET_SECOND
                                                     and not s.capitalised_mid_sentence())
    if t == "of" and modal and nxt != "course":
        return f"'{p} of' should be '{p} have'", 0, "have", 1


@rule("to-verb", "error", "Verb after 'to'", "'to' + verb takes the plain verb: to go, to be, to eat.",
      "to went, to was", "to go, to be")
def r_to(s, t, nxt, orig):
    if s.prev() == "to" and (t in PAST_NOT_PART or t in {"was", "were", "did"}):
        fix = _base_of(t)
        return f"after 'to' use the plain verb" + (f" '{fix}'" if fix else ""), 0, fix, 1


@rule("do-not-verb", "error", "Verb after don't/doesn't/didn't", "After don't / doesn't / didn't comes "
      "the plain verb.", "didn't went, doesn't knows", "didn't go, doesn't know")
def r_do_neg(s, t, nxt, orig):
    p = s.prev()
    if p in DO_NEG and (t in BE_FINITE or t in {"has", "does", "did"} or t in PAST_FORMS or t in THIRD_SG):
        fix = _base_of(t)
        return f"after '{p}' use the plain verb" + (f" '{fix}'" if fix else ""), 0, fix, 1


@rule("have-participle", "error", "have + past participle", "After have / has / had comes the past "
      "participle: has gone (not has went), had eaten (not had ate).",
      "has went, had ate, have saw... I've came", "has gone, had eaten, I've come")
def r_have(s, t, nxt, orig):
    p = s.prev()
    if p in HAVE_FORMS and (t in PAST_NOT_PART or t in {"was", "were"}):
        return f"'{p} {t}' should be '{p} {PARTICIPLE_OF[t]}'", 0, PARTICIPLE_OF[t], 1


@rule("double-comparative", "error", "more + -er", "'bigger' already means 'more big'.",
      "more bigger, most best", "bigger, best")
def r_more(s, t, nxt, orig):
    p = s.prev()
    if ((p == "more" and t in COMPARATIVES) or (p == "most" and t in SUPERLATIVES)) and s.prev(2) != "or":
        return f"'{p} {t}': drop '{p}'", 1, "", 1


@rule("than-then", "error", "than / then", "Comparisons use 'than' (bigger than). 'then' is about time.",
      "bigger then me, rather then", "bigger than me, rather than")
def r_than(s, t, nxt, orig):
    p = s.prev()
    if t == "then" and (p in COMPARATIVES or p in {"rather", "other", "less", "fewer"}):
        return f"'{p} then' should be '{p} than'", 0, "than", 1


@rule("punctuation", "error", "Punctuation", "No two punctuation marks in a row (', .'), no ')' "
      "without '(', and a text can't start with a comma or full stop.", "Hello , . world )", "Hello, world.")
def r_punct(s, t, nxt, orig):
    p = s.prev()
    if p in SOFT_PUNCT and t in ALL_PUNCT:
        return f"'{p}' followed by '{t}'", 1, "", 1
    if p in SENT_END and t in SOFT_PUNCT and not (p == "." and len(s.prev(2) or "xx") == 1):
        return f"'{p}' followed by '{t}'", 0, "", 1
    if p == "(" and t in ALL_PUNCT | {")"}:
        return "empty or broken brackets", 0, None, 1
    if t == ")" and s.parens == 0:
        return "')' without an opening '('", 0, "", 0
    if not s.toks and t in ALL_PUNCT | {")"}:
        return f"text can't start with '{t}'", 0, "", 0


@rule("bad-ending", "error", "Sentence ends too early", "A sentence can't end right after the / a / my / "
      "and / or / because.", "I saw the.", "I saw the dog.")
def r_end(s, t, nxt, orig):
    p = s.prev()
    if t in SENT_END and p in NO_END and not (t == "?" and p not in ARTICLES | POSS_DET) \
            and s.prev_orig() != "A":
        return f"sentence can't end on '{p}'", 1, None, 1


@rule("spelling", "warning", "Common misspellings", "Frequent misspellings and missing apostrophes.",
      "alot, definately, dont, im", "a lot, definitely, don't, I'm")
def r_spell(s, t, nxt, orig):
    if t in MISSPELLED:
        return f"'{t}' → '{MISSPELLED[t]}'", 0, MISSPELLED[t], 0


@rule("a-plural", "warning", "'a' + plural", "'a', 'every', 'each' and 'another' go with one thing.",
      "a dogs, every days", "a dog, every day")
def r_a_plural(s, t, nxt, orig):
    p = s.prev()
    if (p in {"a", "an", "every", "each", "another"} and s.vocab and len(t) > 3 and t.endswith("s")
            and not t.endswith(("ss", "us", "is", "ous", "ics", "ies")) and t[:-1] in s.vocab
            and t not in {"news", "means", "series", "species", "sports", "sales", "rights", "savings",
                          "customs", "arms", "jobs", "goods", "looks", "works", "thanks", "odds", "lots"}):
        return f"'{p} {t}': '{t}' looks plural", 0, t[:-1], 1


@rule("no-verb", "warning", "Sentence with no verb", "Every full sentence needs a verb. (Only verbs the "
      "word lists know are recognised, so check before trusting this one.)",
      "The big red house on the hill.", "The big red house stands on the hill.")
def r_no_verb(s, t, nxt, orig):
    if t in SENT_END and s.sent_words >= 5 and not s.sent_verb:
        return "this sentence doesn't seem to have a verb", 0, None, s.sent_words


@rule("capital", "warning", "Capital letters", "Sentences start with a capital letter, and 'I' is "
      "always a capital.", "i think so. it works.", "I think so. It works.")
def r_capital(s, t, nxt, orig):
    if orig is None or not orig[:1].isalpha():
        return None
    if orig == "i":
        return "'i' should be 'I'", 0, "I", 0
    if s.at_sentence_start() and s.toks and orig[:1].islower():
        return "sentence should start with a capital letter", 0, orig[:1].upper() + orig[1:], 0


# ==========================================================================
# 3. CHECKING TEXT
# ==========================================================================
def check(text, vocab=None, warnings=True):
    """All problems in `text`, each with character positions (start, end) to highlight."""
    matches = list(TOKEN_RE.finditer(text))
    toks = [norm(m.group()) for m in matches]
    state = GrammarState(vocab)
    found = []
    for i, m in enumerate(matches):
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        for iss in state.issues_for(toks[i], nxt, m.group(), warnings):
            iss.start = matches[max(0, iss.first)].start()
            iss.end = matches[iss.last].end()
            found.append(iss)
        state.push(toks[i], m.group())
    if matches and warnings:
        if state.parens:
            found.append(_tail(matches, "unclosed-bracket", "a '(' is never closed"))
        if text.count('"') % 2:
            found.append(_tail(matches, "unclosed-quote", 'a " quote is never closed'))
        if toks[-1] not in SENT_END | {'"', ")", "'"}:
            found.append(_tail(matches, "no-full-stop", "the text doesn't end with . ! or ?"))
    return found


def _tail(matches, rid, msg):
    iss = Issue(rid, "warning", msg, len(matches) - 1, None, len(matches) - 1, len(matches) - 1)
    iss.start, iss.end = matches[-1].start(), matches[-1].end()
    return iss


def fix_text(text, vocab=None, warnings=True):
    """Apply every suggested fix (from the end backwards, so positions stay valid)."""
    for _ in range(3):                       # a fix can reveal another; a few passes
        matches = list(TOKEN_RE.finditer(text))
        issues = [i for i in check(text, vocab, warnings) if i.fix is not None and i.at < len(matches)]
        if not issues:
            break
        done = set()
        for iss in sorted(issues, key=lambda i: -i.at):
            if iss.at in done:
                continue
            done.add(iss.at)
            m = matches[iss.at]
            new = iss.fix
            if new and m.group()[:1].isupper() and new[:1].islower():
                new = new[:1].upper() + new[1:]
            start = m.start()
            if new == "":                                        # delete the word and the space before it
                while start > 0 and text[start - 1] == " ":
                    start -= 1
            text = text[:start] + new + text[m.end():]
    return text


# ==========================================================================
# 4. GENERATING WITH THE RULES
# ==========================================================================
def generate(model, start="", n_words=60, temperature=0.8, rng=None, no_junk=True):
    """Like ngram.py generate, but every word the model picks must pass the rules.
    Returns (tokens, vetoes) where vetoes = [(context, rejected word, reason)]."""
    rng = rng or random.Random()
    history = [norm(t) for t in TOKEN_RE.findall(start)]
    state = GrammarState(vocab=model.word_id)
    for t in history:
        state.push(t)
    vetoes = []

    def reason_against(w):
        if no_junk and is_junk(w):
            return "junk (number/symbol)"
        errs = [i for i in state.issues_for(w, check_warnings=False) if i.severity == "error"]
        return errs[0].message if errs else None

    def choose(cands):
        cands = [c for c in cands if c[0] not in rejected]
        while cands:
            weights = [c ** (1.0 / temperature) for _, c in cands]
            w = rng.choices([w for w, _ in cands], weights)[0]
            why = reason_against(w)
            if why is None:
                return w
            vetoes.append((" ".join(history[-3:]), w, why))
            rejected.add(w)
            cands = [c for c in cands if c[0] != w]
        return None

    for _ in range(n_words):
        rejected = set()
        w = None
        for level in range(min(model.levels, len(history)), 0, -1):    # longest context first
            cands = model.nexts(level, " ".join(history[-level:]))
            if cands:
                w = choose((nw, c) for nw, c, _ in cands)
                if w:
                    break
        if w is None:                                                   # back off to common words
            common = [(model.words[i], model.counts[i]) for i in range(2, min(len(model.words), 400))]
            w = choose(common) or common[0][0]
        history.append(w)
        state.push(w)
    return history, vetoes


# ==========================================================================
# command line
# ==========================================================================
def print_issues(text, issues):
    if not issues:
        print("No problems found.")
        return
    for iss in issues:
        line = text.count("\n", 0, iss.start) + 1
        col = iss.start - (text.rfind("\n", 0, iss.start) + 1) + 1
        fix = f"   → '{iss.fix}'" if iss.fix else ("   → delete" if iss.fix == "" else "")
        mark = "✘" if iss.severity == "error" else "!"
        print(f"{mark} line {line}:{col:<4} {text[iss.start:iss.end]!r:<24} {iss.message}{fix}   [{iss.rule}]")


def main():
    ap = argparse.ArgumentParser(description="English grammar rules as plain logic.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "fix"):
        c = sub.add_parser(name, help=f"{name} grammar in some text")
        c.add_argument("text", nargs="?")
        c.add_argument("--file")
        c.add_argument("--errors-only", action="store_true", help="skip warnings")
    sub.add_parser("rules", help="list every rule with examples")
    g = sub.add_parser("generate", help="generate text with an ngram.py model, filtered by the rules")
    g.add_argument("model")
    g.add_argument("start", nargs="?", default="")
    g.add_argument("--words", type=int, default=60)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--seed", type=int)
    g.add_argument("--allow-junk", action="store_true", help="allow numbers and symbols")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    if args.cmd == "rules":
        for rid, sev, title, explain, bad, good, _ in RULES:
            print(f"[{sev}] {title}  ({rid})\n    {explain}\n    wrong: {bad}\n    right: {good}\n")
        print(f"{len(RULES)} rules · {len(IRR)} irregular verbs · {len(BASE_VERBS)} verbs checked for "
              f"he/she/it agreement")
        return
    if args.cmd in ("check", "fix"):
        text = open(args.file, encoding="utf-8").read() if args.file else args.text
        if text is None:
            sys.exit("Give some text or --file.")
        if args.cmd == "check":
            print_issues(text, check(text, warnings=not args.errors_only))
        else:
            print(fix_text(text, warnings=not args.errors_only))
        return
    if args.cmd == "generate":
        import ngram
        m = ngram.NGramModel(args.model)
        toks, vetoes = generate(m, args.start, args.words, args.temperature,
                                random.Random(args.seed), no_junk=not args.allow_junk)
        print(ngram.detokenize(toks))
        print(f"\n[grammar vetoed {len(vetoes)} word choices]", file=sys.stderr)
        for ctx, w, why in vetoes[:15]:
            print(f"   after '{ctx}': not '{w}'  ({why})", file=sys.stderr)


if __name__ == "__main__":
    main()
