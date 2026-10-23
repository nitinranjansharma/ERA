"""Synthetic six-lane corpus with planted admission and firewall failures.

Documents are the human unit. Training never sees them raw: they are cleaned,
tokenized, sealed into shards, then packed. Plants exist so a gate that is not
load-bearing will fail a negative control rather than look green.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from helix.constants import (
    CANARY_STRING,
    EVAL_ANSWER,
    EVAL_ITEM,
    PII_EMAIL,
    PII_PHONE,
    ROLE_ASSISTANT,
    ROLE_OBS,
    ROLE_PLAN,
    ROLE_TEXT,
    ROLE_TOOL,
    ROLE_USER,
)


@dataclass
class Document:
    doc_id: str
    lane: str
    segments: list[tuple[str, str]]
    language: str
    script: str
    license: str
    source_id: str
    provenance_tier: str = "A"
    held_out: bool = False
    never_train: bool = False
    reserved_for_anneal: bool = False
    plant: str | None = None
    notes: str = ""
    parent_doc_id: str | None = None

    @property
    def text(self) -> str:
        return "".join(body for _, body in self.segments)


def _text(doc_id: str, lane: str, body: str, **kw) -> Document:
    lang = kw.pop("language", "en")
    script = kw.pop("script", "Latn")
    license_ = kw.pop("license", "Apache-2.0")
    source = kw.pop("source_id", f"src-{lane}")
    return Document(
        doc_id=doc_id,
        lane=lane,
        segments=[(ROLE_TEXT, body)],
        language=lang,
        script=script,
        license=license_,
        source_id=source,
        **kw,
    )


def _agentic(doc_id: str, user: str, plan: str, tool: str, obs: str, answer: str, **kw) -> Document:
    return Document(
        doc_id=doc_id,
        lane="agentic",
        segments=[
            (ROLE_USER, user),
            (ROLE_PLAN, plan),
            (ROLE_TOOL, tool),
            (ROLE_OBS, obs),
            (ROLE_ASSISTANT, answer),
        ],
        language=kw.pop("language", "en"),
        script=kw.pop("script", "Latn"),
        license=kw.pop("license", "MIT"),
        source_id=kw.pop("source_id", "src-agentic"),
        **kw,
    )


def build_corpus() -> list[Document]:
    docs: list[Document] = []

    web = [
        "The monsoon shapes agriculture across the Indian subcontinent. Farmers time sowing to the first sustained rains and keep a second crop in reserve if the onset is late.",
        "Open weights do not make a model open. A release is open when the data mixture, the training code, the tokenizer, and the evaluation protocol can be inspected and replayed.",
        "A city library keeps three ledgers: who borrowed a book, when it was returned, and which copies were never loaned because they are reference-only. Training data needs the same split.",
        "Packing short documents into a long context window is a budget decision. Every pad token is a token the model did not learn from, even if the loss mask ignores it.",
        "Byte-pair encoding is convenient for English and brittle for conjunct-rich scripts unless normalization happens before the merge table is frozen.",
        "Curriculum learning for language models is not the same as shuffling. Early stages need diverse easy text; later stages need long arguments and tool traces.",
        "A checkpoint without a data position is incomplete. Restoring weights and then drawing a fresh shuffle silently changes the experiment.",
        "Web text often concatenates unrelated pages. An EOS token is the model's only signal that one author stopped and another began.",
        "Throughput numbers that count pad positions as tokens processed will look healthy while learning stalls. Report loss-bearing tokens per second.",
        "Protected floors exist because a scarce language can be starved by a greedy utility selector. The floor is a promise to the next version of the corpus.",
        "Synthetic data is not free data. It inherits the biases of the generator and must carry a parent hash so the lineage is visible at audit time.",
        "Long-context training is expensive. Unused positions in a 32k window waste a high-value opportunity, which is why long-context packing is a separate policy.",
    ]
    for i, body in enumerate(web):
        docs.append(_text(f"web-{i:04d}", "general_web", body, provenance_tier="B"))

    code = [
        "def fibonacci(n):\n    if n < 2:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n",
        "def tokenize(text, vocab):\n    tokens = []\n    i = 0\n    while i < len(text):\n        match = None\n        for tok in vocab:\n            if text.startswith(tok, i):\n                if match is None or len(tok) > len(match):\n                    match = tok\n        if match is None:\n            tokens.append(text[i]); i += 1\n        else:\n            tokens.append(match); i += len(match)\n    return tokens\n",
        "class HashChain:\n    def __init__(self, genesis):\n        self.prev = genesis\n        self.events = []\n    def append(self, payload, digest_fn):\n        event_hash = digest_fn({'prev': self.prev, 'payload': payload})\n        self.events.append((event_hash, self.prev, payload))\n        self.prev = event_hash\n        return event_hash\n",
        "def packed_attention_mask(segment_ids):\n    n = len(segment_ids)\n    mask = [[False]*n for _ in range(n)]\n    for i in range(n):\n        for j in range(i+1):\n            if segment_ids[i] == segment_ids[j] and segment_ids[i] >= 0:\n                mask[i][j] = True\n    return mask\n",
        "def greedy_pack(lengths, capacity):\n    bins = []\n    for L in lengths:\n        for b in bins:\n            if sum(b) + L <= capacity:\n                b.append(L); break\n        else:\n            bins.append([L])\n    return bins\n",
        "def best_fit_pack(lengths, capacity):\n    order = sorted(lengths, reverse=True)\n    bins = []\n    remain = []\n    for L in order:\n        choice = -1; best = capacity + 1\n        for i, r in enumerate(remain):\n            if L <= r < best:\n                best = r; choice = i\n        if choice < 0:\n            bins.append([L]); remain.append(capacity - L)\n        else:\n            bins[choice].append(L); remain[choice] -= L\n    return bins\n",
        "SELECT shard_id, SUM(loss_tokens) AS useful\nFROM consumption_ledger\nWHERE branch_id = 'run-a'\nGROUP BY shard_id\nORDER BY useful DESC;\n",
        "def nfc_graphemes(text):\n    import unicodedata\n    text = unicodedata.normalize('NFC', text)\n    out = []\n    for ch in text:\n        if out and unicodedata.combining(ch):\n            out[-1] += ch\n        else:\n            out.append(ch)\n    return out\n",
        "def resume_cursor(ledger_offset, planned):\n    if ledger_offset != len(planned):\n        raise ValueError('ledger and planner disagree')\n    return planned[ledger_offset:]\n",
        "fn pack_spans(spans: &[u32], cap: u32) -> Vec<Vec<u32>> {\n    let mut bins = vec![];\n    for &s in spans {\n        if let Some(bin) = bins.iter_mut().find(|b| b.iter().sum::<u32>() + s <= cap) {\n            bin.push(s);\n        } else { bins.push(vec![s]); }\n    }\n    bins\n}\n",
    ]
    for i, body in enumerate(code):
        docs.append(_text(f"code-{i:04d}", "code", body, source_id="src-code", provenance_tier="A"))

    math = [
        "A prime number has exactly two distinct positive divisors, one and itself. Euclid's proof that there are infinitely many primes assumes a finite list and constructs a new prime from their product plus one.",
        "The derivative of sin(x) is cos(x). The chain rule says d/dx f(g(x)) = f'(g(x)) g'(x). Together they give the derivative of sin(x^2) as 2x cos(x^2).",
        "Bayes' theorem: P(H|E) = P(E|H) P(H) / P(E). A high likelihood does not make a rare hypothesis probable unless the evidence is extremely specific.",
        "The entropy of a discrete distribution is H = -sum p log p. Cross-entropy of a model q against true p is -sum p log q, which is the token loss used in language modelling.",
        "A vector space over the reals is closed under addition and scalar multiplication. The standard basis of R^n is orthonormal under the dot product.",
        "Induction: prove a base case n=1, assume the claim for n=k, prove it for n=k+1. The set of integers where it holds is then all positive integers.",
        "A Markov chain has the memoryless property: the next state depends only on the current state. Language models are not Markov of small order; that is why we pack long windows.",
        "The L2 norm of a gradient is a crude spike detector. A large norm at a curriculum seam is more often a mixture change than a poisonous document.",
        "Token perplexity is exp(cross_entropy). A shard with acceptable average loss can still hide a conjunct pattern that remains high-perplexity.",
        "Conservation of expected evidence: on average, a Bayesian cannot expect to become more confident without data. Training is the data.",
    ]
    for i, body in enumerate(math):
        docs.append(_text(f"math-{i:04d}", "math_science", body, source_id="src-stem"))

    indic = [
        "भारत एक विविधताओं से भरा देश है। यहाँ अनेक भाषाएँ, लिपियाँ और साहित्यिक परम्पराएँ साथ-साथ जीवित हैं।",
        "हिन्दी में मात्राएँ स्वर के बाद लगती हैं। सामान्यीकरण के बिना टोकनाइज़र मात्रा को अलग चिह्न मान सकता है।",
        "शिक्षा का उद्देश्य केवल परीक्षा पास करना नहीं है। समझ के साथ अभ्यास से ज्ञान स्थिर होता है।",
        "मानसून की पहली बारिश किसानों के लिए संकेत है। यदि वर्षा देर से आए तो दूसरी फसल का योजना बनाना पड़ता है।",
        "भाषा मॉडल को देवनागरी के संयुक्त व्यंजनों का सम्मान करना चाहिए। गलत काट-छाँट से अर्थ टूट जाता है।",
        "गंगा मैदान की मिट्टी उपजाऊ है। नदियों के किनारे बसे नगर व्यापार और संस्कृति के केन्द्र रहे हैं।",
        "ताज महल संगमरमर से बना है। यह आगरा में यमुना नदी के किनारे स्थित है और मुगल स्थापत्य का उदाहरण है।",
        "कम्प्यूटर विज्ञान में आंकड़ों की वंशावली उतनी ही महत्वपूर्ण है जितनी मॉडल की परतें।",
        "संस्कृत में सन्धि नियमों से शब्द जुड़ते हैं। आधुनिक हिन्दी ने कई रूप सरल कर लिये हैं पर मूल ध्वनि बनी हुई है।",
        "बच्चों की कविताएँ छोटी होती हैं पर उनमें लय और दोहराव होता है। दोहराव सीखने में सहायक है, प्रशिक्षण में हानिकारक भी हो सकता है।",
        "केरल की हरित पहाड़ियाँ और बंगाल की नदियाँ अलग जलवायु दिखाती हैं। एक मॉडल को दोनों का वर्णन करना आना चाहिए।",
        "अनुवाद करते समय मुहावरे शब्दशः नहीं चलते। 'दांत पीसना' का अर्थ क्रोध है, दंत चिकित्सा नहीं।",
    ]
    for i, body in enumerate(indic):
        reserved = i >= 10
        docs.append(
            _text(
                f"indic-{i:04d}",
                "indic",
                body,
                language="hi",
                script="Deva",
                source_id="src-indic",
                provenance_tier="A",
                reserved_for_anneal=reserved,
            )
        )

    agentic_specs = [
        (
            "List files in the current directory.",
            "I will call the shell tool with ls.",
            "ls -1",
            "README.md\nhelix/\nplan/",
            "The directory contains README.md, helix/, and plan/.",
        ),
        (
            "Compute the sha256 of the string helix.",
            "I will hash the utf-8 bytes of helix with sha256.",
            "sha256('helix')",
            "3b4d2e1a9c0f",
            "The sha256 prefix of helix is 3b4d2e1a9c0f.",
        ),
        (
            "Find the next batch after checkpoint 8.",
            "I will read the consumption ledger at offset 8 and return the following sample id.",
            "ledger.next(offset=8)",
            "sample:8f2c",
            "The next sample after checkpoint 8 is 8f2c.",
        ),
        (
            "Translate 'protected floor' into Hindi and keep the technical sense.",
            "I will choose a gloss that preserves the quota meaning rather than architecture.",
            "translate(protected floor, hi)",
            "संरक्षित न्यूनतम हिस्सा",
            "A close technical gloss is संरक्षित न्यूनतम हिस्सा.",
        ),
        (
            "Pack three examples of lengths 7, 5 and 5 into windows of 10.",
            "I will try best-fit decreasing and report the bin count.",
            "best_fit([7,5,5], 10)",
            "[[7],[5,5]]",
            "Best-fit decreasing uses two windows: [7] and [5,5].",
        ),
        (
            "Is evaluation data allowed in a training microbatch?",
            "I will check the eval firewall registry before answering.",
            "firewall.permits(shard)",
            "never_train=true",
            "No. Evaluation shards are registered and blocked from gradient-bearing batches.",
        ),
        (
            "Resume training after a crash at step 16.",
            "I will restore the checkpoint and bind the dataloader to the ledger offset.",
            "restore(step=16)",
            "ledger_offset=16 branch=run-a",
            "Resume from step 16 on branch run-a at ledger offset 16.",
        ),
        (
            "Count Devanagari graphemes in नमस्ते.",
            "I will NFC-normalize and cluster combining marks.",
            "graphemes('नमस्ते')",
            "न म स् ते",
            "नमस्ते clusters into four graphemes under NFC.",
        ),
    ]
    for i, spec in enumerate(agentic_specs):
        reserved = i >= 6
        docs.append(_agentic(f"agentic-{i:04d}", *spec, reserved_for_anneal=reserved, provenance_tier="A"))

    reasoning = [
        "Question: 17 + 25. Step 1: 17 + 20 = 37. Step 2: 37 + 5 = 42. Answer: 42.",
        "Question: A train travels 120 km in 2 hours. Step 1: speed = distance / time. Step 2: 120 / 2 = 60. Answer: 60 km/h.",
        "Question: If 3x + 5 = 20, find x. Step 1: 3x = 15. Step 2: x = 5. Answer: 5.",
        "Question: How many 8-token windows fit in 64 tokens with no padding? Step 1: 64 / 8 = 8. Answer: 8.",
        "Question: A fair coin is flipped twice. Probability of two heads? Step 1: each flip is 1/2. Step 2: independent, so 1/4. Answer: 1/4.",
        "Question: 2^5. Step 1: 2^3 = 8. Step 2: 8 * 4 = 32. Answer: 32.",
        "Question: GCD of 48 and 18. Step 1: 48 = 2*18 + 12. Step 2: 18 = 1*12 + 6. Step 3: 12 = 2*6 + 0. Answer: 6.",
        "Question: Area of a rectangle 7 by 9. Step 1: area = width * height. Step 2: 7*9 = 63. Answer: 63.",
        "Question: Convert 0.75 to a fraction. Step 1: 75/100. Step 2: divide by 25. Answer: 3/4.",
        "Question: 9 * 8 + 4. Step 1: 9*8 = 72. Step 2: 72+4 = 76. Answer: 76.",
    ]
    for i, body in enumerate(reasoning):
        reserved = i >= 8
        docs.append(
            _text(
                f"reason-{i:04d}",
                "reasoning",
                body,
                source_id="src-reason",
                provenance_tier="A",
                reserved_for_anneal=reserved,
            )
        )

    # --- planted failures (must never become loss-bearing training data) ---
    docs.append(
        _text(
            "plant-pii-email",
            "general_web",
            f"Please contact the dataset owner at {PII_EMAIL} for access to the raw crawl.",
            plant="pii",
            notes="email address must fail the PII screen",
        )
    )
    docs.append(
        _text(
            "plant-pii-phone",
            "general_web",
            f"Call the annotator on {PII_PHONE} after shift.",
            plant="pii",
            notes="phone number must fail the PII screen",
        )
    )
    docs.append(
        _text(
            "plant-license-nc",
            "general_web",
            "This page is reproduced from a magazine that forbids commercial training use.",
            license="CC-BY-NC-4.0",
            plant="license",
            notes="non-commercial license is not on the allowlist",
        )
    )
    docs.append(
        _text(
            "plant-ngram-repeat",
            "general_web",
            "xy" * 60,
            plant="repeat_ngram",
            notes="long repeated n-gram run; unique so Jaccard will not catch it",
        )
    )
    docs.append(
        _text(
            "plant-eval-gsm",
            "math_science",
            f"{EVAL_ITEM} The answer is {EVAL_ANSWER}.",
            held_out=True,
            never_train=True,
            plant="eval_item",
            notes="verbatim benchmark item; registered as a test shard",
        )
    )
    docs.append(
        _text(
            "plant-canary",
            "general_web",
            f"An otherwise ordinary article that smuggles {CANARY_STRING} into the body.",
            held_out=True,
            never_train=True,
            plant="canary",
            notes="canary string registered with the eval firewall",
        )
    )
    docs.append(
        _text(
            "plant-containment-host",
            "general_web",
            "A long magazine feature on small-town shops discusses inventory. "
            "In the third section the author quotes a grade-school puzzle in full: "
            f"{EVAL_ITEM} Readers are invited to pause. The rest of the article is local colour "
            "about monsoon markets, tea stalls, and the evening train.",
            plant="ngram_containment",
            notes="Jaccard against the eval item is low; 8-gram containment must catch it",
        )
    )
    docs.append(
        _text(
            "plant-indic-wrong-script",
            "indic",
            "This document is tagged Indic but is written entirely in Latin letters with no Devanagari.",
            language="hi",
            script="Deva",
            plant="language",
            notes="language/script validator must reject Latin text in the Indic lane",
        )
    )
    docs.append(
        _text(
            "plant-val-holdout",
            "code",
            "def validation_only():\n    return 'this function is a canary for the validation split'\n",
            held_out=True,
            never_train=True,
            plant="validation",
            notes="validation shard may be read for eval, never for gradients",
        )
    )

    # Exact duplicate of web-0000: content-hash dedup must drop it.
    original = next(d for d in docs if d.doc_id == "web-0000")
    docs.append(
        Document(
            doc_id="plant-exact-dup",
            lane="general_web",
            segments=list(original.segments),
            language=original.language,
            script=original.script,
            license=original.license,
            source_id="src-web-mirror",
            plant="exact_dup",
            notes="byte-identical to web-0000",
            parent_doc_id="web-0000",
        )
    )

    # Near-duplicate with a tiny paraphrase tail; shingle Jaccard should still fire.
    docs.append(
        _text(
            "plant-near-dup",
            "general_web",
            original.text + " Indeed.",
            plant="near_dup",
            notes="near-duplicate of web-0000",
            parent_doc_id="web-0000",
        )
    )

    return docs


def all_plain_text(docs: list[Document]) -> list[str]:
    texts = [d.text for d in docs]
    # Include wrapper tokens so they appear in the frozen vocab even if unused.
    texts.append(" ".join(["<user>", "<assistant>", "<plan>", "<tool>", "<obs>"]))
    return texts
