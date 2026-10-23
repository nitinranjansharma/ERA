"""Lane names, special tokens, roles, and planted-failure markers."""

from __future__ import annotations

LANES = (
    "general_web",
    "code",
    "math_science",
    "indic",
    "agentic",
    "reasoning",
)

# Token roles. Supervised roles receive next-token loss; context roles do not.
ROLE_PAD = "pad"
ROLE_TEXT = "text"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_PLAN = "plan"
ROLE_TOOL = "tool_call"
ROLE_OBS = "observation"
ROLE_BOUNDARY = "boundary"

SUPERVISED_ROLES = frozenset({ROLE_TEXT, ROLE_ASSISTANT, ROLE_PLAN, ROLE_TOOL, ROLE_BOUNDARY})
CONTEXT_ROLES = frozenset({ROLE_USER, ROLE_OBS, ROLE_PAD})

SPECIAL_TOKENS = (
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    "<user>",
    "</user>",
    "<assistant>",
    "</assistant>",
    "<plan>",
    "</plan>",
    "<tool>",
    "</tool>",
    "<obs>",
    "</obs>",
    "<eot>",
)

PAD, EOS, BOS, UNK = "<pad>", "<eos>", "<bos>", "<unk>"

ROLE_WRAPPERS = {
    ROLE_USER: ("<user>", "</user>"),
    ROLE_ASSISTANT: ("<assistant>", "</assistant>"),
    ROLE_PLAN: ("<plan>", "</plan>"),
    ROLE_TOOL: ("<tool>", "</tool>"),
    ROLE_OBS: ("<obs>", "</obs>"),
}

PACKING_POLICIES = (
    "pad_only",
    "concat_chop",
    "greedy",
    "best_fit",
    "structure_preserving",
    "long_context",
)

LICENSE_ALLOWLIST = ("MIT", "Apache-2.0", "BSD-3-Clause", "CC-BY-4.0", "CC0-1.0")

# Planted markers used by the firewall and cleaner. Unique so they cannot arise
# accidentally in generated prose.
CANARY_STRING = "__EVAL_CANARY_9f3a__"
PII_EMAIL = "leak.user@example.com"
PII_PHONE = "+91-98765-43210"
EVAL_ITEM = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
EVAL_ANSWER = "72"

NGRAM_CONTAINMENT_N = 8
REPEAT_NGRAM_MIN_N = 2
REPEAT_NGRAM_MAX_N = 8
REPEAT_RUN_THRESHOLD = 24
SHINGLE_N = 5
JACCARD_DEDUP_THRESHOLD = 0.82
