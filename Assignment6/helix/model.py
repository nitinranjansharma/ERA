"""Small masked-attention LM + Nesterov SGD. Mask is applied before softmax."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helix.hashutil import sha256_bytes


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


@dataclass
class TinyLM:
    vocab_size: int
    d_model: int = 32
    seq_len: int = 64
    lr: float = 0.08
    momentum: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        s = 1.0 / np.sqrt(self.d_model)
        self.tok_emb = rng.normal(0, s, (self.vocab_size, self.d_model))
        self.pos_emb = rng.normal(0, s, (self.seq_len, self.d_model))
        self.W = rng.normal(0, s, (self.d_model, self.vocab_size))
        self.b = np.zeros(self.vocab_size)
        self.v_tok = np.zeros_like(self.tok_emb)
        self.v_pos = np.zeros_like(self.pos_emb)
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)
        self.opt_step = 0

    def params(self) -> list[np.ndarray]:
        return [self.tok_emb, self.pos_emb, self.W, self.b]

    def velocities(self) -> list[np.ndarray]:
        return [self.v_tok, self.v_pos, self.v_W, self.v_b]

    def weights_hash(self) -> str:
        return sha256_bytes(b"".join(p.astype(np.float64).tobytes() for p in self.params()))

    def optimizer_hash(self) -> str:
        blobs = [v.astype(np.float64).tobytes() for v in self.velocities()]
        blobs.append(int(self.opt_step).to_bytes(8, "little"))
        return sha256_bytes(b"".join(blobs))

    def state_dict(self) -> dict:
        return {
            "tok_emb": self.tok_emb.copy(),
            "pos_emb": self.pos_emb.copy(),
            "W": self.W.copy(),
            "b": self.b.copy(),
            "v_tok": self.v_tok.copy(),
            "v_pos": self.v_pos.copy(),
            "v_W": self.v_W.copy(),
            "v_b": self.v_b.copy(),
            "opt_step": self.opt_step,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "seq_len": self.seq_len,
            "lr": self.lr,
            "momentum": self.momentum,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("tok_emb", "pos_emb", "W", "b", "v_tok", "v_pos", "v_W", "v_b"):
            setattr(self, key, np.array(state[key], dtype=np.float64, copy=True))
        self.opt_step = int(state["opt_step"])

    def forward(
        self,
        token_ids: np.ndarray,
        position_ids: np.ndarray,
        attn_mask: np.ndarray,
        loss_mask: np.ndarray,
    ) -> tuple[float, np.ndarray, dict]:
        """Return mean loss, per-token loss [B,T], and cache for backward."""
        B, T = token_ids.shape
        E = self.tok_emb[token_ids] + self.pos_emb[np.clip(position_ids, 0, self.seq_len - 1)]
        scale = 1.0 / np.sqrt(self.d_model)
        scores = np.matmul(E, np.transpose(E, (0, 2, 1))) * scale
        scores = np.where(attn_mask, scores, -1e9)
        P = _softmax(scores, axis=-1)
        O = np.matmul(P, E)
        logits = O @ self.W + self.b
        labels = token_ids
        shifted = logits - logits.max(-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.clip(exp.sum(-1, keepdims=True), 1e-12, None)
        log_probs = np.log(np.clip(probs, 1e-12, None))
        tok_loss = -np.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
        tok_loss = np.where(loss_mask > 0, tok_loss, 0.0)
        n = max(1.0, float(loss_mask.sum()))
        mean = float(tok_loss.sum() / n)
        cache = {
            "E": E,
            "P": P,
            "O": O,
            "logits": logits,
            "probs": probs,
            "token_ids": token_ids,
            "position_ids": position_ids,
            "attn_mask": attn_mask,
            "loss_mask": loss_mask,
            "labels": labels,
            "n": n,
        }
        return mean, tok_loss, cache

    def train_step(
        self,
        token_ids: np.ndarray,
        position_ids: np.ndarray,
        attn_mask: np.ndarray,
        loss_mask: np.ndarray,
    ) -> tuple[float, np.ndarray, float]:
        mean, tok_loss, cache = self.forward(token_ids, position_ids, attn_mask, loss_mask)
        grad_norm = self._backward_and_step(cache)
        return mean, tok_loss, grad_norm

    def _backward_and_step(self, cache: dict) -> float:
        probs = cache["probs"]
        labels = cache["labels"]
        loss_mask = cache["loss_mask"]
        n = cache["n"]
        B, T, V = probs.shape
        dlogits = probs.copy()
        # one-hot minus
        flat = labels.reshape(-1)
        dlogits.reshape(-1, V)[np.arange(B * T), flat] -= 1.0
        dlogits *= (loss_mask / n)[..., None]

        O = cache["O"]
        dW = np.tensordot(O, dlogits, axes=([0, 1], [0, 1]))
        db = dlogits.sum(axis=(0, 1))
        dO = dlogits @ self.W.T

        E = cache["E"]
        P = cache["P"]
        # O = P @ E
        dP = np.matmul(dO, np.transpose(E, (0, 2, 1)))
        dE_from_v = np.matmul(np.transpose(P, (0, 2, 1)), dO)

        # softmax backward: dP/dscores
        # for each row, (diag(p) - p p^T) 
        sp = (dP * P).sum(axis=-1, keepdims=True)
        dscores = P * (dP - sp)
        dscores = np.where(cache["attn_mask"], dscores, 0.0)
        scale = 1.0 / np.sqrt(self.d_model)
        # scores = (E @ E^T) * scale
        dE_from_q = np.matmul(dscores, E) * scale
        dE_from_k = np.matmul(np.transpose(dscores, (0, 2, 1)), E) * scale
        dE = dE_from_v + dE_from_q + dE_from_k

        d_tok = np.zeros_like(self.tok_emb)
        d_pos = np.zeros_like(self.pos_emb)
        ids = cache["token_ids"]
        pos = np.clip(cache["position_ids"], 0, self.seq_len - 1)
        for b in range(B):
            for t in range(T):
                d_tok[ids[b, t]] += dE[b, t]
                d_pos[pos[b, t]] += dE[b, t]

        grads = [d_tok, d_pos, dW, db]
        vels = self.velocities()
        params = self.params()
        mu = self.momentum
        lr = self.lr
        gnorm = float(np.sqrt(sum(float((g * g).sum()) for g in grads)))
        self.opt_step += 1
        for p, g, v in zip(params, grads, vels):
            v *= mu
            v += g
            # Nesterov: p -= lr * (mu v + g)
            p -= lr * (mu * v + g)
        return gnorm


def pack_batch(sequences) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tok = np.stack([np.asarray(s.token_ids, dtype=np.int32) for s in sequences])
    pos = np.stack([np.asarray(s.position_ids, dtype=np.int32) for s in sequences])
    attn = np.stack([np.asarray(s.attn_mask, dtype=np.bool_) for s in sequences])
    loss = np.stack([np.asarray(s.loss_mask, dtype=np.float64) for s in sequences])
    return tok, pos, attn, loss
