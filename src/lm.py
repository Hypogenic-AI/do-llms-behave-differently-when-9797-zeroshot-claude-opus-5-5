"""Local model wrapper: chat formatting, A/B logit readout, generation, activation capture, steering."""
import contextlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODELS

try:  # torch>=2.14 routes some bmm calls through triton kernels, which need a C compiler at runtime
    from torch._native import registry as _reg
    _reg.deregister_op_overrides(disable_dsl_names="triton")
except Exception as _e:  # pragma: no cover
    print("could not disable triton overrides:", _e)


class LM:
    def __init__(self, key):
        name = MODELS[key]
        self.key = key
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.d = self.model.config.hidden_size
        # answer-letter token variants ("A" and " A"); the readout is a logsumexp over variants
        self.A = sorted({self.tok.encode(v, add_special_tokens=False)[-1] for v in ["A", " A"]})
        self.B = sorted({self.tok.encode(v, add_special_tokens=False)[-1] for v in ["B", " B"]})
        assert all(self.tok.decode([t]).strip() == "A" for t in self.A)
        assert all(self.tok.decode([t]).strip() == "B" for t in self.B)
        self._mask = None

    # ---------- formatting ----------
    def fmt(self, user, prefill=""):
        s = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                         add_generation_prompt=True)
        return s + prefill

    def _batches(self, texts, max_tokens=6000, max_bs=64):
        lens = [len(self.tok(t, add_special_tokens=False).input_ids) for t in texts]
        order = np.argsort(lens)[::-1]
        batch, cur = [], 0
        for i in order:
            L = lens[i]
            if batch and (max(cur, L) * (len(batch) + 1) > max_tokens or len(batch) >= max_bs):
                yield batch
                batch, cur = [], 0
            batch.append(i)
            cur = max(cur, L)
        if batch:
            yield batch

    def _enc(self, texts, offsets=False):
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False,
                        return_offsets_mapping=offsets)

    # ---------- steering ----------
    @contextlib.contextmanager
    def steer(self, spec):
        """spec: list of (layer, vector). Adds the vector to the residual stream (block output) at the token
        positions selected by self._mask (set per batch; None = all positions). During cached decoding steps
        (sequence length differs from the mask) nothing is added, so with a span mask only the prompt-text
        positions are modified."""
        hooks = []
        if spec:
            for layer, v in spec:
                v = torch.as_tensor(v, dtype=torch.bfloat16, device="cuda")

                def hook(mod, inp, out, v=v):
                    h = out[0] if isinstance(out, tuple) else out
                    m = self._mask
                    if m is None:
                        h = h + v
                    elif m.shape[1] == h.shape[1]:
                        h = h + m[..., None].to(h.dtype) * v
                    else:
                        return out
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                hooks.append(self.layers[layer].register_forward_hook(hook))
        try:
            yield
        finally:
            for h in hooks:
                h.remove()

    def _set_mask(self, enc_off, b, spans):
        if spans is None:
            self._mask = None
            return
        st = torch.tensor([spans[i][0] for i in b])[:, None]
        en = torch.tensor([spans[i][1] for i in b])[:, None]
        self._mask = ((enc_off[..., 1] > st) & (enc_off[..., 0] < en)).to("cuda")

    # ---------- readouts ----------
    @torch.no_grad()
    def ab(self, prompts, steer=None, spans=None, return_mass=False):
        """logit(A) - logit(B) for the next token after each (already formatted, prefilled) prompt.
        With return_mass, also the total probability on the A/B tokens (a degradation check)."""
        out = np.zeros(len(prompts), dtype=np.float32)
        mass = np.zeros(len(prompts), dtype=np.float32)
        with self.steer(steer):
            for b in self._batches(prompts):
                enc = self._enc([prompts[i] for i in b], offsets=True)
                self._set_mask(enc.pop("offset_mapping"), b, spans)
                enc = enc.to("cuda")
                lg = self.model(**enc).logits[:, -1].float()
                out[b] = (lg[:, self.A].logsumexp(-1) - lg[:, self.B].logsumexp(-1)).cpu().numpy()
                lp = lg.log_softmax(-1)
                mass[b] = lp[:, self.A + self.B].exp().sum(-1).cpu().numpy()
        return (out, mass) if return_mass else out

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=80, steer=None, progress=False, spans=None):
        outs = [None] * len(prompts)
        bl = list(self._batches(prompts, max_tokens=8000, max_bs=96))
        if progress:
            from tqdm import tqdm
            bl = tqdm(bl, mininterval=30)
        with self.steer(steer):
            for b in bl:
                enc = self._enc([prompts[i] for i in b], offsets=True)
                self._set_mask(enc.pop("offset_mapping"), b, spans)
                enc = enc.to("cuda")
                g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                        pad_token_id=self.tok.pad_token_id)
                txt = self.tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
                for i, t in zip(b, txt):
                    outs[i] = t
        return outs

    @torch.no_grad()
    def acts(self, prompts, spans, layers, steer=None, steer_spans=None):
        """Residual activations (block outputs) at `layers`.
        Returns last[n, len(layers), d] (final prompt token) and mean[n, len(layers), d] (mean over tokens
        inside the character span spans[i] = (start, end) of each prompt)."""
        n = len(prompts)
        last = np.zeros((n, len(layers), self.d), dtype=np.float16)
        mean = np.zeros((n, len(layers), self.d), dtype=np.float16)
        with self.steer(steer):
            for b in self._batches(prompts, max_tokens=5000, max_bs=48):
                enc = self._enc([prompts[i] for i in b], offsets=True)
                off = enc.pop("offset_mapping")
                self._set_mask(off, b, steer_spans)
                enc = enc.to("cuda")
                hs = self.model(**enc, output_hidden_states=True).hidden_states
                st = torch.tensor([spans[i][0] for i in b])[:, None]
                en = torch.tensor([spans[i][1] for i in b])[:, None]
                m = ((off[..., 1] > st) & (off[..., 0] < en) & (enc.attention_mask.cpu() > 0)).to("cuda")
                m = m.float() / m.sum(1, keepdim=True).clamp(min=1)
                for j, L in enumerate(layers):
                    h = hs[L + 1].float()
                    last[b, j] = h[:, -1].cpu().numpy().astype(np.float16)
                    mean[b, j] = (h * m[..., None]).sum(1).cpu().numpy().astype(np.float16)
        return last, mean

    def span(self, formatted, text):
        text = text.strip()  # some chat templates strip surrounding whitespace
        i = formatted.rfind(text)
        assert i >= 0, text[:80]
        return (i, i + len(text))
