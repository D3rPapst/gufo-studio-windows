"""YuE2 reference dumps in FP32 on the CPU, streaming the weights layer by layer.

Uses the official yue2_infer 0.1.6 modules (modeling_yue2, nar, sampling,
protocol, modeling_vae). Only the weight loading differs from the official
pipeline: tensors are read unbuffered from model.safetensors, the weight set
a stage needs stays in RAM as BF16 and each layer is upcast to FP32 when it
runs, so a stage peaks below 5 GB (the 3.58B model needs 14 GB in FP32). The
values are those of an FP32 from_pretrained of the BF16 checkpoint. Written with AI assistance (Claude Code).

Usage (a venv with yue2_infer 0.1.6 and its pinned CPU torch, Windows only
because of the unbuffered reader):
  set YUE2_MODEL=.../YuE2-3B  YUE2_VAE=.../YuE2-Vae  YUE2_ORACLE=out-dir
  python tools/yue2/oracle.py tok vae      (seconds)
  python tools/yue2/oracle.py ar           (about 1 min, < 4.5 GB RAM)
  python tools/yue2/oracle.py nar          (about 2 min, < 4.7 GB RAM)
  python tools/yue2/oracle.py vae          (again: decodes the NAR latents)
One stage per process keeps the peak low.
"""
import glob, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

from yue2 import modeling_yue2 as M
from yue2 import nar as N
from yue2 import sampling as S
from yue2 import protocol as P
from yue2.tokenization_yue2 import YuE2TextTokenizer

MODEL = os.environ.get("YUE2_MODEL", "YuE2-3B")
VAE = os.environ.get("YUE2_VAE", "YuE2-Vae")
OUT = os.environ.get("YUE2_ORACLE", "yue2-oracle")
torch.set_num_threads(int(os.environ.get("ORACLE_THREADS", "8")))
torch.set_grad_enabled(False)

STYLE = "Pop ballad, female vocal, piano, warm, emotional, slow tempo, English"
LYRICS = "[Verse]\nThe city sleeps beneath the rain\nI hear your voice call out my name\n\n[Chorus]\nHold on, hold on to me\n"
GREEDY = int(os.environ.get("ORACLE_STEPS", "64"))


def save(name, array):
    np.save(f"{OUT}/{name}.npy", np.ascontiguousarray(array))


import ctypes as _ct


class Overlapped(_ct.Structure):
    _fields_ = [("Internal", _ct.c_size_t), ("InternalHigh", _ct.c_size_t),
                ("Offset", _ct.c_uint32), ("OffsetHigh", _ct.c_uint32),
                ("hEvent", _ct.c_void_p)]


class NoCacheFile:
    """Unbuffered reads (FILE_FLAG_NO_BUFFERING): the 7 GB checkpoint must
    neither be mapped nor fill the Windows file cache on the 16 GB host."""

    def __init__(self, path):
        import ctypes
        from ctypes import wintypes
        self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k.CreateFileW.restype = wintypes.HANDLE
        self.k.VirtualAlloc.restype = ctypes.c_void_p
        self.ctypes = ctypes
        self.h = self.k.CreateFileW(path, 0x80000000, 1, None, 3, 0x20000000, None)
        if self.h in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_last_error())
        self.size = 64 << 20
        self.buf = self.k.VirtualAlloc(None, self.size, 0x3000, 4)

    def read(self, offset, count):
        ct = self.ctypes
        out = bytearray(count)
        pos = 0
        while pos < count:
            start = offset + pos
            aligned = start & ~4095
            skip = start - aligned
            want = min(self.size, (skip + count - pos + 4095) & ~4095)
            got = ct.c_uint32(0)
            ov = Overlapped(0, 0, aligned & 0xFFFFFFFF, aligned >> 32, None)
            if not self.k.ReadFile(ct.c_void_p(self.h), ct.c_void_p(self.buf), want, ct.byref(got), ct.byref(ov)):
                raise OSError(ct.get_last_error())
            take = min(got.value - skip, count - pos)
            if take <= 0:
                raise OSError("short read")
            out[pos:pos + take] = ct.string_at(self.buf + skip, take)
            pos += take
        return out


class Weights:
    """BF16 safetensors tensors read on demand and upcast to FP32 (the values
    an FP32 from_pretrained of this BF16 checkpoint holds)."""

    def __init__(self, path):
        import struct
        self.f = NoCacheFile(path)
        n = struct.unpack("<Q", bytes(self.f.read(0, 8)))[0]
        self.header = json.loads(bytes(self.f.read(8, n)))
        self.base = 8 + n
        self.kept = {}
        self.cfg = M.YuE2Config(**json.load(open(MODEL + "/config.json")))

    keep = None  # predicate: names kept in RAM as BF16

    def t(self, name, rows=None):
        if rows is None and name in self.kept:
            return self.kept[name].float()
        info = self.header[name]
        assert info["dtype"] == "BF16"
        shape = list(info["shape"])
        a, b = info["data_offsets"]
        if rows is not None:
            width = int(np.prod(shape[1:])) * 2
            a, b = a + rows[0] * width, a + rows[1] * width
            shape[0] = rows[1] - rows[0]
        data = self.f.read(self.base + a, b - a)
        raw = torch.frombuffer(data, dtype=torch.bfloat16).reshape(shape)
        if rows is None and self.keep is not None and self.keep(name):
            self.kept[name] = raw
        return raw.float()

    def bf16(self, name):
        if name not in self.kept:
            info = self.header[name]
            a, b = info["data_offsets"]
            self.kept[name] = torch.frombuffer(self.f.read(self.base + a, b - a),
                                               dtype=torch.bfloat16).reshape(info["shape"])
        return self.kept[name]

    def module(self, mod, prefix, only=None):
        keys = [k for k in mod.state_dict() if only is None or only(k)]
        state = {k: self.t(prefix + k) for k in keys}
        mod.load_state_dict(state, strict=only is None)
        return mod


class Model:
    """Layer-streaming FP32 view of YuE2ForCausalLM (same module classes)."""

    def __init__(self):
        self.w = Weights(MODEL + "/model.safetensors")
        c = self.cfg = self.w.cfg
        self.rotary = M.RotaryEmbedding(c.head_dim, c.rope_theta)
        self.norm = self.w.module(M.RMSNorm(c.hidden_size, c.rms_norm_eps), "model.norm.")
        self.layer = M.DecoderLayer(c).float().eval()
        self.loaded = None
        self.vae2llm = self.w.module(torch.nn.Linear(c.latent_dim, c.hidden_size), "vae2llm.")
        self.llm2vae = self.w.module(torch.nn.Linear(c.hidden_size, c.latent_dim), "llm2vae.")
        self.time_embedder = self.w.module(M.TimestepEmbedder(c.hidden_size), "time_embedder.")
        self.latent_pos_embed = M.AudioPositionEmbedding(c.max_latent_frames, c.hidden_size)
        # The checkpoint stores the sinusoid table (BF16); from_pretrained loads it.
        # Only the rows a short chunk uses are read (pos_rows).
        self.latent_pos_embed.pe = self.w.t("latent_pos_embed.pe", (0, 512))

    @property
    def config(self):
        return self.cfg

    def load_layer(self, i, half="ar"):
        """Loads one layer's AR or NAR weight set (the other set is unused)."""
        if self.loaded != (i, half):
            nar = half == "nar"
            self.w.module(self.layer, f"model.layers.{i}.",
                          only=lambda k: k.startswith("nar_") == nar)
            self.loaded = (i, half)
        return self.layer

    def embed(self, ids):
        return torch.stack([self.w.t("model.embed_tokens.weight", (int(t), int(t) + 1))[0]
                            for t in ids])[None]

    def logits(self, h):  # h [1, 2048] after the final norm
        parts = []
        for a in range(0, self.cfg.vocab_size, 16384):
            b = min(a + 16384, self.cfg.vocab_size)
            parts.append(h @ self.w.bf16("lm_head.weight")[a:b].float().T)
        return torch.cat(parts, -1)

    def _shift_t_value(self, t, device, dtype):
        return M.YuE2ForCausalLM._shift_t_value(self, t, device, dtype)


def ar_forward(model, ids, cache, start):
    """modeling_yue2 AR path (DecoderLayer with ar_mask=None) for new ids."""
    x = model.embed(ids)
    pos = torch.arange(start, start + len(ids))[None]
    cos, sin = model.rotary(pos)
    for i in range(model.cfg.num_hidden_layers):
        layer = model.load_layer(i)
        x = layer(x, cos, sin, cache, layer_idx=i, attention_mask=None, ar_mask=None,
                  cache_position=torch.arange(start, start + len(ids)))
    return model.norm(x)[:, -1]


def greedy(model, prefix, phase, sampling, tag):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    t0 = time.time()
    h = ar_forward(model, prefix, cache, 0)
    logits = model.logits(h)
    save(f"{tag}_prefix_logits", logits[0].numpy().astype(np.float32))
    save(f"{tag}_prefix_hidden", h[0].numpy().astype(np.float32))
    history, tokens, pos = [], [], len(prefix)
    end = P.ABC_END if phase == "abc" else P.MUSIC_END
    for step in range(GREEDY):
        scores = S.distribution(logits, sampling, history, step, phase)
        token = int(scores.argmax(-1))
        tokens.append(token)
        if token == end:
            break
        history.append(token)
        if step + 1 == GREEDY:
            break
        h = ar_forward(model, [token], cache, pos)
        pos += 1
        logits = model.logits(h)
        print(f"  {tag} step {step} token {token} ({time.time()-t0:.0f}s)", flush=True)
    save(f"{tag}_greedy", np.asarray(tokens, dtype=np.int32))
    return tokens


def stage_tok():
    tok = YuE2TextTokenizer(MODEL + "/qwen.tiktoken")
    texts = [STYLE, LYRICS,
             "Grüße aus Köln – Straße, Mädchen, Übermut! 123 4567 ½ 日本語 \r\n\t  x",
             P.INSTRUCTIONS["full"],
             "X:1\nL:1/8\nM:4/4\nK:C\n|: C2 E2 G2 c2 | \"Am\" A4 z4 :|\n",
             "[Verse]\nIch geh durch die Nacht, die Stadt ist still\n",
             "it's <|endoftext|> <abc> literal"]
    out = {"texts": texts, "ids": [tok.encode(t) for t in texts]}
    req = P.SongRequest(style=STYLE, lyrics=LYRICS, cot="full", seed=1234)
    out["prefix_full"] = P.token_prefixes(req, tok)
    out["prefix_off"] = P.token_prefixes(P.SongRequest(style=STYLE, lyrics=LYRICS, cot="off"), tok)
    out["prefix_melody"] = P.token_prefixes(P.SongRequest(style=STYLE, lyrics=LYRICS, cot="melody"), tok)
    json.dump(out, open(f"{OUT}/tokens.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("tok", [len(i) for i in out["ids"]], len(out["prefix_full"]))


def stage_ar():
    tok = YuE2TextTokenizer(MODEL + "/qwen.tiktoken")
    model = Model()
    model.w.keep = lambda n: n.startswith("model.layers.") and ".nar_" not in n
    req = P.SongRequest(style=STYLE, lyrics=LYRICS, cot="full", seed=1234)
    prefix = P.token_prefixes(req, tok)
    cfg = P.GenerationConfig()
    abc_sampling = P.Sampling(0.0, 1.0, cfg.abc.top_k, cfg.abc.repetition_penalty,
                              cfg.abc.penalty_window, cfg.abc.min_tokens, GREEDY)
    if os.path.exists(f"{OUT}/abc_greedy.npy") and not os.environ.get("ORACLE_REDO_ABC"):
        abc = np.load(f"{OUT}/abc_greedy.npy").tolist()
    else:
        abc = greedy(model, prefix, "abc", abc_sampling, "abc")
    abc_ids = [t for t in abc if t != P.ABC_END]
    sem_prefix = P.token_prefixes(req, tok, abc_ids)
    # min_tokens 200 > GREEDY: the end token stays suppressed for every step.
    sem_sampling = P.Sampling(0.0, 1.0, cfg.semantic.top_k, cfg.semantic.repetition_penalty,
                              cfg.semantic.penalty_window, GREEDY, GREEDY)
    sem = greedy(model, sem_prefix, "semantic", sem_sampling, "sem")
    json.dump({"abc_prefix": prefix, "abc_greedy": abc, "sem_prefix": sem_prefix,
               "sem_greedy": sem, "abc_text": tok.decode(abc_ids)},
              open(f"{OUT}/ar.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


class StreamingNAR(N.CachedNAR):
    """nar.CachedNAR with its layer loops over streamed layers (same math)."""

    def __init__(self, model, chunk):
        self.model, self.chunk = model, chunk
        self.backend, self.query_chunk_size = "sdpa", None
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.ar_length, self.nar_length = len(chunk.ar_tokens), len(chunk.noise) + 2
        self.visible_length = self.ar_length
        positions = torch.arange(self.ar_length, self.ar_length + self.nar_length)[None]
        self.cos, self.sin = model.rotary(positions)
        local = torch.arange(self.nar_length).clamp(max=model.cfg.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed(local)[None]
        self.cache = []
        self._prefill()

    def _prefill(self):
        m = self.model
        positions = torch.arange(self.ar_length)[None]
        cos, sin = m.rotary(positions)
        x = m.embed(self.chunk.ar_tokens)
        for i in range(m.cfg.num_hidden_layers):
            layer = m.load_layer(i)
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            self.cache.append((k[0].clone(), v[0].clone()))
            h = self._attention(q[0], k[0], v[0], causal=True)
            x = x + layer.self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.mlp(layer.post_attention_layernorm(x))

    def velocity(self, state, raw_t):
        m = self.model
        x_nar = F.pad(state, (0, 0, 1, 1))
        shifted = m._shift_t_value(raw_t, self.device, self.dtype)
        x = m.vae2llm(x_nar[None])
        x = x + m.time_embedder(shifted.expand(self.nar_length))[None]
        x = x + self.pos_emb
        for i, (ar_k, ar_v) in enumerate(self.cache):
            layer = m.load_layer(i, "nar")
            q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), self.cos, self.sin)
            k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))
            h = self._attention(q[0], k, v)
            x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        return m.llm2vae(m.norm(x))[0, 1:-1]


def stage_nar():
    ar = json.load(open(f"{OUT}/ar.json", encoding="utf-8"))
    model = Model()
    model.w.keep = lambda n: n.startswith("model.layers.") and ".nar_" in n
    prefix = ar["sem_prefix"]
    codec = [t - P.CODEC_OFFSET for t in ar["sem_greedy"] if t != P.MUSIC_END]
    seed = 4321
    chunks = N.song_chunks(prefix, codec, seed)
    assert len(chunks) == 1
    chunk = chunks[0]
    save("nar_ar_tokens", np.asarray(chunk.ar_tokens, dtype=np.int32))
    save("nar_noise", chunk.noise.numpy())
    t0 = time.time()
    engine = StreamingNAR(model, chunk)
    raw1 = torch.logit(torch.tensor(1.0, dtype=torch.float64)).clamp(-20, 20).item()
    v1 = engine.velocity(chunk.noise, raw1)
    save("nar_velocity_t1", v1.numpy())
    gen = torch.Generator().manual_seed(77)
    state = torch.randn(chunk.noise.shape, generator=gen)
    rawm = torch.logit(torch.tensor(0.4, dtype=torch.float64)).item()
    save("nar_state_t04", state.numpy())
    v2 = engine.velocity(state, rawm)
    save("nar_velocity_t04", v2.numpy())
    print(f"velocities {time.time()-t0:.0f}s", flush=True)
    steps = int(os.environ.get("ORACLE_ODE_STEPS", "32"))
    lat = engine.solve(steps, on_progress=lambda a, b: print(f"  ode {a}/{b} {time.time()-t0:.0f}s", flush=True))
    save(f"nar_latents_{steps}", lat.numpy())
    json.dump({"seed": seed, "raw_t1": raw1, "raw_t04": rawm, "frames": len(codec), "steps": steps},
              open(f"{OUT}/nar.json", "w"), indent=1)


def stage_vae():
    from yue2.modeling_vae import YuE2VAE
    vae = YuE2VAE.from_pretrained(VAE, decoder_only=True, device="cpu", local_files_only=True)
    gen = torch.Generator().manual_seed(99)
    frames = 300
    z = torch.randn((1, 64, frames), generator=gen)
    save("vae_latents_rand", z[0].T.contiguous().numpy())  # [frames, 64]
    full = vae.decode(z)
    save("vae_audio_rand_full", full[0].numpy())  # [2, samples]
    tiled = vae.decode_tiled(z, core_frames=128, halo_frames=16)
    save("vae_audio_rand_tiled128", tiled[0].numpy())
    print("vae halo", vae.required_halo(128), vae.required_halo(1024), tuple(full.shape),
          float((full - tiled).abs().max()), flush=True)
    for path in glob.glob(f"{OUT}/nar_latents_*.npy"):
        lat = torch.from_numpy(np.load(path))
        audio = vae.decode(lat.T[None])
        save(os.path.basename(path).replace("nar_latents", "vae_audio_nar")[:-4], audio[0].numpy())


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for stage in sys.argv[1:]:
        t = time.time()
        {"tok": stage_tok, "ar": stage_ar, "nar": stage_nar, "vae": stage_vae}[stage]()
        print(f"stage {stage} done in {time.time()-t:.0f}s", flush=True)
