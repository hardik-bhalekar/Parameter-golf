import collections,copy,glob,io,lzma,math,os
from pathlib import Path
import random,re,subprocess,sys,time,uuid,numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor,nn

# Attempt to import brotli, fallback to lzma if not found
try:
    import brotli
    HAS_BROTLI = True
except ImportError:
    HAS_BROTLI = False

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
except ImportError:
    # CPU fallback for local testing
    def flash_attn_3_func(q, k, v, causal=True):
        # q, k, v: [B, T, H, D]
        q = q.transpose(1, 2) # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return y.transpose(1, 2)

class Hyperparameters:
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = os.environ.get('DATA_DIR', str(repo_root / 'data'))
    seed = int(os.environ.get('SEED', 1337))
    run_id = os.environ.get('RUN_ID', str(uuid.uuid4()))
    iterations = int(os.environ.get('ITERATIONS', 4500)) # SOTA uses ~4550
    warmdown_frac = float(os.environ.get('WARMDOWN_FRAC', 0.75)) # Increased from 0.72
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', 786432))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    train_log_every = int(os.environ.get('TRAIN_LOG_EVERY', 200))
    max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))
    val_batch_tokens = int(os.environ.get('VAL_BATCH_TOKENS', 524288))
    eval_seq_len = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    val_loss_every = int(os.environ.get('VAL_LOSS_EVERY', 2000))
    sliding_window_enabled = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))
    vocab_size = int(os.environ.get('VOCAB_SIZE', 8192))
    num_layers = int(os.environ.get('NUM_LAYERS', 11))
    xsa_last_n = int(os.environ.get('XSA_LAST_N', 11))
    model_dim = int(os.environ.get('MODEL_DIM', 512))
    embedding_dim = int(os.environ.get('EMBEDDING_DIM', 512))
    num_kv_heads = int(os.environ.get('NUM_KV_HEADS', 4))
    num_heads = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult = float(os.environ.get('MLP_MULT', 4.0))
    skip_gates_enabled = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    tie_embeddings = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    logit_softcap = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    rope_base = float(os.environ.get('ROPE_BASE', 10000.0))
    rope_dims = int(os.environ.get('ROPE_DIMS', 16))
    rope_train_seq_len = int(os.environ.get('ROPE_TRAIN_SEQ_LEN', 2048))
    ln_scale = bool(int(os.environ.get('LN_SCALE', '1')))
    qk_gain_init = float(os.environ.get('QK_GAIN_INIT', 5.5)) # Increased from 5.0
    num_loops = int(os.environ.get('NUM_LOOPS', 3)) # Increased from 2
    loop_start = int(os.environ.get('LOOP_START', 3))
    loop_end = int(os.environ.get('LOOP_END', 6)) # Increased from 5
    enable_looping_at = float(os.environ.get('ENABLE_LOOPING_AT', 0.40)) # Slightly later to save time
    parallel_residual_start = int(os.environ.get('PARALLEL_RESIDUAL_START', 0)) # Parallel residuals from layer 0
    min_lr = float(os.environ.get('MIN_LR', 0.0))
    embed_lr = float(os.environ.get('EMBED_LR', 0.6))
    head_lr = float(os.environ.get('HEAD_LR', 0.008))
    tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    matrix_lr = float(os.environ.get('MATRIX_LR', 0.024)) # Slightly increased
    scalar_lr = float(os.environ.get('SCALAR_LR', 0.02))
    muon_momentum = float(os.environ.get('MUON_MOMENTUM', 0.99))
    muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', 6)) # Increased from 5
    muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.92))
    muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
    muon_row_normalize = bool(int(os.environ.get('MUON_ROW_NORMALIZE', '1')))
    beta1 = float(os.environ.get('BETA1', 0.9))
    beta2 = float(os.environ.get('BETA2', 0.95))
    adam_eps = float(os.environ.get('ADAM_EPS', 1e-8))
    grad_clip_norm = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
    eval_stride = int(os.environ.get('EVAL_STRIDE', 64))
    muon_beta2 = float(os.environ.get('MUON_BETA2', 0.95))
    adam_wd = float(os.environ.get('ADAM_WD', 0.02))
    muon_wd = float(os.environ.get('MUON_WD', 0.095))
    embed_wd = float(os.environ.get('EMBED_WD', 0.085))
    ema_decay = float(os.environ.get('EMA_DECAY', 0.9965))
    ttt_enabled = bool(int(os.environ.get('TTT_ENABLED', '1')))
    ttt_lr = float(os.environ.get('TTT_LR', 0.002)) # Reduced for Adam
    ttt_epochs = int(os.environ.get('TTT_EPOCHS', 3))
    ttt_momentum = float(os.environ.get('TTT_MOMENTUM', 0.9))
    ttt_chunk_tokens = int(os.environ.get('TTT_CHUNK_TOKENS', 32768))
    etlb_enabled = bool(int(os.environ.get('ETLB_ENABLED', '0')))
    etlb_lr = float(os.environ.get('ETLB_LR', 0.05))
    etlb_steps = int(os.environ.get('ETLB_STEPS', 5))
    etlb_clip = float(os.environ.get('ETLB_CLIP', 3.0))
    compressor = os.environ.get('COMPRESSOR', 'brotli' if HAS_BROTLI else 'lzma')
    gptq_calibration_batches = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 64))
    gptq_reserve_seconds = float(os.environ.get('GPTQ_RESERVE_SECONDS', 15.0))
    matrix_bits = int(os.environ.get('MATRIX_BITS', 6))
    embed_bits = int(os.environ.get('EMBED_BITS', 8))
    matrix_clip_sigmas = float(os.environ.get('MATRIX_CLIP_SIGMAS', 12.85))
    embed_clip_sigmas = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))
    
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size
    datasets_dir = os.path.join(data_dir, 'datasets', f"fineweb10B_sp{vocab_size}")
    train_files = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path = os.path.join(data_dir, 'tokenizers', f"fineweb_{vocab_size}_bpe.model")
    logfile = f"logs/{run_id}.txt"
    model_path = 'final_model.pt'
    quantized_model_path = f"final_model.int6.{'br' if compressor == 'brotli' else 'lzma'}"

_logger_hparams = None
def set_logging_hparams(h):
    global _logger_hparams
    _logger_hparams = h
def log(msg, console=True):
    if _logger_hparams is None: print(msg); return
    if _logger_hparams.is_main_process:
        if console: print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, 'a', encoding='utf-8') as f: print(msg, file=f)

class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size: raise ValueError(f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}")
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_sentencepiece_luts(self.sp, h.vocab_size, device)

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert sp.piece_to_id(' ') != sp.unk_id() or sp.piece_to_id('▁') != sp.unk_id()
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id): base_bytes_np[token_id] = 1; continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith(' ') or piece.startswith('▁'):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode('utf-8'))
    return torch.tensor(base_bytes_np, dtype=torch.int16, device=device), torch.tensor(has_leading_space_np, dtype=torch.bool, device=device), torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device)

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable <= 0: raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[:usable+1]

def load_data_shard(file):
    header_bytes = 256 * np.dtype('<i4').itemsize
    token_bytes = np.dtype('<u2').itemsize
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1: raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size: raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype='<u2', count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank + h.seed))
        self.num_tokens = []
        for f in self.files:
            header = np.fromfile(f, dtype='<i4', count=256)
            self.num_tokens.append(int(header[2]))
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)): self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)): self._reset_shard(si)
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
                total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = np.memmap(self.files[si], mode='r', dtype='<u2', offset=256*4, shape=(self.num_tokens[si],))
            window = torch.as_tensor(np.array(mm[start_ind:start_ind+self.seq_len+1], dtype=np.int64))
            x[bi] = window[:-1]; y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        w = self.weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

class Rotary(nn.Module):
    def __init__(self, dim, base=1e4, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim; self.base = base; self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1. / base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._seq_len_cached = 0; self._cos_cached = None; self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (rd / (rd - 2))
                inv_freq = 1. / new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd)
            else: inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]; self._sin_cached = freqs.sin()[None, :, None, :]; self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * -sin + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
        super().__init__()
        self.num_heads = num_heads; self.num_kv_heads = num_kv_heads; self.head_dim = dim // num_heads
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False); self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa: y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__(); hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False); self.proj._zero_init = True
    def forward(self, x): return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init, train_seq_len, layer_idx=0, ln_scale=False):
        super().__init__()
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1. / math.sqrt(layer_idx + 1) if ln_scale else 1.
        self.parallel = False

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            return x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
        else:
            x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
            return x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)

class GPT(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.tie_embeddings = h.tie_embeddings; self.tied_embed_init_std = h.tied_embed_init_std; self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else: self.embed_proj = self.head_proj = None
        self.num_layers = h.num_layers
        self.blocks = nn.ModuleList([Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult, h.rope_base, h.qk_gain_init, h.train_seq_len, layer_idx=i, ln_scale=h.ln_scale) for i in range(h.num_layers)])
        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)
        if self.lm_head is not None: self.lm_head._zero_init = True
        for i in range(max(0, h.num_layers - h.xsa_last_n), h.num_layers): self.blocks[i].attn.use_xsa = True
        for i in range(h.parallel_residual_start, h.num_layers): self.blocks[i].parallel = True
        self.looping_active = False
        loop_seg = list(range(h.loop_start, h.loop_end + 1))
        all_indices = list(range(h.loop_start))
        for _ in range(h.num_loops + 1): all_indices.extend(loop_seg)
        all_indices.extend(range(h.loop_end + 1, h.num_layers))
        mid = len(all_indices) // 2
        self.encoder_indices = all_indices[:mid]; self.decoder_indices = all_indices[mid:]
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)) if h.skip_gates_enabled else None
        self._init_weights()

    def _init_weights(self):
        if self.tie_embeddings: nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if getattr(m, '_zero_init', False): nn.init.zeros_(m.weight)
                elif m.weight.ndim == 2 and m.weight.shape[0] >= 64 and m.weight.shape[1] >= 64: nn.init.orthogonal_(m.weight, gain=1.0)

    def forward_logits(self, input_ids):
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        if self.embed_proj is not None: x = self.embed_proj(x)
        x0 = x; skips = []
        enc_iter = self.encoder_indices if self.looping_active else range(self.num_layers // 2)
        dec_iter = self.decoder_indices if self.looping_active else range(self.num_layers // 2, self.num_layers)
        for i in enc_iter: x = self.blocks[i](x, x0); skips.append(x)
        for idx, i in enumerate(dec_iter):
            if idx < self.num_skip_weights and skips:
                scaled_skip = self.skip_weights[idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if self.skip_gates is not None:
                    g = torch.sigmoid(self.skip_gates[idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(scaled_skip, x, g)
                else: x = x + scaled_skip
            x = self.blocks[i](x, x0)
        x = self.final_norm(x)
        if self.head_proj is not None: x = self.head_proj(x)
        logits = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target_ids.reshape(-1), reduction='mean')

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.775, 2.0315); X = G.bfloat16(); X /= X.norm() + eps
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A; X = a * X + B @ X
    return X.T if G.size(0) > G.size(1) else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0, row_normalize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay, row_normalize=row_normalize))
    @torch.no_grad()
    def step(self, closure=None):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        for group in self.param_groups:
            params = group['params']
            if not params: continue
            lr, momentum, steps, nesterov = group['lr'], group['momentum'], group['backend_steps'], group['nesterov']
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad; state = self.state[p]
                    if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']; buf.mul_(momentum).add_(g)
                    if nesterov: g = g.add(buf, alpha=momentum)
                    if group.get('row_normalize', False): g = g / g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=steps); g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr:curr+p.numel()] = g.reshape(-1)
                curr += p.numel()
            if dist.is_initialized(): dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get('weight_decay', 0.0); curr = 0
            for p in params:
                if wd > 0.0: p.data.mul_(1.0 - lr * wd)
                p.add_(updates_flat[curr:curr+p.numel()].view_as(p).to(p.dtype), alpha=-lr); curr += p.numel()

CONTROL_TENSOR_NAME_PATTERNS = ('attn_scale', 'mlp_scale', 'resid_mix', 'q_gain', 'skip_weight', 'skip_gate')
class Optimizers:
    def __init__(self, h, model):
        named_params = list(model.blocks.named_parameters())
        matrix_params = [p for n, p in named_params if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
        scalar_params = [p for n, p in named_params if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
        scalar_params.extend([model.skip_weights])
        if model.skip_gates is not None: scalar_params.append(model.skip_gates)
        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        self.optimizer_tok = torch.optim.AdamW([{'params': [model.tok_emb.weight], 'lr': token_lr, 'base_lr': token_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True)
        self.optimizer_muon = Muon(matrix_params, lr=h.matrix_lr, momentum=h.muon_momentum, backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd, row_normalize=h.muon_row_normalize)
        for g in self.optimizer_muon.param_groups: g['base_lr'] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW([{'params': scalar_params, 'lr': h.scalar_lr, 'base_lr': h.scalar_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
        if model.lm_head is not None: self.optimizers.append(torch.optim.Adam([{'params': [model.lm_head.weight], 'lr': h.head_lr, 'base_lr': h.head_lr}], betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True))

    def zero_grad_all(self):
        for opt in self.optimizers: opt.zero_grad(set_to_none=True)
    def step(self):
        for opt in self.optimizers: opt.step()

def collect_hessians(model, loader, h, device, n=64):
    hessians = {}; hooks = []
    def hook(name):
        def fn(m, i, o):
            x = i[0].detach().float(); x = x.reshape(-1, x.shape[-1])
            if name not in hessians: hessians[name] = torch.zeros(x.shape[1], x.shape[1], device=device)
            hessians[name].addmm_(x.T, x)
        return fn
    for n_mod, m in model.named_modules():
        if isinstance(m, CastedLinear) and m.weight.numel() > 65536: hooks.append(m.register_forward_hook(hook(n_mod + '.weight')))
    model.eval()
    with torch.no_grad():
        for _ in range(n): x, _ = loader.next_batch(h.train_batch_tokens, h.grad_accum_steps); model.forward_logits(x)
    for h_ in hooks: h_.remove()
    return {k: v.cpu() / n for k, v in hessians.items()}

def gptq_quantize(w, H, sigmas=3.0, range_val=63, block=128):
    W = w.float().clone(); cols = W.shape[1]; H = H.float().clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1; H.diagonal().add_(0.01 * H.diag().mean())
    perm = torch.argsort(H.diag(), descending=True); inv = torch.argsort(perm); W = W[:, perm]; H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H)); Hinv = torch.linalg.cholesky(Hinv, upper=True)
    s = (sigmas * W.std(dim=1) / range_val).clamp_min(1e-10).to(torch.float16); Q = torch.zeros(W.shape, dtype=torch.int8)
    for i in range(0, cols, block):
        i2 = min(i + block, cols); W_b = W[:, i:i2].clone(); Hinv_b = Hinv[i:i2, i:i2]; Err = torch.zeros(W.shape[0], i2 - i)
        for j in range(i2 - i):
            w_col = W_b[:, j]; q_col = torch.clamp(torch.round(w_col / s.float()), -range_val, range_val)
            Q[:, i+j] = q_col.to(torch.int8); err = (w_col - q_col * s.float()) / Hinv_b[j, j]; Err[:, j] = err
            W_b[:, j:] -= err.unsqueeze(1) * Hinv_b[j, j:].unsqueeze(0)
        if i2 < cols: W[:, i2:] -= Err @ Hinv[i:i2, i2:]
    return Q[:, inv], s

def serialize(h, model, code):
    torch.save(model.state_dict(), h.model_path); device = torch.device('cuda', h.local_rank)
    loader = ShuffledSequenceLoader(h, device); hessians = collect_hessians(model, loader, h, device, n=h.gptq_calibration_batches)
    res = {}; meta = {}
    for n, p in model.state_dict().items():
        t = p.detach().cpu(); 
        if t.is_floating_point() and t.numel() > 65536:
            q, s = gptq_quantize(t, hessians[n], sigmas=h.embed_clip_sigmas if 'tok_emb' in n else h.matrix_clip_sigmas, range_val=2**(h.matrix_bits-1)-1)
            res[n+'.q'] = q; res[n+'.scale'] = s; meta[n] = f"gptq {h.matrix_bits}"
        else: res[n] = t.to(torch.float16) if t.is_floating_point() else t; meta[n] = 'float16'
    buf = io.BytesIO(); torch.save({'w': res, 'm': meta}, buf)
    blob = (brotli.compress(buf.getvalue(), quality=11) if h.compressor == 'brotli' else lzma.compress(buf.getvalue(), preset=6))
    with open(h.quantized_model_path, 'wb') as f: f.write(blob)
    log(f"Total size: {len(blob) + len(code.encode('utf-8'))} bytes")

def deserialize(h, device):
    m = GPT(h).to(device).bfloat16()
    with open(h.quantized_model_path, 'rb') as f: blob = f.read()
    raw = (brotli.decompress(blob) if h.compressor == 'brotli' else lzma.decompress(blob))
    data = torch.load(io.BytesIO(raw), map_location='cpu'); sd = {}
    for n, info in data['m'].items():
        if 'float16' in info: sd[n] = data['w'][n].to(torch.bfloat16)
        else: q, s = data['w'][n+'.q'], data['w'][n+'.scale']; sd[n] = (q.float() * s.float().view(-1, 1)).to(torch.bfloat16)
    m.load_state_dict(sd); return m

def eval_val(h, device, val_data, model, sliding=False, ttt=False):
    model.eval(); seq_len = h.eval_seq_len; stride = h.eval_stride if sliding or ttt else seq_len
    tokens = val_data.val_tokens; total = tokens.numel() - 1; window_starts = range(0, total - (seq_len - stride), stride)
    my_windows = list(window_starts)[h.rank::h.world_size]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
    
    if ttt:
        opt = torch.optim.Adam(model.parameters(), lr=h.ttt_lr, betas=(0.9, 0.95))
        chunk_size = h.ttt_chunk_tokens
    
    with torch.inference_mode(not ttt):
        for i, ws in enumerate(my_windows):
            if ttt and i > 0 and (ws // chunk_size) > ((my_windows[i-1]) // chunk_size):
                model.train()
                for _ in range(h.ttt_epochs):
                    # Simple TTT: train on the chunk we just evaluated
                    start = max(0, ws - chunk_size); end = ws
                    chunk = tokens[start:end+1].to(device)
                    x, y = chunk[:-1].reshape(-1, seq_len), chunk[1:].reshape(-1, seq_len)
                    opt.zero_grad(); loss = model(x, y); loss.backward(); opt.step()
                model.eval()
            
            we = min(ws + seq_len, total); chunk = tokens[ws:we+1].to(device)
            x, y = chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0)
            logits = model.forward_logits(x)
            s = 0 if ws == 0 else seq_len - stride
            nll = F.cross_entropy(logits[:, s:].reshape(-1, logits.size(-1)).float(), y[:, s:].reshape(-1), reduction='none')
            loss_sum += nll.sum(); token_cnt += nll.numel()
            tgt, prev = y[0, s:], x[0, s:]
            tb = val_data.base_bytes_lut[tgt].to(torch.float64)
            tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
            byte_cnt += tb.sum()
            
    if dist.is_initialized():
        dist.all_reduce(loss_sum); dist.all_reduce(token_cnt); dist.all_reduce(byte_cnt)
    loss = (loss_sum / token_cnt).item()
    bpb = loss / math.log(2) * (token_cnt / byte_cnt).item()
    return loss, bpb

def train_model(h, device, val_data):
    base_model = GPT(h).to(device).bfloat16()
    model = DDP(torch.compile(base_model), device_ids=[h.local_rank]) if h.distributed else torch.compile(base_model)
    opts = Optimizers(h, base_model); loader = ShuffledSequenceLoader(h, device)
    t0 = time.perf_counter(); train_ms = 0; step = 0
    ema = {n: p.detach().float().clone() for n, p in base_model.state_dict().items()}
    
    while step < h.iterations:
        elapsed = train_ms + (time.perf_counter() - t0) * 1000
        if elapsed >= (h.max_wallclock_seconds - h.gptq_reserve_seconds) * 1000: break
        frac = elapsed / ((h.max_wallclock_seconds - h.gptq_reserve_seconds) * 1000)
        lr_scale = max((1.0 - frac) / h.warmdown_frac, h.min_lr) if frac > (1.0 - h.warmdown_frac) else 1.0
        if h.num_loops > 0 and not base_model.looping_active and frac >= h.enable_looping_at:
            base_model.looping_active = True; log(f"Looping enabled at step {step}")
        
        opts.zero_grad_all()
        for _ in range(h.grad_accum_steps):
            x, y = loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast('cuda', dtype=torch.bfloat16): loss = model(x, y); (loss / h.grad_accum_steps).backward()
        
        for opt in opts.optimizers:
            for g in opt.param_groups: g['lr'] = g['base_lr'] * lr_scale
        torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm); opts.step()
        
        with torch.no_grad():
            for n, p in base_model.state_dict().items(): ema[n].mul_(h.ema_decay).add_(p.detach().float(), alpha=1-h.ema_decay)
        
        step += 1
        if step % h.train_log_every == 0: log(f"Step {step}, Loss: {loss.item():.4f}")
    
    base_model.load_state_dict({n: p.to(dtype=torch.bfloat16) for n, p in ema.items()}); return base_model

def main():
    h = Hyperparameters(); local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank); torch.cuda.set_device(device)
    if h.distributed: dist.init_process_group('nccl')
    set_logging_hparams(h); val_data = ValidationData(h, device)
    model = train_model(h, device, val_data)
    serialize(h, model, Path(__file__).read_text(encoding='utf-8'))
    if h.is_main_process:
        m_eval = deserialize(h, device); m_eval.looping_active = True
        l, b = eval_val(h, device, val_data, m_eval, sliding=True, ttt=True)
        log(f"Final BPB: {b:.5f}")

if __name__ == '__main__': main()
