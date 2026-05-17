"""
date:2025/5/18
id: 23375158
description: Attention Is All You Need — Transformer 复现与位置编码消融实验
trainSet: Multi30k (英语→德语翻译)
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('HF_HOME', 'e:/Project/ML/.cache/huggingface')
os.environ.setdefault('HF_HUB_CACHE', 'e:/Project/ML/.cache/huggingface/hub')
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import sacrebleu
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

os.makedirs('asserts', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# 1. Tokenization & Vocabulary

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
SPECIALS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


def build_vocab(sentences, max_size=30000):
    """Build word-level vocabulary from list of sentences."""
    counter = Counter()
    for sent in sentences:
        counter.update(sent.strip().split())
    vocab = {word: idx + len(SPECIALS) for idx, (word, _) in
             enumerate(counter.most_common(max_size - len(SPECIALS)))}
    for i, token in enumerate(SPECIALS):
        vocab[token] = i
    return vocab


def tokenize(sent, vocab, max_len=128):
    """Convert sentence to token indices."""
    tokens = [vocab.get(w, UNK_IDX) for w in sent.strip().split()]
    tokens = [SOS_IDX] + tokens[:max_len - 2] + [EOS_IDX]
    return tokens


def load_multi30k():
    """Load Multi30k dataset and build vocabularies."""
    print("Loading Multi30k dataset...")
    dataset = load_dataset("bentrevett/multi30k")

    train_src = [item['en'] for item in dataset['train']]
    train_tgt = [item['de'] for item in dataset['train']]
    val_src = [item['en'] for item in dataset['validation']]
    val_tgt = [item['de'] for item in dataset['validation']]
    test_src = [item['en'] for item in dataset['test']]
    test_tgt = [item['de'] for item in dataset['test']]

    src_vocab = build_vocab(train_src, max_size=10000)
    tgt_vocab = build_vocab(train_tgt, max_size=10000)

    print(f"Source vocab size: {len(src_vocab)}, Target vocab size: {len(tgt_vocab)}")
    print(f"Train: {len(train_src)}, Val: {len(val_src)}, Test: {len(test_src)}")
    return (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt), src_vocab, tgt_vocab


class TranslationDataset(Dataset):
    def __init__(self, src_sents, tgt_sents, src_vocab, tgt_vocab, max_len=128):
        self.src_data = [torch.tensor(tokenize(s, src_vocab, max_len), dtype=torch.long)
                         for s in src_sents]
        self.tgt_data = [torch.tensor(tokenize(s, tgt_vocab, max_len), dtype=torch.long)
                         for s in tgt_sents]

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded



# 2. Transformer Model Components


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Concatenate heads and project
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.W_o(out)
        return out


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model=256, d_ff=512, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SinusoidalPositionalEncoding(nn.Module):
    """Original sinusoidal positional encoding from the paper."""
    def __init__(self, d_model=256, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned absolute positional embedding (ablation variant)."""
    def __init__(self, d_model=256, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


class NoPositionalEncoding(nn.Module):
    """No positional encoding (ablation variant)."""
    def __init__(self, d_model=256, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention with residual + layer norm (Post-LN)
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN with residual + layer norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_heads=8, d_ff=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Cross-attention with encoder output
        cross_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=512,
                 num_layers=3, dropout=0.1, pos_encoding='sinusoidal', max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.d_model = d_model

        if pos_encoding == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == 'learned':
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == 'none':
            self.pos_encoding = NoPositionalEncoding(d_model, max_len, dropout)
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, d_ff=512,
                 num_layers=3, dropout=0.1, pos_encoding='sinusoidal', max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.d_model = d_model

        if pos_encoding == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == 'learned':
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_len, dropout)
        elif pos_encoding == 'none':
            self.pos_encoding = NoPositionalEncoding(d_model, max_len, dropout)
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.output_proj(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8,
                 d_ff=512, num_layers=3, dropout=0.1, pos_encoding='sinusoidal'):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff,
                               num_layers, dropout, pos_encoding)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff,
                               num_layers, dropout, pos_encoding)
        # Weight tying: share embedding between decoder input and output projection
        self.decoder.output_proj.weight = self.decoder.embedding.weight

    def forward(self, src, tgt):
        src_mask = self._make_pad_mask(src)
        tgt_mask = self._make_tgt_mask(tgt)
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return out

    def _make_pad_mask(self, seq):
        return (seq != PAD_IDX).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)

    def _make_tgt_mask(self, tgt):
        """Causal mask + padding mask."""
        batch_size, tgt_len = tgt.shape
        # Causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        # Padding mask
        pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        # Combine: (B, 1, T, T) & causal
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        return causal_mask & pad_mask  # (B, 1, T, T)



# 3. Optimizer with Warmup (from paper)

class NoamOptimizer:
    """Adam optimizer with warmup schedule from the paper."""
    def __init__(self, model, d_model, warmup_steps=4000, lr_factor=1.0):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr_factor = lr_factor
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.lr_factor * (self.d_model ** (-0.5) *
                                min(self._step ** (-0.5), self._step * self.warmup_steps ** (-1.5)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict(), 'step': self._step}

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state['optimizer'])
        self._step = state['step']



# 4. Training & Evaluation

class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross entropy loss."""
    def __init__(self, smoothing=0.1, vocab_size=10000, pad_idx=PAD_IDX):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

    def forward(self, pred, target):
        # pred: (B*T, vocab), target: (B*T)
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.vocab_size - 1)
        true_dist = torch.full_like(pred, smooth_val)
        true_dist.scatter_(1, target.unsqueeze(1), confidence)
        # Mask padding
        mask = (target != self.pad_idx).float().unsqueeze(1)
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(true_dist * log_probs).sum(dim=-1)
        loss = (loss * mask.squeeze(1)).sum() / mask.sum()
        return loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc='Training', leave=False):
        src, tgt = src.to(device), tgt.to(device)
        # Teacher forcing: decoder input = tgt[:-1], target = tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        pred = model(src, tgt_input)
        loss = criterion(pred.reshape(-1, pred.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc='Validating', leave=False):
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        pred = model(src, tgt_input)
        loss = criterion(pred.reshape(-1, pred.size(-1)), tgt_output.reshape(-1))
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def translate(model, src_sent, src_vocab, tgt_vocab, device, max_len=128):
    """Greedy decoding for a single sentence."""
    model.eval()
    idx_to_tgt = {v: k for k, v in tgt_vocab.items()}

    src_tokens = tokenize(src_sent, src_vocab, max_len)
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device)

    enc_out = model.encoder(src_tensor, model._make_pad_mask(src_tensor))

    # Greedy decoding
    tgt_tokens = [SOS_IDX]
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
        tgt_mask = model._make_tgt_mask(tgt_tensor)
        src_mask = model._make_pad_mask(src_tensor)
        out = model.decoder(tgt_tensor, enc_out, src_mask, tgt_mask)
        next_token = out[:, -1, :].argmax(dim=-1).item()
        tgt_tokens.append(next_token)
        if next_token == EOS_IDX:
            break

    return ' '.join([idx_to_tgt.get(t, UNK_TOKEN) for t in tgt_tokens[1:-1]])


@torch.no_grad()
def compute_bleu(model, src_sents, tgt_sents, src_vocab, tgt_vocab, device, max_len=128):
    """Compute BLEU score on a dataset."""
    model.eval()
    predictions = []
    references = []
    for src, tgt in zip(src_sents, tgt_sents):
        pred = translate(model, src, src_vocab, tgt_vocab, device, max_len)
        predictions.append(pred)
        references.append(tgt.strip())
    # SacreBLEU expects list of references (list of lists)
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# 5. Main Experiment (data passed in to avoid redundant loading)

def run_experiment(data, pos_encoding='sinusoidal', d_model=256, num_heads=8, d_ff=512,
                   num_layers=3, dropout=0.1, epochs=20, batch_size=64,
                   warmup_steps=2000, label_smoothing=0.1):
    """Train a Transformer model and return results."""
    print(f"\n{'='*60}")
    print(f"Experiment: Positional Encoding = {pos_encoding}")
    print(f"{'='*60}")

    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt), src_vocab, tgt_vocab = data

    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab)
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    # Model
    model = Transformer(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=d_model, num_heads=num_heads, d_ff=d_ff, num_layers=num_layers,
        dropout=dropout, pos_encoding=pos_encoding
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    # Loss & Optimizer
    criterion = LabelSmoothingLoss(label_smoothing, len(tgt_vocab), PAD_IDX)
    optimizer = NoamOptimizer(model, d_model, warmup_steps)

    # Training
    best_val_loss = float('inf')
    best_state = None
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    # Load best model
    model.load_state_dict(best_state)

    # Evaluate BLEU on validation and test subsets
    val_subset_size = min(200, len(val_src))
    val_bleu = compute_bleu(model, val_src[:val_subset_size], val_tgt[:val_subset_size],
                            src_vocab, tgt_vocab, device)

    test_subset_size = min(200, len(test_src))
    test_bleu = compute_bleu(model, test_src[:test_subset_size], test_tgt[:test_subset_size],
                             src_vocab, tgt_vocab, device)

    print(f"Best Val Loss: {best_val_loss:.4f} | Val BLEU: {val_bleu:.1f} | Test BLEU: {test_bleu:.1f}")

    return {
        'pos_encoding': pos_encoding,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'val_bleu': val_bleu,
        'test_bleu': test_bleu,
        'num_params': count_parameters(model),
        'best_state': best_state,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
    }



# 6. Ablation Study Runner

def run_all_ablations(data):
    """Run all three positional encoding variants and compare."""
    configs = ['sinusoidal', 'learned', 'none']
    results = {}

    for pe_type in configs:
        results[pe_type] = run_experiment(data, pos_encoding=pe_type, epochs=15)
        # Save intermediate results
        torch.save(results, 'asserts/ablation_results.pt')

    return results


def run_quick_single(data):
    """Run a quick single-model training (sinusoidal only) for testing."""
    return run_experiment(data, pos_encoding='sinusoidal', epochs=3)



# 7. Plotting

def plot_results(results):
    """Generate comparison plots for the ablation study."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss curves
    ax = axes[0]
    colors = {'sinusoidal': '#2196F3', 'learned': '#FF9800', 'none': '#F44336'}
    labels = {'sinusoidal': 'Sinusoidal (Original)', 'learned': 'Learned Embedding', 'none': 'No Positional Encoding'}

    for pe_type, result in results.items():
        ax.plot(result['train_losses'], color=colors[pe_type], linestyle='-',
                label=f"{labels[pe_type]} (train)", alpha=0.7)
        ax.plot(result['val_losses'], color=colors[pe_type], linestyle='--',
                label=f"{labels[pe_type]} (val)", linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # BLEU comparison
    ax = axes[1]
    pe_types = list(results.keys())
    x = np.arange(len(pe_types))
    val_bleus = [results[p]['val_bleu'] for p in pe_types]
    test_bleus = [results[p]['test_bleu'] for p in pe_types]
    width = 0.35
    bars1 = ax.bar(x - width / 2, val_bleus, width, label='Validation BLEU', color='#2196F3')
    bars2 = ax.bar(x + width / 2, test_bleus, width, label='Test BLEU', color='#FF9800')
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p] for p in pe_types], fontsize=8)
    ax.set_ylabel('BLEU Score')
    ax.set_title('BLEU Score Comparison')
    ax.legend(fontsize=8)
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', fontsize=8)

    # Final validation loss comparison
    ax = axes[2]
    best_losses = [results[p]['best_val_loss'] for p in pe_types]
    bars = ax.bar(x, best_losses, color=[colors[p] for p in pe_types])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[p] for p in pe_types], fontsize=8)
    ax.set_ylabel('Best Validation Loss')
    ax.set_title('Best Validation Loss')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('asserts/ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved to asserts/ablation_comparison.png")

    # Also create individual training curve plots for the paper
    for pe_type in pe_types:
        plt.figure(figsize=(8, 4))
        result = results[pe_type]
        plt.plot(result['train_losses'], label='Train Loss', color='#2196F3')
        plt.plot(result['val_losses'], label='Validation Loss', color='#FF9800')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Curves — {labels[pe_type]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'asserts/{pe_type}_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION STUDY SUMMARY")
    print("=" * 70)
    print(f"{'Positional Encoding':<25} {'Best Val Loss':<16} {'Val BLEU':<12} {'Test BLEU':<12}")
    print("-" * 70)
    for pe_type in pe_types:
        r = results[pe_type]
        print(f"{labels[pe_type]:<25} {r['best_val_loss']:<16.4f} {r['val_bleu']:<12.1f} {r['test_bleu']:<12.1f}")



# 8. Example Translations

def print_example_translations(results, test_src, test_tgt):
    """Print example translations from each model variant."""
    print("\n" + "=" * 70)
    print("EXAMPLE TRANSLATIONS")
    print("=" * 70)

    for i in range(min(5, len(test_src))):
        print(f"\n--- Example {i+1} ---")
        print(f"Source (EN): {test_src[i]}")
        print(f"Target (DE): {test_tgt[i]}")
        for pe_type, result in results.items():
            if 'best_state' in result:
                # Build a temporary model to translate
                model = Transformer(
                    src_vocab_size=len(result['src_vocab']),
                    tgt_vocab_size=len(result['tgt_vocab']),
                    pos_encoding=pe_type
                ).to(device)
                model.load_state_dict(result['best_state'])
                pred = translate(model, test_src[i], result['src_vocab'], result['tgt_vocab'], device)
                print(f"  [{pe_type:12s}]: {pred}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['quick', 'full'], default='full',
                        help='quick: 3-epoch test run; full: 15-epoch ablation study')
    args = parser.parse_args()

    print("=" * 60)
    print("HW4: Attention Is All You Need")
    print("Ablation Study: Positional Encoding")
    print("=" * 60)

    # Load data once
    data = load_multi30k()

    if args.mode == 'quick':
        print("\n[Quick test mode] Running 3 epochs with sinusoidal encoding only...")
        result = run_quick_single(data)
        results = {'sinusoidal': result}
        torch.save(results, 'asserts/quick_results.pt')
    else:
        results = run_all_ablations(data)

    # Generate plots
    plot_results(results)

    # Print example translations
    (_, _), (_, _), (test_src, test_tgt), _, _ = data
    print_example_translations(results, test_src, test_tgt)

