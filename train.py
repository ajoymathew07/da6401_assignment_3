"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import shutil
from typing import Optional, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sacrebleu
import wandb

from model import EncoderLayer, Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]
            target : shape [batch * tgt_len]
        Returns:
            Scalar loss value.
        """
        smooth_dist = torch.full(
            (logits.size(0), self.vocab_size),
            self.smoothing / (self.vocab_size - 2),
            device=logits.device
        )
        smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_dist[:, self.pad_idx] = 0.0

        pad_mask = (target == self.pad_idx)
        smooth_dist[pad_mask] = 0.0

        log_probs = torch.log_softmax(logits, dim=-1)
        loss      = -(smooth_dist * log_probs).sum(dim=-1)
        non_pad   = (~pad_mask).sum()
        return loss.sum() / non_pad.clamp(min=1)


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance (or DataParallel wrapper).
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    pad_idx      = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for src, tgt in data_iter:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx).to(device)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx).to(device)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            _, _, vocab_size = logits.shape
            logits_flat  = logits.contiguous().view(-1, vocab_size)
            targets_flat = tgt_target.contiguous().view(-1)

            loss = loss_fn(logits_flat, targets_flat)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/lr":        optimizer.param_groups[0]["lr"],
                })

            non_pad       = (targets_flat != pad_idx).sum().item()
            total_loss   += loss.item() * non_pad   # type: ignore
            total_tokens += non_pad                  # type: ignore


    avg_loss = total_loss / max(total_tokens, 1)    # type: ignore
    prefix   = "train" if is_train else "val"
    wandb.log({f"{prefix}/epoch_loss": avg_loss, "epoch": epoch_num})
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys     = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask  = make_tgt_mask(ys, pad_idx=0).to(device)
            logits    = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys         = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break

    return ys  # [1, out_len]


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab: dict,
    device: str = "cpu",
    max_len: int = 50,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
        tgt_vocab       : Target vocabulary dict {token: idx}.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    model.eval()

    pad_idx      = tgt_vocab["<pad>"]
    sos_idx      = tgt_vocab["<sos>"]
    eos_idx      = tgt_vocab["<eos>"]
    idx_to_token = {idx: token for token, idx in tgt_vocab.items()}

    predictions: list[str] = []
    references:  list[str] = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_i, pad_idx).to(device)

                output = greedy_decode(
                    model, src_i, src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                )

                pred_tokens = [
                    idx_to_token[idx] for idx in output[0].tolist()
                    if idx not in (sos_idx, eos_idx, pad_idx)
                ]
                ref_tokens = [
                    idx_to_token[idx] for idx in tgt[i].tolist()
                    if idx not in (sos_idx, eos_idx, pad_idx)
                ]
                predictions.append(detokenize(pred_tokens))
                references.append(detokenize(ref_tokens))

    result = sacrebleu.corpus_bleu(predictions, [references])
    return float(result.score)


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.
    Handles DataParallel-wrapped models automatically.
    """
    # Unwrap DataParallel if needed
    raw_model     = cast(Transformer, model.module if isinstance(model, nn.DataParallel) else model)  # type: ignore
    encoder_layer = cast(EncoderLayer, raw_model.encoder.layers[0])

    torch.save({
        "epoch":                epoch,
        "model_state_dict":     raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "src_vocab": raw_model.src_vocab,
        "tgt_vocab": raw_model.tgt_vocab,
        "model_config": {
            "d_model":   raw_model.src_embed.embedding_dim,
            "N":         len(raw_model.encoder.layers),
            "num_heads": encoder_layer.self_attn.num_heads,
            "d_ff":      encoder_layer.ffn.linear1.out_features,
            "dropout":   encoder_layer.dropout.p,
        },
    }, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"])

def detokenize(tokens: list[str]) -> str:

    sentence = " ".join(tokens)

    for p in [".", ",", "!", "?", ":", ";"]:

        sentence = sentence.replace(f" {p}", p)

    for c in ["n't", "'s", "'re", "'ve", "'ll", "'m"]:

        sentence = sentence.replace(f" {c}", c)

    return sentence

# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler
    import argparse
    # ── Argument Parser ───────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="DA6401 Assignment 3 - Transformer NMT")

    # Model hyperparameters
    parser.add_argument("--d_model",      type=int,   default=512,   help="Model dimensionality")
    parser.add_argument("--N",            type=int,   default=6,     help="Number of encoder/decoder layers")
    parser.add_argument("--num_heads",    type=int,   default=8,     help="Number of attention heads")
    parser.add_argument("--d_ff",         type=int,   default=2048,  help="FFN inner dimensionality")
    parser.add_argument("--dropout",      type=float, default=0.1,   help="Dropout probability")

    # Training hyperparameters
    parser.add_argument("--warmup_steps", type=int,   default=4000,  help="Noam scheduler warmup steps")
    parser.add_argument("--batch_size",   type=int,   default=128,   help="Batch size (per GPU if multi-GPU)")
    parser.add_argument("--num_epochs",   type=int,   default=15,    help="Number of training epochs")
    parser.add_argument("--smoothing",    type=float, default=0.1,   help="Label smoothing factor")

    # Experiment control
    parser.add_argument("--train",        action="store_true",        help="Train from scratch")
    parser.add_argument("--checkpoint",   type=str,   default="best_checkpoint.pt",
                        help="Path to checkpoint to load/save")
    parser.add_argument("--wandb_project",type=str,   default="da6401-a3", help="W&B project name")
    parser.add_argument("--wandb_run",    type=str,   default=None,   help="W&B run name (optional)")

    args = parser.parse_args()

    # ── 1. W&B ────────────────────────────────────────────────────────
    config = {
        "d_model":      args.d_model,
        "N":            args.N,
        "num_heads":    args.num_heads,
        "d_ff":         args.d_ff,
        "dropout":      args.dropout,
        "warmup_steps": args.warmup_steps,
        "batch_size":   args.batch_size,
        "num_epochs":   args.num_epochs,
        "smoothing":    args.smoothing,
    }
    wandb.init(project="Neural Machine Translation", config=config)
    cfg    = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 2. Build model first — vocab is constructed inside __init__ ───
    base_model = Transformer(
        d_model      = cfg.d_model,
        N            = cfg.N,
        num_heads    = cfg.num_heads,
        d_ff         = cfg.d_ff,
        dropout      = cfg.dropout,
        load_weights = not args.train,
    ).to(device)

    # ── 3. Wrap with DataParallel if multiple GPUs available ──────────
    if torch.cuda.device_count() > 1:
        print(f"[Training] Using {torch.cuda.device_count()} GPUs")
        model: nn.Module = nn.DataParallel(base_model)
    else:
        model = base_model

    # ── 4. Reuse exact vocab the model built ──────────────────────────
    src_vocab = base_model.src_vocab
    tgt_vocab = base_model.tgt_vocab
    pad_idx   = src_vocab["<pad>"]

    # ── 5. Dataset — assign model's vocab, then process ───────────────
    train_ds = Multi30kDataset(split="train")
    val_ds   = Multi30kDataset(split="validation")
    test_ds  = Multi30kDataset(split="test")

    for ds in (train_ds, val_ds, test_ds):
        ds.src_vocab = src_vocab
        ds.tgt_vocab = tgt_vocab

    train_data = train_ds.process_data()
    val_data   = val_ds.process_data()
    test_data  = test_ds.process_data()

    # ── 6. DataLoaders ────────────────────────────────────────────────
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = nn.utils.rnn.pad_sequence(
            [torch.tensor(s) for s in src_batch],
            batch_first=True, padding_value=pad_idx
        )
        tgt_padded = nn.utils.rnn.pad_sequence(
            [torch.tensor(t) for t in tgt_batch],
            batch_first=True, padding_value=pad_idx
        )
        return src_padded, tgt_padded

    class PairDataset(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self):        return len(self.data)
        def __getitem__(self, i): return self.data[i]

    train_loader = DataLoader(PairDataset(train_data), batch_size=cfg.batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(PairDataset(val_data),   batch_size=cfg.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(PairDataset(test_data),  batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    # ── 7. Optimizer, scheduler, loss ─────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps
    )
    loss_fn = LabelSmoothingLoss(
        len(tgt_vocab), pad_idx, smoothing=cfg.smoothing
    )

    # ── 8. Training loop ──────────────────────────────────────────────
    if args.train:
        best_val_loss = float("inf")
        for epoch in range(cfg.num_epochs):
            train_loss = run_epoch(
                train_loader, model, loss_fn, optimizer, scheduler,
                epoch_num=epoch, is_train=True, device=device
            )
            val_loss = run_epoch(
                val_loader, model, loss_fn, None, None,
                epoch_num=epoch, is_train=False, device=device
            )
            print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, args.checkpoint)
                print(f"Checkpoint saved → {args.checkpoint}")
            if (epoch + 1) % 5 == 0:
                load_checkpoint(args.checkpoint, base_model)
                quick_bleu = evaluate_bleu(
                    base_model, test_loader, tgt_vocab,
                    device=device, max_len=50
                )
                print(f"Epoch {epoch+1:02d} | quick_bleu={quick_bleu:.2f}")
                wandb.log({"val/bleu": quick_bleu, "epoch": epoch})

        shutil.copy(args.checkpoint, "transformer_weights.pt")
        print("Copied → transformer_weights.pt")
        load_checkpoint(args.checkpoint, base_model)

    # ── 10. Evaluate BLEU on test set using best weights ──────────────
    bleu = evaluate_bleu(base_model, test_loader, tgt_vocab, device=device)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()