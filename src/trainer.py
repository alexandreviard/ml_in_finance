import math, pandas as pd, torch
from pathlib import Path
from datetime import datetime
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class BinaryTrainer:
    """ Class Trainer for binary classification of stock prediction movements.

    Parameters
    ----------
    model : torch.nn.Module
        The model (e.g. `MG_Daily`). Must return logits of shape (batch, 1) or (batch,).
    X_train, y_train, X_val, y_val : torch.Tensor
        Input features and labels. Labels are expected to be {0., 1.} floats.
    lr : float, default 3e-4
        AdamW learning-rate.
    batch_size : int, default 64
        Mini-batch size.
    num_epochs : int, default 5
        Number of training epochs.
    gamma : float, default 0.0
        Extra regularisation penalty multiplier (expects model.block*.mha._get_penalty()).
    name : str, default "model"
        A friendly name used only for logging / checkpoint naming.
    ckpt_dir : str, default "checkpoints"
        Directory where checkpoints are stored.
    ckpt_name : str | None, optional
        Filename for checkpoints. If *None*, a name derived from *name* is used.
    val_max_samples : int, default 4000
        Optional subsampling limit for the validation set.
    save_every_epoch : int, default 1
        Save a checkpoint every *n* epochs.
    seed : int, default 42
        Random seed used only for the validation subsampling.
    device : torch.device | None, optional
        If *None*, derives the device from the model's first parameter.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        lr: float = 3e-4,
        batch_size: int = 64,
        num_epochs: int = 5,
        gamma: float = 0.0,
        name: str = "model",
        ckpt_dir: str = "checkpoints",
        ckpt_name: Optional[str] = None,
        save_every_epoch: int = 1,
        seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:

        self.name = name
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

        nb_pos = y_train.sum().item()
        nb_neg = len(y_train) - nb_pos
        self.pos_weight = torch.tensor([nb_neg / nb_pos], dtype=torch.float32, device=self.device)

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_every_epoch = save_every_epoch

        g = torch.Generator().manual_seed(seed)
        self.train_loader = DataLoader(
            TensorDataset(X_train.float(), y_train.float()),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
        )
        self.val_loader = DataLoader(
            TensorDataset(X_val.float(), y_val.float()),
            batch_size=batch_size,
            shuffle=False,
        )

        self.X_train, self.y_train = X_train.float(), y_train.float()

        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(exist_ok=True)
        if ckpt_name is None:
            ckpt_name = f"{name}.pt"
        self.ckpt_path = self.ckpt_dir / ckpt_name
        self.metrics_path = self.ckpt_dir / "metrics.csv"

        self.start_epoch = 0
        if self.ckpt_path.exists():
            state = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optim"])
            self.start_epoch = state["epoch"] + 1

    def _penalty(self) -> torch.Tensor:
        if self.gamma <= 0:
            return torch.tensor(0.0, device=self.device)
        p = 0.0
        for attr in ("block1", "block2", "block3"):
            mod = getattr(self.model, attr, None)
            mha = getattr(mod, "mha", None) if mod is not None else None
            if mha is not None and hasattr(mha, "_get_penalty"):
                p = p + mha._get_penalty()
        return p

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        batch_bar = tqdm(self.train_loader, desc="Batchs")
        
        for xb, yb in batch_bar:
            xb, yb = xb.to(self.device), yb.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(xb).view(-1)
            yb_flat = yb.view(-1)

            loss = self.loss_fn(logits, yb_flat)
            if self.gamma > 0:
                loss = loss + self.gamma * self._penalty()
                
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                preds = torch.sigmoid(logits)
                acc = preds.gt(0.5).eq(yb_flat).float().mean().item()
                nb_one_predicted = preds.gt(0.5).sum().item()

            batch_bar.set_postfix({
                "loss_batch": f"{loss.item():.4f}",
                "acc_batch": f"{acc:.4f}",
                "nb_one_predicted": int(nb_one_predicted)
            })

            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)


    def _compute_metrics(self, loader):
        self.model.eval()
        tot_loss, tp, fp, tn, fn_, n = 0.0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb).view(-1)
                loss = self.loss_fn(logits, yb.view(-1))
                preds = torch.sigmoid(logits) > 0.5
                tot_loss += loss.item() * len(xb)
                tp += (preds & (yb == 1)).sum().item()
                fp += (preds & (yb == 0)).sum().item()
                tn += ((~preds) & (yb == 0)).sum().item()
                fn_ += ((~preds) & (yb == 1)).sum().item()
                n += len(xb)
        P, N = tp + fn_, tn + fp
        acc = (tp + tn) / n if n else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / P if P else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        denom = math.sqrt((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_))
        mcc = (tp * tn - fp * fn_) / denom if denom else 0
        return {
            "loss": tot_loss / n if n else 0,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "tp_rate": tp / P if P else 0,
            "fp_rate": fp / N if N else 0,
            "tn_rate": tn / N if N else 0,
            "fn_rate": fn_ / P if P else 0,
        }

    def _eval(self):
        return self._compute_metrics(self.val_loader)

    def _append_metrics_csv(self, df: pd.DataFrame):
        df_out = df.reset_index().rename(columns={"index": "split"})
        df_out.insert(0, "model", self.name)
        df_out.insert(1, "timestamp", datetime.now().isoformat(timespec="seconds"))
        df_out.to_csv(self.metrics_path, mode="a", header=not self.metrics_path.exists(), index=False)

    def evaluate(
            self,
            X: Optional[torch.Tensor] = None,
            y: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            save: bool = True,
            force: bool = False,  
        ):

            expected_split = "test" if X is not None else "val"
            expected_splits = ["train", expected_split]

            if not force and self.metrics_path.exists():
                try:
                    df_log = pd.read_csv(self.metrics_path)
                    df_log = df_log[df_log["model"] == self.name]   
                    if set(expected_splits).issubset(df_log["split"].unique()):

                        df_log["timestamp"] = pd.to_datetime(
                            df_log["timestamp"], errors="coerce"
                        )
                        df_latest = (
                            df_log.sort_values("timestamp")
                                .groupby("split", as_index=False)
                                .last()
                                .set_index("split")
                                .loc[expected_splits]
                        )

                        df = df_latest.drop(columns=["model", "timestamp"])

                        def _highlight(row):
                            if row.name != expected_split:
                                return [''] * len(row)
                            colors = []
                            for col in row.index:
                                if col in ("acc", "f1", "mcc"):
                                    colors.append("background-color: #4caf50")
                                elif col in ("fp_rate", "fn_rate"):
                                    colors.append("background-color: #e53935")
                                else:
                                    colors.append("")
                            return colors

                        return (
                            df.style.format("{:.4f}")
                            .apply(_highlight, axis=1)
                            .set_caption(f"{self.name} (cached)")
                        )
                except Exception:

                    pass

            metrics_train = self._compute_metrics(
                DataLoader(
                    TensorDataset(self.X_train, self.y_train),
                    batch_size=batch_size or self.batch_size,
                    shuffle=False,
                )
            )

            if X is None or y is None:
                metrics_eval = self._eval()
                df = pd.DataFrame([metrics_train, metrics_eval],
                                index=["train", "val"])
            else:
                metrics_eval = self._compute_metrics(
                    DataLoader(
                        TensorDataset(X.float(), y.float()),
                        batch_size=batch_size or self.batch_size,
                        shuffle=False,
                    )
                )
                df = pd.DataFrame([metrics_train, metrics_eval],
                                index=["train", "test"])

            if save:
                self._append_metrics_csv(df)

            target_row = expected_split

            def _highlight(row):
                if row.name != target_row:
                    return [''] * len(row)
                colors = []
                for col in row.index:
                    if col in ("acc", "f1", "mcc"):
                        colors.append("background-color: #4caf50")
                    elif col in ("fp_rate", "fn_rate"):
                        colors.append("background-color: #e53935")
                    else:
                        colors.append("")
                return colors

            return (
                df.style.format("{:.4f}")
                .apply(_highlight, axis=1)
                .set_caption(self.name)
            )

            
            return s

    def _save(self, epoch: int):
        torch.save(
            {"epoch": epoch, "model": self.model.state_dict(), "optim": self.optimizer.state_dict()},
            self.ckpt_path,
        )

    def fit(self):
        for epoch in tqdm(range(self.start_epoch, self.num_epochs), desc="Epochs"):
            tloss = self._train_epoch()
            metrics = self._eval()
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Epoch {epoch:02d} | train_loss={tloss:.4f} "
                f"| val_loss={metrics['loss']:.4f} | val_acc={metrics['acc']:.4f} "
                f"| val_f1={metrics['f1']:.4f} | val_prec={metrics['precision']:.4f} "
                f"| val_rec={metrics['recall']:.4f} | val_mcc={metrics['mcc']:.4f}"
            )
            if (epoch + 1) % self.save_every_epoch == 0:
                self._save(epoch)
