import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import requests, warnings
from typing import List, Union, Optional, Tuple
from numpy.lib.stride_tricks import sliding_window_view

class MarketDataset:
    
    """
    Classe qui gère le téléchargement des données et la création du dataset pour les données daily
    • tickers : liste des tickers à télécharger
    • start : date de début (format YYYY-MM-DD)
    • end : date de fin (format YYYY-MM-DD)
    • freq : fréquence des données (1d)
    • beta_rise : seuil de hausse pour le label positif (faire en sorte qu'il se crée automatiquement?)
    • beta_fall : seuil de baisse pour le label négatif (faire en sorte qu'il se crée automatiquement?)
    • horizon_k : nombre de pas à prédire (1, 2, 3, ...)
    • window_N : taille de la fenêtre glissante (20)
    • stride : pas de la fenêtre glissante (1) (pour faire des fenêtres qui se chevauchent ou non)
    """
    
    def __init__(
        self,
        tickers: List[str],
        start: str,
        end: str,
        freq: str = "1d",
        beta_rise: float = 0.0055,
        beta_fall: float = -0.001,
        horizon_k: int = 1,
        window_N: int = 20,
        stride: int = 1,
        verbose: bool = True
    ):
        self.tickers    = tickers
        self.start      = start
        self.end        = end
        self.freq       = freq
        self.beta_rise  = beta_rise
        self.beta_fall  = beta_fall
        self.horizon_k  = horizon_k
        self.window_N   = window_N
        self.stride     = stride
        self.verbose    = verbose

        self._download()
        self._engineer_features()
        self._build_examples()

    def _download(self) -> None:
        df = yf.download(
            tickers     = self.tickers,
            start       = self.start,
            end         = self.end,
            interval    = self.freq,
            group_by    = "ticker",
            progress    = False,
            auto_adjust = False,
            threads     = False,
        )

        good_tickers = [
            tkr for tkr in self.tickers
            if (tkr, "Close") in df.columns and not df[(tkr, "Close")].isna().all()
        ]

        missing = sorted(set(self.tickers) - set(good_tickers))
        if missing and self.verbose:
            print(f"[!] {len(missing)} ticker(s) sans données : {', '.join(missing)}")

        df = df.loc[:, df.columns.get_level_values(0).isin(good_tickers)]
        if df.empty:
            raise RuntimeError("Aucune donnée récupérée.")


        df = (df.stack(level=0)
                .reorder_levels([1, 0])
                .sort_index())

        self.raw = df


    def _engineer_features(self) -> None:
        def _feat_one(tdf: pd.DataFrame) -> pd.DataFrame:
            tdf = tdf.copy()
            tdf["close_prev"]      = tdf["Close"].shift(1)
            tdf["adj_close_prev"]  = tdf["Adj Close"].shift(1)
            tdf["c_open"]          = tdf["Open"] / tdf["Close"] - 1
            tdf["c_high"]          = tdf["High"] / tdf["Close"] - 1
            tdf["c_low"]           = tdf["Low"]  / tdf["Close"] - 1
            tdf["n_close"]         = tdf["Close"] / tdf["close_prev"]      - 1
            tdf["n_adj_close"]     = tdf["Adj Close"] / tdf["adj_close_prev"] - 1
            tdf["c_volume"]        = tdf["Volume"] / tdf["Volume"].shift(1) - 1
            return tdf[["c_open", "c_high", "c_low", "n_close", "n_adj_close"]]

        feats = (self.raw
                 .groupby(level=0, group_keys=False)
                 .apply(_feat_one)
                 .dropna())
        self.features = feats


    def _build_examples(self) -> None:
        X_list, y_list, ts_list, tk_list = [], [], [], []

        for ticker, tdf in self.features.groupby(level=0):
            tdf = tdf.droplevel(0)              # index = dates
            arr = tdf.values.astype(np.float32)

            min_len = self.window_N + self.horizon_k
            if len(arr) < min_len:
                if self.verbose:
                    print(f"[!] {ticker} ignoré (< {min_len} lignes)")
                continue

            dates = tdf.index.to_numpy(dtype="datetime64[ns]")
            windows = sliding_window_view(
                arr, window_shape=(self.window_N, arr.shape[1])
            )[:: self.stride, 0]

            close_ret   = tdf["n_close"].to_numpy()
            future_ret  = close_ret[self.window_N + self.horizon_k - 1 :]
            future_date = dates[self.window_N + self.horizon_k - 1 :]

            mask_pos = future_ret > self.beta_rise
            mask_neg = future_ret < self.beta_fall
            keep     = mask_pos | mask_neg

            if keep.any():
                X_keep  = windows[: len(future_ret)][keep]
                y_keep  = np.where(mask_pos[keep], 1, 0).astype(np.int64)
                ts_keep = future_date[keep]
                tk_keep = np.repeat(ticker, len(ts_keep))

                X_list.append(X_keep)
                y_list.append(y_keep)
                ts_list.append(ts_keep)
                tk_list.append(tk_keep)

        if not X_list:
            raise RuntimeError("Aucune fenêtre valide générée.")

        self.X           = np.vstack(X_list)
        self.y           = np.concatenate(y_list)
        self.timestamps  = np.concatenate(ts_list)
        self.tickers_idx = np.concatenate(tk_list) 

    def get_splits(self, tensor: bool = False
                   ) -> Tuple[np.ndarray, ...]:
        order = np.argsort(self.timestamps)
        self.X, self.y = self.X[order], self.y[order]
        self.timestamps = self.timestamps[order]
        self.tickers_idx = self.tickers_idx[order]

        split_80      = int(0.8 * len(self.timestamps))
        cutoff_date   = self.timestamps[split_80]

        mask_train = self.timestamps <= cutoff_date
        remaining  = ~mask_train

        tmp_dates     = self.timestamps[remaining]
        split_valtest = int(0.5 * len(tmp_dates))
        cutoff_val    = tmp_dates[split_valtest]

        mask_val  = (self.timestamps > cutoff_date) & (self.timestamps <= cutoff_val)
        mask_test = self.timestamps > cutoff_val

        splits = []
        for m in (mask_train, mask_val, mask_test):
            splits.extend([self.X[m], self.y[m]])

        if tensor:
            to_float  = lambda x: torch.from_numpy(x).float()
            to_long   = lambda x: torch.from_numpy(x).long()
            splits = [to_float(s) if i % 2 == 0 else to_long(s)
                      for i, s in enumerate(splits)]

        return tuple(splits)


    def get_dataset(
        self,
        start: Optional[str] = None,
        end:   Optional[str] = None,
        tickers: Optional[Union[str, List[str]]] = None,
        tensor: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Récupère (X, y, timestamps, tickers) filtrés.
        • start / end : YYYY-MM-DD  (aucun => non filtré)
        • tickers     : str ou liste (None => tous)
        • tensor=True : retourne torch.Tensor
        """
        mask = np.ones(len(self.timestamps), dtype=bool)

        if start is not None or end is not None:
            start_dt = np.datetime64(start) if start else self.timestamps.min()
            end_dt   = np.datetime64(end)   if end   else self.timestamps.max()
            mask &= (self.timestamps >= start_dt) & (self.timestamps <= end_dt)

        if tickers is not None:
            if isinstance(tickers, str):
                tickers = [tickers]
            tickers = set(tickers)
            mask &= np.isin(self.tickers_idx, list(tickers))

        X_sel, y_sel = self.X[mask], self.y[mask]
        ts_sel       = self.timestamps[mask]
        tk_sel       = self.tickers_idx[mask]

        if tensor:
            X_sel = torch.from_numpy(X_sel).float()
            y_sel = torch.from_numpy(y_sel).long()
            ts_sel = torch.from_numpy(ts_sel.astype("datetime64[ns]"))
            tk_sel = np.array(tk_sel)  # reste ndarray (labels string)

        return X_sel, y_sel, ts_sel, tk_sel


    def set_window(self, window_N: int, stride: Optional[int] = None):
        self.window_N = window_N
        if stride is not None:
            self.stride = stride
        self._build_examples()

    def set_betas(self, beta_rise: float, beta_fall: float):
        self.beta_rise = beta_rise
        self.beta_fall = beta_fall
        self._build_examples()
        
        
        
