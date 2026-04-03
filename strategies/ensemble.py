"""
ensemble.py

Sistema de votación ensemble para señales de trading.
Opera solo cuando M de N estrategias están de acuerdo → mayor WR, menos trades.

Lógica:
  - Cada estrategia vota: "buy", "sell" o "hold"
  - Si votos_buy >= threshold  → señal "buy"
  - Si votos_sell >= threshold → señal "sell"
  - En caso contrario          → "hold"
  - En empate buy==sell        → "hold" (conservador)

Uso:
    from strategies.ensemble import EnsembleVoter

    voter = EnsembleVoter(threshold=3)  # 3/N acuerdo mínimo
    voter.add("smc",     SMCStrategy())
    voter.add("mtf",     MultiTFSMC())
    voter.add("macd",    MACDRSIStrategy())
    voter.add("ichimoku",IchimokuStrategy())
    voter.add("adv_mtf", AdaptiveMTFStrategy())

    signal, meta = voter.vote(df)
    # signal: "buy" | "sell" | "hold"
    # meta:   {"votes_buy": 3, "votes_sell": 1, "hold": 1, "agreeing": [...]}
"""

import numpy as np
import pandas as pd
from typing import Optional


class EnsembleVoter:
    """
    Votador ensemble: combina N estrategias y genera señal consensuada.

    Parámetros
    ----------
    threshold  : número mínimo de votos coincidentes para emitir señal (default 3)
    strategies : dict nombre→estrategia (opcionales en constructor, añadibles con .add())
    """

    def __init__(self, threshold: int = 3, strategies: Optional[dict] = None):
        self.threshold  = threshold
        self._strategies: dict = strategies or {}

    def add(self, name: str, strategy) -> "EnsembleVoter":
        """Añade una estrategia al ensemble."""
        self._strategies[name] = strategy
        return self

    def remove(self, name: str) -> "EnsembleVoter":
        self._strategies.pop(name, None)
        return self

    # ── Votación puntual (última vela) ────────────────────────────────────────

    def vote(self, df: pd.DataFrame) -> tuple[str, dict]:
        """
        Genera señal consensuada para la última vela del DataFrame.

        Retorna (señal, metadata_dict)
        """
        votes_buy, votes_sell, votes_hold = [], [], []

        for name, strat in self._strategies.items():
            try:
                sig = self._get_signal(strat, df)
            except Exception as e:
                sig = "hold"

            if sig == "buy":
                votes_buy.append(name)
            elif sig == "sell":
                votes_sell.append(name)
            else:
                votes_hold.append(name)

        nb, ns = len(votes_buy), len(votes_sell)

        if nb >= self.threshold and nb > ns:
            final = "buy"
            agreeing = votes_buy
        elif ns >= self.threshold and ns > nb:
            final = "sell"
            agreeing = votes_sell
        else:
            final = "hold"
            agreeing = []

        meta = {
            "votes_buy":   nb,
            "votes_sell":  ns,
            "votes_hold":  len(votes_hold),
            "agreeing":    agreeing,
            "total_strats": len(self._strategies),
            "threshold":   self.threshold,
        }
        return final, meta

    # ── Votación vectorial (toda la serie) ───────────────────────────────────

    def vote_batch(self, df: pd.DataFrame) -> list[str]:
        """
        Genera señales para toda la serie.  O(n·k) donde k = nº estrategias.

        Útil para backtest ensemble.
        """
        n = len(df)
        vote_matrix: dict[str, list[str]] = {}

        for name, strat in self._strategies.items():
            try:
                sigs = self._get_signals_batch(strat, df)
                if len(sigs) != n:
                    sigs = ["hold"] * n
            except Exception:
                sigs = ["hold"] * n
            vote_matrix[name] = sigs

        results = []
        for i in range(n):
            nb = sum(1 for s in vote_matrix.values() if s[i] == "buy")
            ns = sum(1 for s in vote_matrix.values() if s[i] == "sell")
            if nb >= self.threshold and nb > ns:
                results.append("buy")
            elif ns >= self.threshold and ns > nb:
                results.append("sell")
            else:
                results.append("hold")

        return results

    # ── Análisis de estrategias ───────────────────────────────────────────────

    def strategy_agreement_stats(self, df: pd.DataFrame) -> dict:
        """
        Estadísticas de acuerdo entre estrategias en el historial.
        Útil para diagnosticar qué combinaciones son más comunes.
        """
        n = len(df)
        vote_matrix: dict[str, list[str]] = {}

        for name, strat in self._strategies.items():
            try:
                sigs = self._get_signals_batch(strat, df)
                if len(sigs) != n:
                    sigs = ["hold"] * n
            except Exception:
                sigs = ["hold"] * n
            vote_matrix[name] = sigs

        # Señal ensemble
        ensemble = []
        for i in range(n):
            nb = sum(1 for s in vote_matrix.values() if s[i] == "buy")
            ns = sum(1 for s in vote_matrix.values() if s[i] == "sell")
            if nb >= self.threshold and nb > ns:
                ensemble.append("buy")
            elif ns >= self.threshold and ns > nb:
                ensemble.append("sell")
            else:
                ensemble.append("hold")

        total = n
        n_buy  = ensemble.count("buy")
        n_sell = ensemble.count("sell")
        n_hold = ensemble.count("hold")

        # Frecuencia de señal por estrategia
        strat_freq = {}
        for name, sigs in vote_matrix.items():
            strat_freq[name] = {
                "buy":  sigs.count("buy"),
                "sell": sigs.count("sell"),
                "hold": sigs.count("hold"),
                "signal_rate_pct": round((sigs.count("buy") + sigs.count("sell")) / total * 100, 1),
            }

        return {
            "total_candles":    total,
            "ensemble_buy":     n_buy,
            "ensemble_sell":    n_sell,
            "ensemble_hold":    n_hold,
            "ensemble_signal_rate_pct": round((n_buy + n_sell) / total * 100, 1),
            "threshold":        self.threshold,
            "strategy_stats":   strat_freq,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_signal(strat, df: pd.DataFrame) -> str:
        """Obtiene señal puntual — intenta varios métodos comunes."""
        # Batch method (preferred)
        if hasattr(strat, "generate_signals_batch"):
            sigs = strat.generate_signals_batch(df)
            return sigs[-1] if sigs else "hold"
        # Point method
        if hasattr(strat, "generate_signal"):
            result = strat.generate_signal(df)
            if isinstance(result, dict):
                return result.get("signal", "hold")
            return str(result)
        return "hold"

    @staticmethod
    def _get_signals_batch(strat, df: pd.DataFrame) -> list:
        """Obtiene señales para toda la serie."""
        if hasattr(strat, "generate_signals_batch"):
            return strat.generate_signals_batch(df)
        # Fallback: señal puntual (slow but compatible)
        if hasattr(strat, "generate_signal"):
            results = []
            for i in range(1, len(df) + 1):
                try:
                    r = strat.generate_signal(df.iloc[:i])
                    s = r.get("signal", "hold") if isinstance(r, dict) else str(r)
                except Exception:
                    s = "hold"
                results.append(s)
            return results
        return ["hold"] * len(df)

    def __repr__(self):
        return (f"EnsembleVoter(strategies={list(self._strategies.keys())}, "
                f"threshold={self.threshold}/{len(self._strategies)})")


# ── Configuraciones preestablecidas ──────────────────────────────────────────

def build_default_ensemble(threshold: int = 3) -> EnsembleVoter:
    """
    Ensemble por defecto con las 5 mejores estrategias del torneo.
    threshold=3 → 3/5 acuerdo mínimo (60% consenso).
    """
    from strategies.mtf_smc              import MultiTFSMC
    from strategies.smc_strategy         import SMCStrategy
    from strategies.advanced_strategies  import (
        MACDRSIStrategy, IchimokuStrategy, AdaptiveMTFStrategy,
        MarketStructureStrategy, BreakoutVolStrategy,
    )

    voter = EnsembleVoter(threshold=threshold)
    voter.add("mtf_smc",       MultiTFSMC(swing_window=5, trend_ema=50))
    voter.add("smc",           SMCStrategy(swing_window=5))
    voter.add("macd_rsi",      MACDRSIStrategy())
    voter.add("ichimoku",      IchimokuStrategy())
    voter.add("adaptive_mtf",  AdaptiveMTFStrategy())
    return voter


def build_aggressive_ensemble(threshold: int = 2) -> EnsembleVoter:
    """
    Ensemble agresivo: threshold=2 → más señales, mayor riesgo.
    Combina estrategias de momentum y ruptura.
    """
    from strategies.mtf_smc              import MultiTFSMC
    from strategies.advanced_strategies  import (
        BreakoutVolStrategy, BollingerMomStrategy,
        OrderBlockStrategy, MarketStructureStrategy,
    )
    from strategies.momentum_burst       import MomentumBurstStrategy

    voter = EnsembleVoter(threshold=threshold)
    voter.add("mtf_smc",          MultiTFSMC(swing_window=5, trend_ema=50))
    voter.add("breakout_vol",     BreakoutVolStrategy())
    voter.add("bb_momentum",      BollingerMomStrategy())
    voter.add("order_block",      OrderBlockStrategy())
    voter.add("market_structure", MarketStructureStrategy())
    voter.add("momentum_burst",   MomentumBurstStrategy())
    return voter


def build_conservative_ensemble(threshold: int = 4) -> EnsembleVoter:
    """
    Ensemble conservador: threshold=4/6 → muy pocas señales, alta precisión.
    """
    from strategies.mtf_smc              import MultiTFSMC
    from strategies.smc_strategy         import SMCStrategy
    from strategies.advanced_strategies  import (
        MACDRSIStrategy, IchimokuStrategy, AdaptiveMTFStrategy, LinearRegStrategy,
    )

    voter = EnsembleVoter(threshold=threshold)
    voter.add("mtf_smc",       MultiTFSMC(swing_window=5, trend_ema=50))
    voter.add("smc",           SMCStrategy(swing_window=5))
    voter.add("macd_rsi",      MACDRSIStrategy())
    voter.add("ichimoku",      IchimokuStrategy())
    voter.add("adaptive_mtf",  AdaptiveMTFStrategy())
    voter.add("linear_reg",    LinearRegStrategy())
    return voter
