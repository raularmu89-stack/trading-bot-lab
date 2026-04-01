"""
scenario_router.py

Router automático de estrategias por régimen de mercado.

Detecta el régimen actual y delega a la estrategia especializada correcta:

  Régimen                 → Estrategia
  ──────────────────────────────────────────────────────────────
  strong_trend_bull/bear  → TrendRiderStrategy
  weak_trend_bull/bear    → OBRejectionStrategy  (pullback a OB)
  ranging                 → RangeScalperStrategy
  breakout                → BreakoutStrategy
  mean_reversion_*        → MeanReversionStrategy
  high_volatility         → HOLD (reducir riesgo)
  insufficient_data       → SMCStrategy (fallback)

También puede ejecutarse en modo backtest con precompute_all() que
clasifica régimen vela a vela y aplica la señal del router.

Uso live:
    router = ScenarioRouter()
    result = router.generate_signal(df)
    # → {"signal": "buy", "regime": "strong_trend_bull",
    #    "strategy_used": "TrendRiderStrategy", "reason": "..."}

Uso backtest (compatible con FastBacktester/KellyBacktester):
    router = ScenarioRouter()
    # Usar como estrategia en KellyBacktester:
    bt = KellyBacktester(router, data, sizer=sizer)
    result = bt.run()
"""

from indicators.regime_detector import RegimeDetector

from strategies.smc_strategy        import SMCStrategy
from strategies.trend_rider         import TrendRiderStrategy
from strategies.range_scalper       import RangeScalperStrategy
from strategies.breakout_strategy   import BreakoutStrategy
from strategies.ob_rejection        import OBRejectionStrategy
from strategies.mean_reversion      import MeanReversionStrategy
from strategies.volatility_filter   import VolatilityFilterStrategy
from strategies.mtf_strategy        import MTFStrategy
from strategies.macd_divergence     import MACDDivergenceStrategy
from strategies.bollinger_squeeze   import BollingerSqueezeStrategy
from strategies.momentum_burst      import MomentumBurstStrategy


# Mapeo régimen → nombre de estrategia
REGIME_TO_STRATEGY = {
    "strong_trend_bull":  "trend_rider",
    "strong_trend_bear":  "trend_rider",
    "weak_trend_bull":    "ob_rejection",
    "weak_trend_bear":    "ob_rejection",
    "ranging":            "range_scalper",
    "breakout":           "breakout",
    "mean_reversion_bull":"mean_reversion",
    "mean_reversion_bear":"mean_reversion",
    "high_volatility":    "hold",          # no operar
    "insufficient_data":  "smc_fallback",
}


class ScenarioRouter:
    """
    Router automático que selecciona la estrategia según el régimen actual.

    Parámetros
    ----------
    swing_window         : compat FastBacktester (usado por fallback SMC)
    adx_period           : período ADX para detector de régimen
    ema_fast/slow        : EMAs para detector de régimen
    override_strategy    : forzar siempre una estrategia específica (debug)
    use_mtf_trend        : usar MTFStrategy en vez de TrendRider para tendencias fuertes
    regime_lookback      : velas de lookback para el detector de régimen
    verbose              : incluir detalles de régimen en la señal
    """

    def __init__(self,
                 swing_window: int = 5,
                 adx_period: int = 14,
                 ema_fast: int = 20,
                 ema_slow: int = 50,
                 override_strategy: str = None,
                 use_mtf_trend: bool = False,
                 verbose: bool = True,
                 # Parámetros de cada sub-estrategia
                 trend_adx_threshold: float = 22.0,
                 trend_pullback_pct: float = 0.008,
                 range_entry_zone: float = 0.18,
                 breakout_consol_bars: int = 30,
                 ob_fvg_confirm: bool = False,
                 mr_rsi_oversold: float = 32.0,
                 mr_rsi_overbought: float = 68.0,
                 ):
        self.swing_window      = swing_window
        self.override_strategy = override_strategy
        self.use_mtf_trend     = use_mtf_trend
        self.verbose           = verbose

        # FastBacktester compat
        self.require_fvg      = False
        self.use_choch_filter = False

        # Detector de régimen
        self.detector = RegimeDetector(
            adx_period=adx_period,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
        )

        # Sub-estrategias
        self._strategies = {
            "smc_fallback": SMCStrategy(swing_window=swing_window),
            "volatility_filter": VolatilityFilterStrategy(swing_window=swing_window),
            "trend_rider": TrendRiderStrategy(
                swing_window=swing_window,
                adx_threshold=trend_adx_threshold,
                pullback_pct=trend_pullback_pct,
            ),
            "ob_rejection": OBRejectionStrategy(
                swing_window=swing_window,
                ob_window=swing_window,
                fvg_confirm=ob_fvg_confirm,
            ),
            "range_scalper": RangeScalperStrategy(
                swing_window=swing_window,
                entry_zone_pct=range_entry_zone,
            ),
            "breakout": BreakoutStrategy(
                swing_window=swing_window,
                consolidation_bars=breakout_consol_bars,
            ),
            "mean_reversion": MeanReversionStrategy(
                swing_window=swing_window,
                rsi_oversold=mr_rsi_oversold,
                rsi_overbought=mr_rsi_overbought,
            ),
        }
        # Estrategias adicionales disponibles (asignables via override_strategy)
        self._strategies["macd_divergence"]   = MACDDivergenceStrategy()
        self._strategies["bollinger_squeeze"] = BollingerSqueezeStrategy()
        self._strategies["momentum_burst"]    = MomentumBurstStrategy()

        if use_mtf_trend:
            self._strategies["trend_rider"] = MTFStrategy(
                high_tf="4h", high_tf_window=10, low_tf_window=swing_window,
                use_pd_filter=False,
            )

    def generate_signal(self, data) -> dict:
        """Detecta régimen y delega a la estrategia correcta."""
        # Detectar régimen
        regime_info = self.detector.detect(data)
        regime      = regime_info.get("regime", "insufficient_data")

        # Override para debug
        strat_name = self.override_strategy or REGIME_TO_STRATEGY.get(regime, "smc_fallback")

        # Alta volatilidad → delegar a VolatilityFilterStrategy
        if strat_name == "hold":
            strat_name = "volatility_filter"

        strategy = self._strategies.get(strat_name, self._strategies["smc_fallback"])
        result   = strategy.generate_signal(data)
        if self.verbose and strat_name == "volatility_filter":
            result["size_reduction"] = result.get("size_reduction",
                                                   strategy.size_reduction_factor)

        if self.verbose:
            result["regime"]        = regime
            result["regime_label"]  = RegimeDetector.regime_label(regime)
            result["strategy_used"] = strat_name
            if "adx" not in result:
                result["adx"]       = regime_info.get("adx")
            result["atr_ratio"]     = regime_info.get("atr_ratio")

        return result

    def strategy_for(self, regime: str):
        """Devuelve la instancia de estrategia para un régimen dado (para inspección)."""
        name = REGIME_TO_STRATEGY.get(regime, "smc_fallback")
        return self._strategies.get(name)

    @property
    def strategy_map(self) -> dict:
        """Devuelve el mapa régimen → nombre de estrategia."""
        return REGIME_TO_STRATEGY.copy()


# ── Señales precomputadas para KellyBacktester ────────────────────────────────

def precompute_router_signals(data, router: ScenarioRouter) -> list:
    """
    Calcula señales vela a vela aplicando el router en cada punto.
    Necesario para el KellyBacktester que usa _precompute_signals internamente.

    NOTA: Este método es O(n) llamadas a generate_signal — puede ser lento
    para datasets grandes. Para 15m de 1 año (~35k velas) usa ventanas móviles.
    """
    import pandas as pd
    n       = len(data)
    signals = ["hold"] * n
    min_len = 60  # mínimo de velas para detectar régimen

    for i in range(min_len, n):
        window = data.iloc[: i + 1]
        result = router.generate_signal(window)
        signals[i] = result.get("signal", "hold")

    return signals


def precompute_router_signals_fast(data, router: ScenarioRouter,
                                   step: int = 1) -> list:
    """
    Versión optimizada: precomputa regímenes de golpe con detect_regime_series,
    luego aplica la sub-estrategia apropiada por bloques de régimen.

    step: recalcular régimen cada N velas (tradeoff velocidad/precisión).
    """
    from backtests.backtester_fast import _precompute_signals
    import pandas as pd

    n = len(data)
    if n < 60:
        return ["hold"] * n

    # Régimen para cada vela
    all_regimes = router.detector.detect_all(data)

    # Precomputa señales para cada sub-estrategia
    strat_signals = {}
    for strat_name, strat in router._strategies.items():
        try:
            sigs = _precompute_signals(
                data,
                swing_window=strat.swing_window,
                require_fvg=getattr(strat, "require_fvg", False),
                use_choch_filter=getattr(strat, "use_choch_filter", False),
            )
            strat_signals[strat_name] = sigs
        except Exception:
            strat_signals[strat_name] = ["hold"] * n

    # Combina: para cada vela, usa la señal de la estrategia del régimen
    combined = []
    for i, regime in enumerate(all_regimes):
        strat_name = REGIME_TO_STRATEGY.get(regime, "smc_fallback")
        if strat_name == "hold":
            combined.append("hold")
        else:
            combined.append(strat_signals.get(strat_name, ["hold"] * n)[i])

    return combined
