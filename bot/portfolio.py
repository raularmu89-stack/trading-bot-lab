"""
portfolio.py

Simulador de portfolio multi-par.

Distribuye capital entre varios pares y combina las curvas de equity
para calcular metricas a nivel de portfolio.

Modos de asignacion:
  equal       : mismo peso a todos los pares
  inv_vol     : pesos inversamente proporcionales a la volatilidad de retornos
  custom      : pesos definidos por el usuario (dict symbol -> fraccion)

Uso rapido:
    from bot.portfolio import Portfolio
    from bot.mock_data import generate_mock_klines
    from strategies.smc_strategy import SMCStrategy
    from strategies.risk_manager import RiskManager

    pairs = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"]
    data  = {s: generate_mock_klines(s, n_candles=500, timeframe="1h") for s in pairs}

    pf = Portfolio(
        pairs=pairs,
        data=data,
        strategy=SMCStrategy(),
        risk_manager=RiskManager(method="atr"),
        initial_capital=100.0,
        allocation="equal",
    )
    results = pf.run()
    pf.print_summary(results)
"""

import numpy as np
from backtests.backtester_fast import FastBacktester
from backtests.metrics import compute_all


# Periodos por año segun timeframe
_PERIODS = {
    "1m":  525_600,
    "5m":  105_120,
    "15m":  35_040,
    "1h":    8_760,
    "4h":    2_190,
    "1d":      252,
}


def _periods_for_tf(timeframe: str) -> int:
    return _PERIODS.get(timeframe, 252)


class Portfolio:
    """
    Parametros
    ----------
    pairs          : lista de simbolos (ej. ["BTCUSDT", "ETHUSDT"])
    data           : dict {symbol: DataFrame OHLCV}
    strategy       : instancia de SMCStrategy (compartida o dict por par)
    risk_manager   : instancia de RiskManager (compartida o dict por par)
    initial_capital: capital inicial total en unidades monetarias
    allocation     : "equal" | "inv_vol" | dict {symbol: weight}
    timeframe      : string para calcular periods_per_year ("1h", "4h", "1d", ...)
    max_hold       : velas maximas en posicion
    """

    def __init__(
        self,
        pairs,
        data,
        strategy,
        risk_manager=None,
        initial_capital=100.0,
        allocation="equal",
        timeframe="1h",
        max_hold=10,
    ):
        self.pairs = pairs
        self.data = data
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.initial_capital = initial_capital
        self.allocation = allocation
        self.timeframe = timeframe
        self.max_hold = max_hold

    def _get_strategy(self, symbol):
        if isinstance(self.strategy, dict):
            return self.strategy[symbol]
        return self.strategy

    def _get_risk_manager(self, symbol):
        if isinstance(self.risk_manager, dict):
            return self.risk_manager.get(symbol)
        return self.risk_manager

    def _compute_weights(self, pair_results):
        """Calcula pesos de asignacion segun el modo elegido."""
        if isinstance(self.allocation, dict):
            total = sum(self.allocation.values())
            return {s: self.allocation[s] / total for s in self.pairs}

        if self.allocation == "equal":
            w = 1.0 / len(self.pairs)
            return {s: w for s in self.pairs}

        if self.allocation == "inv_vol":
            vols = {}
            for s in self.pairs:
                eq = pair_results[s]["equity_curve"]
                if len(eq) < 2:
                    vols[s] = 1.0
                else:
                    r = np.diff(eq) / np.where(np.array(eq[:-1]) != 0,
                                               np.array(eq[:-1]), 1e-10)
                    vols[s] = float(np.std(r)) if np.std(r) > 0 else 1e-6

            inv = {s: 1.0 / vols[s] for s in self.pairs}
            total = sum(inv.values())
            return {s: inv[s] / total for s in self.pairs}

        raise ValueError(f"allocation desconocido: {self.allocation!r}")

    def run(self):
        """
        Ejecuta el backtest para cada par y combina resultados.

        Devuelve dict con:
          pairs          : resultados individuales por par
          weights        : asignacion de capital por par
          capital_per_pair: capital asignado a cada par
          portfolio_equity: curva de equity combinada (capital real)
          metrics        : metricas de riesgo del portfolio
          total_return   : retorno total del portfolio
          final_capital  : capital final
        """
        periods_per_year = _periods_for_tf(self.timeframe)
        pair_results = {}

        for symbol in self.pairs:
            bt = FastBacktester(
                strategy=self._get_strategy(symbol),
                data=self.data[symbol],
                max_hold=self.max_hold,
                risk_manager=self._get_risk_manager(symbol),
            )
            pair_results[symbol] = bt.run(periods_per_year=periods_per_year)

        weights = self._compute_weights(pair_results)
        capital_per_pair = {
            s: self.initial_capital * weights[s] for s in self.pairs
        }

        # Combinar curvas de equity ponderadas
        # Interpola todas al mismo largo (el maximo)
        max_len = max(len(pair_results[s]["equity_curve"]) for s in self.pairs)
        combined = np.zeros(max_len)

        for s in self.pairs:
            eq = np.array(pair_results[s]["equity_curve"])
            cap = capital_per_pair[s]
            # Escalar a capital real
            eq_capital = eq * cap
            # Rellenar hasta max_len repitiendo el ultimo valor
            if len(eq_capital) < max_len:
                eq_capital = np.concatenate(
                    [eq_capital,
                     np.full(max_len - len(eq_capital), eq_capital[-1])]
                )
            combined += eq_capital

        portfolio_metrics = compute_all(
            combined.tolist(), periods_per_year=periods_per_year
        )

        return {
            "pairs":            pair_results,
            "weights":          weights,
            "capital_per_pair": capital_per_pair,
            "portfolio_equity": combined.tolist(),
            "metrics":          portfolio_metrics,
            "total_return":     portfolio_metrics["total_return"],
            "final_capital":    combined[-1] if len(combined) > 0 else self.initial_capital,
        }

    @staticmethod
    def print_summary(results, initial_capital=None):
        """Imprime un resumen formateado del portfolio."""
        sep = "─" * 65

        print(sep)
        print("  PORTFOLIO SUMMARY")
        print(sep)

        # Encabezado de tabla por par
        print(f"  {'Par':<12} {'Peso':>6} {'Capital':>10} "
              f"{'Trades':>7} {'WR':>6} {'PnL%':>8} "
              f"{'Sharpe':>7} {'MaxDD':>7}")
        print("  " + "─" * 63)

        for s, res in results["pairs"].items():
            w   = results["weights"][s]
            cap = results["capital_per_pair"][s]
            pnl = res["total_pnl"] * 100
            wr  = res["winrate"] * 100
            sharpe = res.get("sharpe", 0.0)
            mdd    = res.get("max_drawdown", 0.0) * 100
            print(f"  {s:<12} {w:>5.1%} {cap:>10.2f} "
                  f"{res['trades']:>7d} {wr:>5.1f}% {pnl:>+7.2f}% "
                  f"{sharpe:>7.2f} {mdd:>+6.2f}%")

        print(sep)
        m = results["metrics"]
        ic = initial_capital or results["portfolio_equity"][0] if results["portfolio_equity"] else 0
        fc = results["final_capital"]

        print(f"  Capital inicial:  {ic:>10.2f}")
        print(f"  Capital final:    {fc:>10.2f}")
        print(f"  Retorno total:    {m['total_return']:>+9.2%}")
        print(f"  Retorno anual:    {m['ann_return']:>+9.2%}")
        print(f"  Volatilidad:      {m['volatility']:>9.2%}")
        print(f"  Max Drawdown:     {m['max_drawdown']:>+9.2%}")
        print(f"  Sharpe ratio:     {m['sharpe']:>10.4f}")
        print(f"  Sortino ratio:    {m['sortino']:>10.4f}")
        print(f"  Calmar ratio:     {m['calmar']:>10.4f}")
        print(sep)
