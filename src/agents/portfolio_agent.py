

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)

AssetType = Literal["stock", "bond", "cash", "etf", "mutual_fund", "crypto", "other"]


@dataclass(frozen=True)
class Holding:
    """Single portfolio holding.

    MVP rule: we require *value_usd* for each holding so we can compute weights deterministically.
    Market-price lookup can be added later.
    """

    symbol: str
    asset_type: AssetType
    value_usd: float
    quantity: Optional[float] = None
    cost_basis_usd: Optional[float] = None
    expense_ratio: Optional[float] = None  # e.g., 0.0003 for 0.03%


@dataclass(frozen=True)
class PortfolioInput:
    holdings: List[Holding]
    cash_usd: float = 0.0
    account_type: Optional[Literal["taxable", "ira", "401k", "other"]] = None


@dataclass(frozen=True)
class PortfolioMetrics:
    total_value_usd: float
    weights: Dict[str, float]  # symbol -> weight 0..1
    allocation: Dict[str, float]  # asset_type -> weight 0..1
    top_holdings: List[Tuple[str, float]]  # [(symbol, weight)] sorted desc
    concentration_top_1: float
    concentration_top_5: float
    herfindahl_index: float  # sum(w^2)
    weighted_expense_ratio: Optional[float]
    unknowns: List[str]


class PortfolioValidationError(ValueError):
    pass


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def validate_portfolio(payload: Dict[str, Any]) -> PortfolioInput:
    """Validate and normalize a user-provided portfolio payload.

    Expected shape (MVP):
      {
        "holdings": [
          {"symbol": "VOO", "asset_type": "etf", "value_usd": 12000.0},
          {"symbol": "BND", "asset_type": "etf", "value_usd": 3000.0, "expense_ratio": 0.0003},
        ],
        "cash_usd": 500.0,
        "account_type": "taxable"
      }

    Hard rule for MVP: each holding must include value_usd > 0.
    """

    if not isinstance(payload, dict):
        raise PortfolioValidationError("Portfolio payload must be a dict.")

    holdings_raw = payload.get("holdings")
    if not isinstance(holdings_raw, list) or len(holdings_raw) == 0:
        raise PortfolioValidationError("Portfolio payload must include a non-empty 'holdings' list.")

    cash_usd = float(payload.get("cash_usd", 0.0) or 0.0)
    if cash_usd < 0:
        raise PortfolioValidationError("cash_usd cannot be negative.")

    account_type = payload.get("account_type")
    if account_type is not None and account_type not in {"taxable", "ira", "401k", "other"}:
        raise PortfolioValidationError("account_type must be one of: taxable, ira, 401k, other")

    holdings: List[Holding] = []
    for i, h in enumerate(holdings_raw):
        if not isinstance(h, dict):
            raise PortfolioValidationError(f"holding[{i}] must be an object")

        symbol = h.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise PortfolioValidationError(f"holding[{i}].symbol must be a non-empty string")
        symbol_n = _normalize_symbol(symbol)

        asset_type = h.get("asset_type")
        if asset_type not in {"stock", "bond", "cash", "etf", "mutual_fund", "crypto", "other"}:
            raise PortfolioValidationError(
                f"holding[{i}].asset_type must be one of: stock, bond, cash, etf, mutual_fund, crypto, other"
            )

        try:
            value_usd = float(h.get("value_usd"))
        except Exception:
            raise PortfolioValidationError(f"holding[{i}].value_usd must be a number")
        if value_usd <= 0:
            raise PortfolioValidationError(f"holding[{i}].value_usd must be > 0")

        quantity = h.get("quantity")
        quantity_f = float(quantity) if quantity is not None else None

        cost_basis = h.get("cost_basis_usd")
        cost_basis_f = float(cost_basis) if cost_basis is not None else None

        exp = h.get("expense_ratio")
        exp_f = float(exp) if exp is not None else None
        if exp_f is not None and (exp_f < 0 or exp_f > 0.10):
            raise PortfolioValidationError(
                f"holding[{i}].expense_ratio looks invalid; expected a decimal like 0.0003 (0.03%)"
            )

        holdings.append(
            Holding(
                symbol=symbol_n,
                asset_type=asset_type,  # type: ignore[arg-type]
                value_usd=value_usd,
                quantity=quantity_f,
                cost_basis_usd=cost_basis_f,
                expense_ratio=exp_f,
            )
        )

    return PortfolioInput(holdings=holdings, cash_usd=cash_usd, account_type=account_type)


def compute_metrics(portfolio: PortfolioInput) -> PortfolioMetrics:
    """Compute deterministic portfolio metrics (no LLM).

    MVP includes:
    - Total value
    - Weights by holding
    - Allocation by asset_type
    - Concentration (top-1, top-5, HHI)
    - Weighted expense ratio (if enough data)
    """

    unknowns: List[str] = []

    holdings_value = sum(h.value_usd for h in portfolio.holdings)
    total_value = holdings_value + portfolio.cash_usd
    if total_value <= 0:
        raise PortfolioValidationError("Total portfolio value must be > 0.")

    # Symbol weights (cash is not included as a symbol weight)
    weights: Dict[str, float] = {}
    for h in portfolio.holdings:
        weights[h.symbol] = weights.get(h.symbol, 0.0) + (h.value_usd / total_value)

    # Allocation by asset type (include cash bucket)
    allocation: Dict[str, float] = {}
    for h in portfolio.holdings:
        allocation[h.asset_type] = allocation.get(h.asset_type, 0.0) + (h.value_usd / total_value)
    if portfolio.cash_usd > 0:
        allocation["cash"] = allocation.get("cash", 0.0) + (portfolio.cash_usd / total_value)

    # Top holdings
    top_holdings = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    top1 = top_holdings[0][1] if top_holdings else 0.0
    top5 = sum(w for _, w in top_holdings[:5])

    # Herfindahl-Hirschman Index (HHI) on holding weights (cash excluded)
    hhi = sum(w * w for w in weights.values())

    # Weighted expense ratio (only for holdings where expense_ratio is provided)
    weighted_exp: Optional[float] = None
    exp_weight_sum = 0.0
    exp_contrib = 0.0
    for h in portfolio.holdings:
        if h.expense_ratio is None:
            continue
        w = h.value_usd / total_value
        exp_weight_sum += w
        exp_contrib += w * h.expense_ratio

    if exp_weight_sum > 0:
        weighted_exp = exp_contrib / exp_weight_sum
    else:
        unknowns.append("No expense_ratio provided for any holding; weighted fee could not be computed.")

    return PortfolioMetrics(
        total_value_usd=total_value,
        weights=weights,
        allocation=allocation,
        top_holdings=top_holdings[:10],
        concentration_top_1=top1,
        concentration_top_5=top5,
        herfindahl_index=hhi,
        weighted_expense_ratio=weighted_exp,
        unknowns=unknowns,
    )


def run_portfolio_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Portfolio Analysis Agent (MVP).

    Returns JSON-serializable dict for UI rendering.
    No LLM is used here yet; this is deterministic analysis.
    """

    portfolio = validate_portfolio(payload)
    metrics = compute_metrics(portfolio)

    summary = (
        f"Portfolio total value: ${metrics.total_value_usd:,.2f}. "
        f"Top holding concentration: {metrics.concentration_top_1:.1%}. "
        f"Top 5 holdings concentration: {metrics.concentration_top_5:.1%}."
    )

    return {
        "summary": summary,
        "total_value_usd": metrics.total_value_usd,
        "allocation": metrics.allocation,
        "top_holdings": metrics.top_holdings,
        "concentration": {
            "top_1": metrics.concentration_top_1,
            "top_5": metrics.concentration_top_5,
            "hhi": metrics.herfindahl_index,
        },
        "weighted_expense_ratio": metrics.weighted_expense_ratio,
        "unknowns": metrics.unknowns,
        "disclaimer": "Educational information only — not financial, tax, or legal advice.",
    }