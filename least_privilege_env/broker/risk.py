from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


RISK_DIMS = ("write", "exec", "external", "secret", "scope", "persistent")
DEFAULT_WEIGHTS = (0.22, 0.20, 0.18, 0.16, 0.14, 0.10)


@dataclass(frozen=True)
class RiskVector:
    write: float = 0.0
    exec: float = 0.0
    external: float = 0.0
    secret: float = 0.0
    scope: float = 0.0
    persistent: float = 0.0

    def __post_init__(self) -> None:
        for name in RISK_DIMS:
            value = getattr(self, name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} risk must be in [0, 1], got {value}")

    @classmethod
    def from_mapping(cls, value: dict[str, float] | None) -> "RiskVector":
        value = value or {}
        return cls(**{dim: float(value.get(dim, 0.0)) for dim in RISK_DIMS})

    def to_dict(self) -> dict[str, float]:
        return {dim: getattr(self, dim) for dim in RISK_DIMS}

    def values(self) -> tuple[float, ...]:
        return tuple(getattr(self, dim) for dim in RISK_DIMS)

    def max(self, other: "RiskVector") -> "RiskVector":
        return RiskVector(**{dim: max(getattr(self, dim), getattr(other, dim)) for dim in RISK_DIMS})

    def positive_part_minus(self, other: "RiskVector") -> "RiskVector":
        return RiskVector(**{dim: max(0.0, getattr(self, dim) - getattr(other, dim)) for dim in RISK_DIMS})

    def leq(self, other: "RiskVector") -> bool:
        return all(getattr(self, dim) <= getattr(other, dim) + 1e-9 for dim in RISK_DIMS)

    def weighted_sum(self, weights: Iterable[float] = DEFAULT_WEIGHTS) -> float:
        return sum(v * w for v, w in zip(self.values(), tuple(weights), strict=True))


ZERO_RISK = RiskVector()
UNKNOWN_PRIOR = RiskVector(write=0.4, exec=0.5, external=0.3, secret=0.3, scope=0.6, persistent=0.3)
BLOCKED_HIGH_RISK = RiskVector(write=1.0, exec=0.8, external=0.8, secret=0.8, scope=1.0, persistent=1.0)


def max_all(risks: Iterable[RiskVector]) -> RiskVector:
    out = ZERO_RISK
    for risk in risks:
        out = out.max(risk)
    return out
