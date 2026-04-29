"""Proxy exports with lazy imports to avoid package-level import cycles."""

__all__ = ["OracleProxy", "SurrogateProxy"]


def __getattr__(name: str):
    if name == "OracleProxy":
        from proxies.oracle_proxy import OracleProxy

        return OracleProxy
    if name == "SurrogateProxy":
        from proxies.surrogate_proxy import SurrogateProxy

        return SurrogateProxy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
