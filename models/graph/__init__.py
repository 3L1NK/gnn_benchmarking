from .static_gnn import StaticGNN

__all__ = ["StaticGNN"]

try:
    from .tgcn_static import StaticTGCN, TemporalGCNModel

    __all__.extend(["StaticTGCN", "TemporalGCNModel"])
except Exception:
    pass

try:
    from .tgat_static import StaticTGAT

    __all__.append("StaticTGAT")
except Exception:
    pass
