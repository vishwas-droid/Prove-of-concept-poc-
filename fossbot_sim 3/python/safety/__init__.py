"""
safety package — Chapter 15
============================
FOSSBot Guardian: Dual-Layer Safety Guardrails

    from safety.middleware import SafetyMiddleware, SafetyConfig
"""
from .middleware import SafetyMiddleware, SafetyConfig
__all__ = ["SafetyMiddleware", "SafetyConfig"]
