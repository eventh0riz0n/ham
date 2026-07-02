"""Test bootstrap.

The plugin subclasses ``agent.memory_provider.MemoryProvider`` from the
Hermes host. When the suite runs inside a Hermes checkout (the normal case,
see the header of test_ham_v2.py) the real ABC is used. In CI — where there
is no Hermes installation — we register a minimal structural stand-in so the
plugin module can import. The stub only mirrors the contract surface the
plugin relies on; it must not grow behavior.
"""

from __future__ import annotations

import sys
import types


def _install_agent_stub() -> None:
    try:
        import agent.memory_provider  # noqa: F401 — real host available
        return
    except ImportError:
        pass

    from abc import ABC, abstractmethod
    from typing import Any, Dict, List, Optional

    class MemoryProvider(ABC):
        @property
        @abstractmethod
        def name(self) -> str: ...

        @abstractmethod
        def is_available(self) -> bool: ...

        @abstractmethod
        def initialize(self, session_id: str, **kwargs) -> None: ...

        @abstractmethod
        def get_tool_schemas(self) -> List[Dict[str, Any]]: ...

        def system_prompt_block(self) -> str: return ""
        def prefetch(self, query: str, *, session_id: str = "") -> str: return ""
        def queue_prefetch(self, query: str, *, session_id: str = "") -> None: ...
        def sync_turn(self, user_content: str, assistant_content: str, *,
                      session_id: str = "",
                      messages: Optional[List[Dict[str, Any]]] = None) -> None: ...
        def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
            raise NotImplementedError
        def shutdown(self) -> None: ...
        def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None: ...
        def on_session_end(self, messages: List[Dict[str, Any]]) -> None: ...
        def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "",
                              reset: bool = False, rewound: bool = False, **kwargs) -> None: ...
        def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str: return ""
        def on_delegation(self, task: str, result: str, *,
                          child_session_id: str = "", **kwargs) -> None: ...
        def get_config_schema(self) -> List[Dict[str, Any]]: return []
        def save_config(self, values: Dict[str, Any], hermes_home: str) -> None: ...
        def on_memory_write(self, action: str, target: str, content: str,
                            metadata: Optional[Dict[str, Any]] = None) -> None: ...
        def backup_paths(self) -> List[str]: return []

    agent_pkg = types.ModuleType("agent")
    provider_mod = types.ModuleType("agent.memory_provider")
    provider_mod.MemoryProvider = MemoryProvider
    agent_pkg.memory_provider = provider_mod
    sys.modules["agent"] = agent_pkg
    sys.modules["agent.memory_provider"] = provider_mod


_install_agent_stub()
