# utils/runtime.py
class _RuntimeRegistry:      # simple namespaced bag
    retriever = None
    graph_schema = None

registry = _RuntimeRegistry()
