"""MCP server exposing the deployed UMA-Inverse model as an agent-callable tool.

See :mod:`src.mcp.server`. The single tool ``design_sequence_for_structure``
calls the deployed ``/design`` endpoint over HTTP and returns a markdown result,
so an agent can retrieve a structure (e.g. via genesis-bio-mcp) and then ask
this model to redesign it — same agent, end to end.
"""
