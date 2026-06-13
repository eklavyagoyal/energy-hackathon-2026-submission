"""FastAPI surface for the battery feature (bolt-on).

Holds exactly one live pandapower net plus a cached baseline N-1 sweep,
and exposes the battery recommendation and verification endpoints on top
of the Phase 1 engine. Same boundary rules as docs/02-architecture.md:
the API serves engine/battery outputs; it never computes physics itself.
"""
