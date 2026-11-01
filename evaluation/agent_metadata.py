"""Trusted local agent.yaml parsing, safe before config/Isaac initialization.

Callers retain file/mapping/failure policy. Isaac-tagged env.yaml snapshots use
their separate loader; legacy unsafe YAML must only come from trusted checkpoints.
"""
import yaml


def parse_agent_yaml(text, *, legacy_errors=(Exception,), on_legacy=None):
    try:
        return yaml.safe_load(text)
    except legacy_errors as error:
        if on_legacy is not None:
            on_legacy(error)
        return yaml.unsafe_load(text)
