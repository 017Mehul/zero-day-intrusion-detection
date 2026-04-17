"""Reporting: deployment reports and paper artifact generation."""

__all__ = [
    "build_deployment_report",
    "write_deployment_report",
    "collect_model_sizes",
    "generate_paper_artifacts",
]


def __getattr__(name):
    if name in __all__:
        from sa_zd_nids.reporting.deployment import (  # noqa: F401
            build_deployment_report, write_deployment_report, collect_model_sizes
        )
        from sa_zd_nids.reporting.paper_artifacts import generate_paper_artifacts  # noqa: F401
        _map = {
            "build_deployment_report": build_deployment_report,
            "write_deployment_report": write_deployment_report,
            "collect_model_sizes": collect_model_sizes,
            "generate_paper_artifacts": generate_paper_artifacts,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
