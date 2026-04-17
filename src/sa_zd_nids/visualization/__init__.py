"""Visualization: experiment plots and system architecture diagrams."""

__all__ = ["generate_experiment_plots", "generate_system_diagram_files"]


def __getattr__(name):
    if name in __all__:
        from sa_zd_nids.visualization.plots import generate_experiment_plots  # noqa: F401
        from sa_zd_nids.visualization.system_diagram import generate_system_diagram_files  # noqa: F401
        return {"generate_experiment_plots": generate_experiment_plots,
                "generate_system_diagram_files": generate_system_diagram_files}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
