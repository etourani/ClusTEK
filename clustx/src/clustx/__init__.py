#from .core import run_pipeline_2d          # 2D
#from .core3d import run_pipeline_3d     # 3D
#__all__ = ["run_pipeline"]#, "run_pipeline_3d"]

from .core import run_pipeline_2d as run_pipeline

__all__ = ["run_pipeline"]
