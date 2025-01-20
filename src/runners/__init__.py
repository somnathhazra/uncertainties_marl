REGISTRY = {}

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .parallel_runner_dist import ParallelRunnerDist
REGISTRY["parallel_dist"] = ParallelRunnerDist
