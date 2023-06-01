"""A module implementing several executor wrappers"""
from contextlib import contextmanager
from datetime import datetime, timedelta
from distributed import as_completed

from .. import log


class DaskExecutor:
    """A executor wrapper for Dask"""

    _PROCESS_INTERVAL = timedelta(minutes=1)

    def __init__(self, executor, result_hook=None) -> None:
        """Initializes the executor wrapper"""
        self._executor = executor
        self._result_hook = result_hook
        self._jobs = []
        self._last_processing = datetime.now()

    def _timed_process_jobs(self) -> bool:
        if datetime.now() - self._last_processing > self._PROCESS_INTERVAL:
            self.process_jobs(self._jobs)
            # Use the time after processing to get an evenly timed job submission window
            self._last_processing = datetime.now()

    def submit(self, func, args, extra_data):
        """Submits a new routine to be run by the distributed framework"""
        job = self._executor.submit(func, *args)
        job.extra_data = extra_data
        self._timed_process_jobs()
        self._jobs.append(job)

    def process_jobs(self, jobs=None):
        """Process completed jobs, blocking to wait for all jobs if none are passed"""
        if not jobs:
            jobs = as_completed(self._jobs)
        self._jobs = []
        for job in jobs:
            if job.done():
                if self._result_hook:
                    self._result_hook(job.result(), job.extra_data)
                job.release()
            else:
                self._jobs.append(job)


class SerialExecutor:
    """The serial executor wrapper, which immediately runs the user function"""

    def __init__(self, result_hook) -> None:
        """Initializes the serial executor wrapper"""
        self._result_hook = result_hook

    def submit(self, func, args, extra_data):
        """Submits a new routine (which is run immediately)"""
        result = func(*args)  # Run it inplace
        if self._result_hook:
            self._result_hook(result, extra_data)
        return result


@contextmanager
def serial_ctx(result_hook):
    """A plain serial executor, basically no-op"""
    yield SerialExecutor(result_hook)
    # DONE!


@contextmanager
def dask_ctx(result_hook, executor_params: dict):
    """An executor using the Dask system"""
    from dask.distributed import Client

    # Dask requires numeric params to go as the native type
    for k, val in executor_params.items():
        if val.isdecimal():
            try:
                executor_params[k] = int(val)
            except ValueError:
                executor_params[k] = float(val)

    client = Client(**executor_params)
    executor_wrapper = DaskExecutor(client, result_hook)

    yield executor_wrapper

    log.info("Jobs submitted to DASK")

    executor_wrapper.process_jobs()

    log.info("DASK jobs finished")


def in_context(options, params, result_hook=None):
    """An auto-selector of the executor context manager"""
    log.info("Starting Execution context")
    if options.parallel:
        return dask_ctx(result_hook, params)
    return serial_ctx(result_hook)
