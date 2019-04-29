import os

class TaskRunner(object):
    """Helper for running tasks on the univa grid engine.

    Usage: instantiate this class, then call do_task multiple times.
    If SGE_TASK_ID is not present in the environment, or SGE_TASK_ID
    is the same as the number of times do_task has been called (starting at 1!), then the task
    provided will be executed.
    Otherwise, the task will be skipped. We assume another process will
    be created with the appropriate SGE_TASK_ID so that the task will be executed there.

    It is imperative that when submitting the job with qsub, the correct task id range
    is used such that all tasks will eventually be run. So the command should look something
    like

    qsub -t 1-X
    where X is the the number of tasks expected to run. Note we don't do 0-indexing here because
    the grid does not task ids of 0.

    Example:
    task_runner = TaskRunner()
    task = lambda trial: do_something_interesting(trial)
    for trial in range(10):
      task_runner.do_task(task, trial)"""
    def __init__(self):
        self.called = 1

    def do_task(self, task, *params):
        task_id = os.environ.get('SGE_TASK_ID')
        if not task_id or int(task_id) == self.called:
            task(*params)
        self.called += 1
