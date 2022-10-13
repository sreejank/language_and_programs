from .grid import *


def create_all_grid_tasks():
    import itertools

    # Generate all goal states
    all_goals = [
        np.array(board).reshape((4, 4))
        for board in itertools.product((0, 1), repeat=16)
    ]

    # Make them into tasks
    tasks = [
        GridTask(
            f'exhaustive{i}',
            start=np.zeros((4, 4)),
            goal=goal,
            location=(-1, -1),
            try_all_start=True,
            partial_progress_weight=10.,
            # invtemp is default value of 1.
        )
        for i, goal in enumerate(all_goals)
    ]

    return tasks


def recognitionEnumeration(
    recognitionModel, task, *,
    enumerationTimeout,
    evaluationTimeout=1,
    testing=False,
    maximumFrontiers=5,
    budgetIncrement=1.5,
):
    '''
    A routine meant for enumeration using a recognition model. Mimics scheme
    used to call into solver by multicoreEnumeration, where increasing bounds
    on prior are used until a timeout is reached.
    '''

    def numberOfHits(f):
        return sum(e.logLikelihood > -0.01 for e in f)

    # TODO: check that grammar isn't none?
    g = recognitionModel.grammarOfTask(task).untorch()

    frontier = Frontier([], task=task)
    totalProgramCount = 0

    lowerBound = 0

    start = time.time()
    elapsed = 0

    while elapsed < enumerationTimeout:
        # HACK: for now, discard searchTimes
        newFrontiers, searchTimes, programCount = solveForTask_ocaml(
            g=g,

            elapsedTime=elapsed,
            timeout=enumerationTimeout,

            tasks=[task],

            lowerBound=lowerBound,
            upperBound=lowerBound + budgetIncrement,
            budgetIncrement=budgetIncrement,

            maximumFrontiers={task: maximumFrontiers - numberOfHits(frontier)},
            evaluationTimeout=evaluationTimeout,
            testing=testing,
            likelihoodModel=None, # since we're doing ocaml
            CPUs=1,
            hide_stderr=True,
        )

        # Incorporate new solutions
        assert len(newFrontiers) == 1
        frontier = frontier.combine(newFrontiers[task])
        totalProgramCount += programCount

        # Increase bound
        lowerBound += budgetIncrement

        # update this before we check loop condition
        elapsed = time.time() - start

    return dict(frontier=frontier.topK(maximumFrontiers), totalProgramCount=totalProgramCount)


def solve(model_output, tasks, *, CPUs, enumerationTimeout):
    '''
    Finds solutions for several tasks by using a recognition model. Different
    from multicoreEnumeration in that there is no prioritization scheme. This
    routine simply solves tasks at some concurrency (CPUs) and with some
    per-task timeout (enumerationTimeout).
    '''
    from joblib import Parallel, delayed

    rm = model_output['result'].recognitionModel

    results = Parallel(n_jobs=CPUs, verbose=5)(
        delayed(recognitionEnumeration)(rm, task, enumerationTimeout=enumerationTimeout, maximumFrontiers=1)
        for task in tasks
    )

    return dict(results=results, tasks=tasks)
