import os, json, subprocess
import numpy as np
from . import grid

inf = float('inf')

def eval_in_ocaml(program, task):
    # from enumeration.py solveForTask_ocaml()
    def taskMessage(t):
        m = {
            "examples": [{"inputs": list(xs), "output": y} for xs, y in t.examples],
            "name": t.name,
            "request": t.request.json(),
            #"maximumFrontier": maximumFrontiers[t],
            }
        if hasattr(t, "specialTask"):
            special, extra = t.specialTask
            m["specialTask"] = special
            m["extras"] = extra
        return m

    message = dict(
        task=taskMessage(task),
        program=str(program),
    )

    message = json.dumps(message)

    try:
        # Get relative path
        compressor_file = os.path.join(grid.get_root_dir(), 'solvers/_build/default/gridTest.exe')
        process = subprocess.Popen(compressor_file, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        response, error = process.communicate(bytes(message, encoding="utf-8"))
        response = json.loads(response.decode("utf-8"))
    except OSError:
        print(message)
        raise

    return response['logLikelihood']

def eval_in_python(program, task):
    return task.logLikelihood(program)

def check_log_likelihood(program, task, expected_ll):
    for eval_fn in [eval_in_python, eval_in_ocaml]:
        try:
            assert eval_fn(program, task) == expected_ll
        except:
            print('Checking', eval_fn)
            raise


def test_basics():
    start = np.zeros((3, 3))
    location = (1, 0)
    goal = np.array([
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])

    # Testing no action first
    program = grid.parseGrid('()')
    assert np.all(grid.executeGrid(program, grid.GridState(start, location, settings=grid.SETTINGS)).grid == start)
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location), -inf)

    # Now test a solution
    program = grid.parseGrid('((grid_move) (grid_right) (grid_move))')
    assert np.all(grid.executeGrid(program, grid.GridState(start, location, settings=grid.SETTINGS)).grid == goal)
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location), -3)
    # Testing inverse temperature here too
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location, invtemp=5.), -15)

def test_pen():
    start = np.zeros((3, 3))
    location = (1, 1)
    goal = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    program = grid.parseGrid('((grid_move) (grid_right) (grid_dopenup) (grid_move) (grid_right) (grid_dopendown) (grid_move))')
    for s, expected_cost in [
        (grid.Settings(cost_pen_change=False, cost_when_penup=False), -3),
        (grid.Settings(cost_pen_change=True, cost_when_penup=False), -4), # This is a weird case; technically the pendown happens while we're penup.
        (grid.Settings(cost_pen_change=False, cost_when_penup=True), -5),
        (grid.Settings(cost_pen_change=True, cost_when_penup=True), -7),
    ]:
        print(s)
        assert np.all(grid.executeGrid(program, grid.GridState(start, location, settings=s)).grid == goal)
        check_log_likelihood(program, grid.GridTask("test case", start, goal, location, settings=s), expected_cost)

def test_grid_embed():
    start = np.zeros((3, 3))
    location = (2, 0)
    goal = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ])

    program = grid.parseGrid('''(
        (
            grid_embed
            (grid_move)
            (
                grid_embed
                (grid_move)
            )
            (grid_right)
            (grid_move)
        )
        (grid_right)
        (grid_move)
    )''')
    assert np.all(grid.executeGrid(program, grid.GridState(start, location, settings=grid.SETTINGS)).grid == goal)
    # Accumulates reward appropriately for grid_embed
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location), -6)

def test_grid_with_penup():
    start = np.zeros((3, 3))
    location = (2, 0)
    goal = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])

    program = grid.parseGrid('''(
        (grid_move)
        (
            grid_with_penup
            (grid_move)
        )
        (grid_right)
        (grid_move)
    )''')
    final = grid.executeGrid(program, grid.GridState(start, location, settings=grid.SETTINGS))
    assert np.all(final.grid == goal), final
    for s, expected_cost in [
        (grid.Settings(cost_pen_change=False, cost_when_penup=False), -3),
        (grid.Settings(cost_pen_change=True, cost_when_penup=True), -4),
    ]:
        check_log_likelihood(program, grid.GridTask("test case", start, goal, location, settings=s), expected_cost)

def test_grid_explicit_mark():
    start = np.zeros((3, 3))
    location = (2, 0)
    goal = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])

    program = grid.parseGrid('''(
        (grid_move_no_mark)
        (grid_mark_current_location)
        (grid_move_no_mark)
        (grid_right)
        (grid_move_no_mark)
        (grid_mark_current_location)
    )''')
    final = grid.executeGrid(program, grid.GridState(start, location, settings=grid.SETTINGS))
    assert np.all(final.grid == goal), final
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location), -6)

def test_setlocation():
    start = np.zeros((3, 3))

    program = grid.parseGrid('''((grid_setlocation 0 1))''')
    assert grid.executeGrid(program, grid.GridState(start, (2, 0), settings=grid.SETTINGS)) is None, 'Can only setlocation if it is not set'

    for primitive in ['grid_move', 'grid_dopenup', 'grid_dopendown', 'grid_right', 'grid_left']:
        p = grid.parseGrid(f'(({primitive}))')
        assert grid.executeGrid(p, grid.GridState(start, (-1, -1), settings=grid.SETTINGS)) is None, f'No {primitive} before a setlocation'
        check_log_likelihood(p, grid.GridTask("test case", start, start, (-1, -1)), -inf)

    goal1 = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 0],
    ])
    program = grid.parseGrid('''((grid_setlocation 1 2))''')
    assert np.all(grid.executeGrid(program, grid.GridState(start, (-1, -1), settings=grid.SETTINGS)).grid == goal1)

    goal2 = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])
    program = grid.parseGrid('''((grid_setlocation 1 2) (grid_move))''')
    assert np.all(grid.executeGrid(program, grid.GridState(start, (-1, -1), settings=grid.SETTINGS)).grid == goal2)
    check_log_likelihood(program, grid.GridTask("test case", start, goal2, (-1, -1)), -1)

def test_try_all_start():
    start = np.zeros((3, 3))
    goal = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    program = grid.parseGrid('((grid_move) (grid_right) (grid_move))')
    check_log_likelihood(program, grid.GridTask("test case", start, goal, (-1, -1), try_all_start=True), -3)

    # a bit silly
    check_log_likelihood(program, grid.GridTask("test case", start, start, (-1, -1), try_all_start=True), -inf)

def test_partial_progress():
    start = np.zeros((3, 3))
    location = (1, 0)
    goal = np.array([
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ])

    # Testing no action first
    program = grid.parseGrid('()')
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location, partial_progress_weight=1.), -1002)
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location, partial_progress_weight=2.), -1004)

    # Now, one step of progress.
    program = grid.parseGrid('((grid_move))')
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location, partial_progress_weight=1.), -1002)
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location, partial_progress_weight=2.), -1003)

    # Now test a solution. Weight doesn't change value here since we only weight distance from goal.
    program = grid.parseGrid('((grid_move) (grid_right) (grid_move))')
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location, partial_progress_weight=1.), -3)
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location, partial_progress_weight=2.), -3)

    # Now test something invalid
    program = grid.parseGrid('((grid_right) (grid_move))')
    check_log_likelihood(program, grid.GridTask("test case", start, goal, location, partial_progress_weight=1.), -inf)

def test_is_connected_shape():
    assert grid.is_connected_shape(np.array([
        [0, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
    ]))
    assert not grid.is_connected_shape(np.array([
        [0, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 1],
    ]))

def test_make_board_set_contains():
    bsc = grid._make_board_set_contains(np.array([
        [
            [0, 1],
            [0, 0],
        ],
    ]))
    assert bsc(np.array([
        [0, 1],
        [0, 0],
    ]))
    # some random cases
    assert not bsc(np.array([
        [0, 0],
        [1, 0],
    ]))
    assert not bsc(np.array([
        [0, 0],
        [0, 0],
    ]))
    assert not bsc(np.array([
        [1, 1],
        [1, 1],
    ]))
