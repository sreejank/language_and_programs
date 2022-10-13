from dreamcoder.task import *
from dreamcoder.program import *
from dreamcoder.dreamcoder import *
from dreamcoder.utilities import *
import pickle, os
import sys
import joblib
from collections import namedtuple
from dreamcoder.utilities import numberOfCPUs
import functools
import tempfile

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("gibbs-500-try_all_start_fix")

import torch.nn as nn
import torch.nn.functional as F
from dreamcoder.recognition import variable

Settings = namedtuple('Settings', ['cost_pen_change', 'cost_when_penup'])
SETTINGS = Settings(cost_pen_change=False, cost_when_penup=False)

def is_connected_shape(grid):
    active = set(zip(*np.where(grid==1)))

    visited = set()
    q = [min(active)] # Arbitrarily pick a place to start
    while q:
        x, y = s = q.pop(0)
        visited.add(s)
        for dx, dy in [
            (-1, 0),
            (+1, 0),
            (0, -1),
            (0, +1),
        ]:
            nx = x + dx
            ny = y + dy
            ns = nx, ny
            if ns in active and ns not in visited and ns not in q:
                q.append(ns)

    return len(visited) == len(active)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        return x.view(x.size(0),-1)

class GridCNN(nn.Module):
    fixed_location = (1, 1)
    newGridTask = lambda *args, **kwargs: GridTask(*args, **kwargs)

    def __init__(self,tasks,testingTasks=[],cuda=False):
        super(GridCNN,self).__init__()
        self.CUDA=cuda
        self.recomputeTasks=True

        self.conv1=nn.Conv2d(1,16,3,stride=1)
        self.outputDimensionality=64
        self.fc1=nn.Linear(64,self.outputDimensionality)

        if cuda:
            self.CUDA=True
            self.cuda()

    def forward(self,v):
        assert v.shape[0]==4
        if len(v.shape)==2:
            v=np.expand_dims(v,(0,1))
            inserted_batch=True
        elif len(v.shape)==3:
            v=np.expand_dims(v,0)
            inserted_batch=True

        v=variable(v,cuda=self.CUDA).float()
        v=F.relu(self.conv1(v))
        v=torch.reshape(v,(-1,64))
        v=F.relu(self.fc1(v))

        if inserted_batch:
            return v.view(-1)
        else:
            return v
    def featuresOfTask(self, t):  # Take a task and returns [features]
        assert t.goal.shape==(4,4)
        return self(t.goal)
    def featuresOfTasks(self, ts):  # Take a task and returns [features]
        """Takes the goal first; optionally also takes the current state second"""
        assert all(t.goal.shape==(4,4) for t in ts)
        return self(np.array([t.goal for t in ts]))
    def taskOfProgram(self,p,t):
        # Excluding trivial programs so we don't get confused about samples
        if str(p) == '(lambda $0)':
            return None
        start = np.zeros((4,4))
        if GridCNN.fixed_location != (-1, -1):
            start[GridCNN.fixed_location] = 1
        start_state=GridState(start, GridCNN.fixed_location, settings=SETTINGS)
        p1=executeGrid(p, start_state)
        if p1 is None:
            print(f'non-trivial program had an execution error: {p}')
            return None
        assert p1.grid.shape == start_state.grid.shape
        t=GridCNN.newGridTask("grid dream",start=start_state.grid,goal=p1.grid,location=start_state.location)
        return t





class GridException(Exception):
    pass

currdir = os.path.abspath(os.path.dirname(__file__))

def tasks_from_grammar_boards(newGridTask):
    with open(f'{currdir}/tasks/grammar_boards.pkl', 'rb') as f:
        boards = pickle.load(f)

    for idx, (board, steps) in enumerate(boards.items()):
         board = np.asarray(board).reshape((4, 4))
         start = steps[0]
         loc = next(zip(*np.where(start)))
         yield newGridTask(f'grammar_boards.pkl_{idx}', start=start, goal=board, location=loc)

def _make_board_set_contains(boards):
    def board_key(board):
        assert len(board.shape) == 2
        return tuple(map(tuple, board))

    board_set = {board_key(board) for board in boards}

    return lambda board: board_key(board) in board_set

def tasks_people_gibbs(newGridTask, *, disconnected=False, fn=f'{currdir}/tasks/people_sampled_boards.npy', exclude_fn=None):
    boards = np.load(fn)
    if exclude_fn is not None:
        exclude_set_contains = _make_board_set_contains(np.load(exclude_fn))
    for idx, board in enumerate(boards):
        if exclude_fn is not None:
            if exclude_set_contains(board):
                continue
        start = np.zeros(boards.shape[1:])
        location = list(zip(*np.where(board)))[0] # arbitrarily pick a start spot
        start[location] = 1
        t = newGridTask(
            f'{os.path.basename(fn)}_{idx}',
            start=start, goal=board, location=location)
        if disconnected and t.is_connected():
            continue
        yield t

def tasks_people_gibbs_500(*args, **kwargs):
    yield from tasks_people_gibbs(*args, **kwargs, fn=f'{currdir}/tasks/500_gsp_samples.npy', exclude_fn=f'{currdir}/tasks/gsp_4x4_sample.npy')

def tree_tasks(newGridTask):
    st = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    loc = (2, 0)
    st[loc] = 1
    return [
        newGridTask(f'left', start=st, location=loc, goal=np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])),
        newGridTask(f'right', start=st, location=loc, goal=np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])),
        newGridTask(f'both', start=st, location=loc, goal=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])),
        newGridTask(f'both-leftboth', start=st, location=loc, goal=np.array([[1, 0, 0], [1, 1, 0], [1, 1, 0]])),
        newGridTask(f'both-rightboth', start=st, location=loc, goal=np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1]])),
        newGridTask(f'both-rightboth-leftboth', start=st, location=loc, goal=np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])),
    ]

def discon_tasks(newGridTask, *, curriculum=True):
    st = np.zeros((4, 4))

    ts = []

    def add_tasks(*locs):
        ts.append(np.copy(st))
        for loc in locs:
            ts[-1][:, loc] = 1

        ts.append(np.copy(st))
        for loc in locs:
            ts[-1][loc, :] = 1

    if curriculum:
        # Making a really simple shape that's disconnected, but also 4x4 for recognition (instead of the 3x1 this used to be)
        g = np.copy(st)
        g[0, 0] = 1
        g[2, 0] = 1
        ts.append(g)

        # And a shape with just one column; this functions as a curriculum for non-ppw
        add_tasks(0)
    add_tasks(0, 2)
    add_tasks(0, 3)
    add_tasks(1, 3)

    return [
        newGridTask(f'discon_{i}', start=st, location=(-1, -1), goal=t)
        for i, t in enumerate(ts)
    ]


class GridState:
    def __init__(self, start, location, *, orientation=0, pendown=True, history=None, reward=0., settings=None):
        self.grid = start
        self.location = location
        self.orientation = orientation % 4 # [0, 1, 2, 3]
        self.pendown = pendown
        self.reward = reward
        self.settings = settings

        # HACK ordering of next statements is intentional to avoid saving `history` into the history.
        if history is not None:
            state_copy = dict(self.__dict__)
            assert 'history' not in state_copy, 'History should not include the `history` key.'
            history += [state_copy]
        self.history = history
    def next_state(self, **kwargs):
        a = dict(self.__dict__, **kwargs)
        if a.pop('step_cost', False):
            cost = 1
            if not self.pendown and not self.settings.cost_when_penup:
                cost = 0
            a['reward'] -= cost
        return type(self)(a.pop('grid'), a.pop('location'), **a)
    def left(self):
        self._ensure_location()
        return self.next_state(orientation=self.orientation - 1, step_cost=True)
    def right(self):
        self._ensure_location()
        return self.next_state(orientation=self.orientation + 1, step_cost=True)

    def move(self):
        # HACK: we don't compose `move_no_mark` here b/c it would add an extra step cost.
        return self.next_state(location=self._move_location()).mark_current_location()

    def move_no_mark(self):
        return self.next_state(location=self._move_location(), step_cost=True)

    def _move_location(self):
        self._ensure_location()
        dx, dy = [
            (-1, 0), # up
            (0, +1), # right
            (+1, 0), # down
            (0, -1), # left
        ][self.orientation]
        xlim, ylim = self.grid.shape
        prevx, prevy = self.location
        x = prevx + dx
        if not (0 <= x < xlim):
            x = prevx
        y = prevy + dy
        if not (0 <= y < ylim):
            y = prevy
        return x, y

    def mark_current_location(self):
        self._ensure_location()
        grid = self.grid
        if self.pendown:
            grid = np.copy(grid)
            grid[self.location] = 1
        return self.next_state(grid=grid, step_cost=True)

    def dopendown(self):
        self._ensure_location()
        return self.next_state(pendown=True, step_cost=self.settings.cost_pen_change)
    def dopenup(self):
        self._ensure_location()
        return self.next_state(pendown=False, step_cost=self.settings.cost_pen_change)
    def setlocation(self, location):
        if self.location != (-1, -1):
            raise GridException('Location can only be set when unspecified.')
        grid = self.grid
        valid = 0 <= location[0] < grid.shape[0] and 0 <= location[1] < grid.shape[1]
        if self.pendown:
            if not valid:
                raise GridException(f'Invalid location {location} for grid with shape {grid.shape}')
            grid = np.copy(grid)
            grid[location] = 1
        return self.next_state(grid=grid, location=location)

    def _ensure_location(self):
        if self.location == (-1, -1):
            raise GridException('Location must be set')
    def __repr__(self):
        return f'GridState({self.grid}, {self.location}, orientation={self.orientation}, pendown={self.pendown}, settings={self.settings})'


class GridTask(Task):
    incorrect_penalty = -1000

    def __init__(self, name, start, goal, location, *, invtemp=1., partial_progress_weight=0., try_all_start=False, settings=SETTINGS):
        assert start.shape == goal.shape
        assert location == (-1, -1) or (0 <= location[0] < start.shape[0] and 0 <= location[1] < start.shape[1])
        if try_all_start:
            if location != (-1, -1):
                assert np.sum(start) == 1 and start[location] == 1, 'We make sure the tasks are simple, with only with marked location at the start location.'
                location = (-1, -1)
                start = np.zeros(start.shape)
        if location == (-1, -1):
            assert np.sum(start) == 0
        self.start = start
        self.goal = goal
        self.location = location
        self.invtemp = invtemp
        self.partial_progress_weight = partial_progress_weight
        self.settings = settings
        self.try_all_start = try_all_start
        super().__init__(name, arrow(tgrid_cont,tgrid_cont), [], features=[])

    @property
    def specialTask(self):
        # Computing this dynamically since we modify the task when there's the option to set location.
        return ("GridTask", {
            "start": self.start.astype(bool).tolist(), "goal": self.goal.astype(bool).tolist(),
            "location": tuple(map(int, self.location)),
            "invtemp": self.invtemp,
            "partial_progress_weight": self.partial_progress_weight,
            "try_all_start": self.try_all_start,
            "settings": self.settings._asdict(),
            "log_program": False,
        })

    def _score_from_location(self, e, state, *, timeout=None):
        yh = executeGrid(e, state, timeout=timeout)

        if yh is None:
            return NEGATIVEINFINITY

        correct = np.all(yh.grid == self.goal)

        if self.partial_progress_weight != 0:
            '''
            num_incorrect = (yh.grid != self.goal).sum()
            return (
                (0 if correct else -1000)
                + self.invtemp * yh.reward
                - self.partial_progress_weight * num_incorrect)
            '''
            final = yh.grid.astype(bool)
            goal = self.goal.astype(bool)
            if np.any(final & ~goal):
                return NEGATIVEINFINITY
            num_not_done = (~final & goal).sum()
            return (
                (0 if correct else GridTask.incorrect_penalty)
                + self.invtemp * yh.reward
                - self.partial_progress_weight * num_not_done)

        if correct:
            return self.invtemp * yh.reward
        return NEGATIVEINFINITY

    def _score_for_all_locations(self, e, timeout=None):
        return [
            (
                self._score_from_location(e, GridState(self.start, (-1, -1), settings=self.settings).setlocation((x, y)), timeout=timeout),
                x,
                y,
            )
            for x in range(self.start.shape[0])
            for y in range(self.start.shape[1])
        ]

    def logLikelihood(self, e, timeout=None):
        if self.try_all_start:
            score, x, y = max(self._score_for_all_locations(e, timeout=timeout))
            return score
        else:
            return self._score_from_location(e, GridState(self.start, self.location, settings=self.settings), timeout=timeout)

    def is_connected(self):
        return is_connected_shape(self.goal)

def parseGrid(s):
    from sexpdata import loads, Symbol
    s = loads(s)
    def command(k, environment, continuation):
        assert isinstance(k,list)
        if k[0] in (
            Symbol("grid_right"), Symbol("grid_left"),
            Symbol("grid_move"),
            Symbol("grid_move_no_mark"),
            Symbol("grid_mark_current_location"),
            Symbol("grid_dopenup"), Symbol("grid_dopendown"),
        ):
            assert len(k) == 1
            return Application(Program.parse(k[0].value()),continuation)
        if k[0] in (
            Symbol('grid_setlocation'),
        ):
            assert len(k) == 3
            return Application(
                Application(
                    Application(Program.parse(k[0].value()), expression(k[1], environment)),
                    expression(k[2], environment)),
            continuation)
        if k[0] in (
            Symbol("grid_embed"),
            Symbol("grid_with_penup"),
        ):
            # TODO issues with incorrect continuations probably need to be dealt with here
            # I think the issue is that we hardcode Index(0)?
            body = block(k[1:], [None] + environment, Index(0))
            return Application(Application(Program.parse(k[0].value()),Abstraction(body)),continuation)
        assert False

    def expression(e, environment):
        for n, v in enumerate(environment):
            if e == v: return Index(n)
        if isinstance(e,int): return Program.parse(str(e))

        assert isinstance(e,list)
        if e[0] == Symbol('+'): return Application(Application(_addition, expression(e[1], environment)),
                                                   expression(e[2], environment))
        if e[0] == Symbol('-'): return Application(Application(_subtraction, expression(e[1], environment)),
                                                   expression(e[2], environment))
        assert False

    def block(b, environment, continuation):
        if len(b) == 0: return continuation
        return command(b[0], environment, block(b[1:], environment, continuation))

    return Abstraction(block(s, [], Index(0)))
    #try: return Abstraction(command(s, [], Index(0)))
    #except: return Abstraction(block(s, [], Index(0)))



def _grid_left(k): return lambda s: k(s.left())
def _grid_right(k): return lambda s: k(s.right())
def _grid_move(k): return lambda s: k(s.move())
def _grid_dopendown(k): return lambda s: k(s.dopendown())
def _grid_dopenup(k): return lambda s: k(s.dopenup())
def _grid_move_no_mark(k): return lambda s: k(s.move_no_mark())
def _grid_mark_current_location(k): return lambda s: k(s.mark_current_location())
def _grid_setlocation(x): return lambda y: lambda k: lambda s: k(s.setlocation((x, y)))

def _grid_embed(body):
    def f(k):
        def g(s):
            s._ensure_location()

            identity = lambda x: x
            # TODO: use of identity here feels a bit heuristic; it's what tower's impl does, but it seems
            # to let misuse of the continuation happen in program induction (use of $0 and $1 in an embed
            # result in same value, but $1 should be incorrect & terminate program?)
            ns = body(identity)(s)
            # We keep the grid & reward state, but restore the agent state
            ns = s.next_state(grid=ns.grid, reward=ns.reward)
            return k(ns)
        return g
    return f

def _grid_with_penup(body):
    def f(k):
        def g(s):
            # HACK: we explicitly do next_state instead of dopen* to avoid the step cost.
            identity = lambda x: x # HACK: see above note in grid_embed impl
            # We first run body without pen
            ns = body(identity)(s.next_state(pendown=False))
            # Then bring pen back
            return k(ns.next_state(pendown=True))
        return g
    return f

# TODO still not clear to me what types are doing in Python; how is this bound? Does it require definition in ocaml?
tgrid_cont = baseType("grid_cont")
CONTINUATION_TYPE = arrow(tgrid_cont, tgrid_cont)
primitives_base = [
    Primitive("grid_left", arrow(tgrid_cont, tgrid_cont), _grid_left),
    Primitive("grid_right", arrow(tgrid_cont, tgrid_cont), _grid_right),
    Primitive("grid_move", arrow(tgrid_cont, tgrid_cont), _grid_move),
    Primitive("grid_embed", arrow(arrow(tgrid_cont, tgrid_cont), tgrid_cont, tgrid_cont), _grid_embed),
]

primitives_pen = primitives_base + [
    Primitive("grid_dopendown", arrow(tgrid_cont, tgrid_cont), _grid_dopendown),
    Primitive("grid_dopenup", arrow(tgrid_cont, tgrid_cont), _grid_dopenup),
]

primitives_penctx = primitives_base + [
    Primitive("grid_with_penup", arrow(arrow(tgrid_cont, tgrid_cont), tgrid_cont, tgrid_cont), _grid_with_penup),
]

primitives_numbers_only = [
    Primitive(str(j), tint, j) for j in range(0,4) # HACK this needs to be based on grid size
]

primitives_loc = primitives_pen + [
    Primitive("grid_setlocation", arrow(tint, tint, tgrid_cont, tgrid_cont), _grid_setlocation),
] + primitives_numbers_only

primitives_explicit_mark = [
    Primitive("grid_left", arrow(tgrid_cont, tgrid_cont), _grid_left),
    Primitive("grid_right", arrow(tgrid_cont, tgrid_cont), _grid_right),
    Primitive("grid_move_no_mark", arrow(tgrid_cont, tgrid_cont), _grid_move_no_mark),
    Primitive("grid_mark_current_location", arrow(tgrid_cont, tgrid_cont), _grid_mark_current_location),
    Primitive("grid_embed", arrow(arrow(tgrid_cont, tgrid_cont), tgrid_cont, tgrid_cont), _grid_embed),
]

def uniform_with_excluded(primitives, excluded, continuationType):
    '''
    ContextualGrammar requires we have all primitives at all times, so we exclude them
    by giving them log probability of negative inf.
    '''
    neginf = -float('inf')
    return Grammar(
        0.0,
        [(neginf if p in excluded else 0.0, p.infer(), p) for p in primitives],
        continuationType=continuationType)

def make_grammar(primitives, continuationType):
    '''
    This function returns a grammar when passed primitives. It handles a special
    case to ensure generation of valid programs that use setlocation. When
    setlocation is used, this function ensures the resulting grammar can only
    use setlocation when there's no parent primitive.
    '''

    setloc = next((p for p in primitives if p.name == 'grid_setlocation'), None)

    # If setloc isn't present, then we just return a uniform grammar
    if setloc is None:
        return Grammar.uniform(primitives, continuationType=continuationType)

    primitives_without_loc = [p for p in primitives if p != setloc]

    g_noloc = uniform_with_excluded(primitives, [setloc], continuationType=continuationType)
    g_loc = uniform_with_excluded(primitives, primitives_without_loc, continuationType=continuationType)

    return ContextualGrammar(
        # If we have no parent, then we need to be a setlocation
        g_loc,
        # If we have a variable as parent, we avoid setloc
        g_noloc,
        # If we have anything else as parent (setloc or other primitives), we avoid setloc
        {e: [g_noloc]*len(e.infer().functionArguments()) for e in primitives})

def executeGrid(p, state, *, timeout=None):
    try:
        identity = lambda x: x
        return runWithTimeout(lambda : p.evaluate([])(identity)(state),
                              timeout=timeout)
    except RunWithTimeout: return None
    except GridException: return None

def parseArgs(parser):
    def boolarg(name, default):
        parser.add_argument(f'--{name}', dest=name, default=default, action='store_true')
        parser.add_argument(f'--no-{name}', dest=name, default=default, action='store_false')
    parser.add_argument(
        "-f",
        dest="DELETE_var",
        help="just adding this here to capture a jupyter notebook variable",
        default='x',
        type=str)
    parser.add_argument("--task", dest="task", default="grammar")
    parser.add_argument("--log_file_path_for_mlflow", dest="log_file_path_for_mlflow", help='This is the file our output is being written to. Python does not configure this, but assumes the command has been run so this is the case. It is uploaded to mlflow.')
    parser.add_argument("--grammar", dest="grammar", default='pen', type=str)
    parser.add_argument("--invtemp", dest="invtemp", default=1., type=float)
    parser.add_argument("--partial_progress_weight", dest="partial_progress_weight", default=0., type=float)
    boolarg('try_all_start', False)

def main():
    arguments = commandlineArguments(
        enumerationTimeout=120,
        solver='ocaml',
        compressor="ocaml",
        activation='tanh',
        iterations=5,
        recognitionTimeout=120,
        # TODO what does this arity do? seems to relate to grammar?
        a=3,
        maximumFrontier=5, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5,
        structurePenalty=1.,
        extras=parseArgs,
        featureExtractor=GridCNN,
        CPUs=numberOfCPUs(),
    )
    del arguments['DELETE_var']

    # Making a copy to add to mlflow
    complete_arguments = dict(arguments)

    # Make sure to add this to mlflow! Helps with debugging crashes.
    log_file_path_for_mlflow = arguments.pop('log_file_path_for_mlflow')
    taskname = arguments.pop('task')
    try_all_start = arguments.pop('try_all_start')
    invtemp = arguments.pop('invtemp')
    partial_progress_weight = arguments.pop('partial_progress_weight')

    grammar = arguments.pop('grammar')
    p = dict(
        no_pen=primitives_base,
        pen=primitives_pen,
        penctx=primitives_penctx,
        pen_setloc=primitives_loc,
        explicit_mark=primitives_explicit_mark,
    )[grammar]

    using_setloc = any(prim.name == 'grid_setlocation' for prim in p)

    newGridTask = lambda *args, **kwargs: GridTask(
        *args,
        try_all_start=try_all_start,
        invtemp=invtemp,
        partial_progress_weight=partial_progress_weight,
        **kwargs)
    # task dist
    train_dict = dict(
        grammar=tasks_from_grammar_boards,
        people_gibbs=tasks_people_gibbs,
        people_gibbs_discon=functools.partial(tasks_people_gibbs, disconnected=True),
        people_gibbs_500=tasks_people_gibbs_500,
        people_gibbs_discon_500=functools.partial(tasks_people_gibbs_500, disconnected=True),
        tree=tree_tasks,
        discon=discon_tasks,
        discon_no_curr=functools.partial(discon_tasks, curriculum=False),
    )
    train = list(train_dict[taskname](newGridTask))
    if using_setloc:
        # make_grammar below will always start a program with setlocation when this is true,
        # so we have to make GridCNN use it too
        # HACK in the future, should we make this be dynamic within taskOfProgram?
        # could walk the program, see if setlocation is used, then choose to do location
        # dynamically?
        GridCNN.fixed_location = (-1, -1)
        for task in train:
            task.start = np.zeros(task.start.shape)
            task.location = (-1, -1)

    # Once the training tasks have been configured, we set the test tasks as well.
    test = train
    # We also make sure recognition makes tasks in the same way.
    GridCNN.newGridTask = newGridTask

    g0 = Grammar.uniform(
        p,
        # when doing grid_cont instead, we only consider $0
        # but when we only have type=tgrid_cont, then we get a nicer library for tree_tasks()
        continuationType=CONTINUATION_TYPE)

    arguments['contextual'] = isinstance(g0, ContextualGrammar)
    generator = ecIterator(g0, train,
                           testingTasks=test,
                           **arguments)
    with mlflow.start_run():
        mlflow.log_params(complete_arguments)
        for iter, result in enumerate(generator):
            # HACK: does this factor in recognition?
            total_hits = 0
            total_discon_hits = 0
            for task in train:
                es = result.frontiersOverTime[task][-1].entries
                if any(e.logLikelihood > GridTask.incorrect_penalty for e in es):
                    total_hits += 1
                    if not task.is_connected():
                        total_discon_hits += 1
                v = max(e.logLikelihood for e in es) if es else -float('inf')
                mlflow.log_metric(key=str(task), value=v, step=iter)
            print(f'Hits-Solutions {total_hits}/{len(train)}')
            # Adding an underscore to have them sort to beginning in UI
            mlflow.log_metric(key="_solved", value=total_hits, step=iter)
            mlflow.log_metric(key="_solved_disconnected", value=total_discon_hits, step=iter)

            print('-' * 100 + '\n' * 5)

            with tempfile.TemporaryDirectory() as tmpdir:
                # We need to log every iteration since the recognition model / hidden state
                # isn't saved from iter to iter.
                joblib.dump(dict(result=result,train=train,arguments=arguments), f'{tmpdir}/output-iter{iter}.bin')
                mlflow.log_artifacts(tmpdir)

            # We do this on every iteration; the file just gets overwritten, but this lets us stay up-to-date.
            if log_file_path_for_mlflow:
                mlflow.log_artifact(log_file_path_for_mlflow)


def try_exhaustive_enumeration():

    # If we're not in exhaustive enumeration mode, then skip.
    if sys.argv[1] != 'enumerate':
        return

    mlflow.set_experiment("exhaustive-enumeration")
    model_output_path = sys.argv[2]

    with mlflow.start_run():
        m = joblib.load(model_output_path)
        mlflow.log_params(dict(arguments=sys.argv))

        from . import exhaustive_enum
        tasks = exhaustive_enum.create_all_grid_tasks()
        result = exhaustive_enum.solve(m, tasks, CPUs=40, enumerationTimeout=30)

        with tempfile.TemporaryDirectory() as tmpdir:
            joblib.dump(dict(
                result,
                model_output_path=model_output_path,
            ), f'{tmpdir}/exhaustive-enum.bin')
            mlflow.log_artifacts(tmpdir)

    sys.exit(0)


if __name__ == '__main__':
    try_exhaustive_enumeration()
    main()
