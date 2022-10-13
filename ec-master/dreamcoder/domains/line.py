from dreamcoder.task import *
from dreamcoder.program import *
from dreamcoder.dreamcoder import *
from dreamcoder.utilities import *



class SupervisedLine(Task):
    def __init__(self, name, start, goal, mustTrain=False):
        self.start = start
        self.goal = goal
        super(SupervisedLine, self).__init__(name, arrow(tline_cont,tline_cont), [],
                                              features=[])
        self.specialTask = ("SupervisedLine",
                            {"start": self.start, "goal": self.goal})
        self.mustTrain = mustTrain

    def logLikelihood(self, e, timeout=None):
        yh = executeTower(e, LineState(pos=self.start), timeout=timeout)
        if yh is not None and yh.pos == self.goal: return 0.
        return NEGATIVEINFINITY

def parseTower(s):
    from sexpdata import loads, Symbol
    s = loads(s)
    def command(k, environment, continuation):
        assert isinstance(k,list)
        if k[0] in (Symbol("line_right"), Symbol("line_left")):
            return Application(Application(Program.parse(k[0].value()), expression(k[1],environment)),continuation)
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



class LineState:
    def __init__(self, pos=0):
        self.pos = pos
    def __str__(self): return f"S(pos={self.pos})"
    def __repr__(self): return str(self)
    def left(self, n):
        return LineState(pos = self.pos - n)
    def right(self, n):
        return LineState(pos = self.pos + n)


def _left(d):
    return lambda k: lambda s: k(s.left(d))
def _right(d):
    return lambda k: lambda s: k(s.right(d))

tline_cont = baseType("line_cont")
primitives = [
    Primitive("line_left", arrow(tint, tline_cont, tline_cont), _left),
    Primitive("line_right", arrow(tint, tline_cont, tline_cont), _right)
] + [
    Primitive(str(j), tint, j) for j in range(1,9)
]

def executeTower(p, state, *, timeout=None):
    try:
        identity = lambda x: x
        return runWithTimeout(lambda : p.evaluate([])(identity)(state),
                              timeout=timeout)
    except RunWithTimeout: return None
    except: return None

if __name__ == '__main__':
    # this is just making sure this is all wired up.
    assert executeTower(parseTower('((line_left 4) (line_right 2))'), LineState(pos=1)).pos == -1
    assert SupervisedLine("lol", 1, 0).logLikelihood(parseTower('((line_left 1))')) == 0

    train = [
        SupervisedLine("lol1", 1, 0),
        SupervisedLine("lol2", 2, 0),
        SupervisedLine("lol3", 3, 0),
    ]
    test = [
        SupervisedLine("loltest1", 1, 0),
        SupervisedLine("loltest2", 2, 0),
        SupervisedLine("loltest3", 3, 0),
    ]

    g0 = Grammar.uniform(primitives, continuationType=tline_cont)
    arguments = commandlineArguments(
        #iterations=1,
        #enumerationTimeout=1,
        #maximumFrontier=10,
        enumerationTimeout=10, activation='tanh',
        iterations=3, recognitionTimeout=3600,
        a=3, maximumFrontier=3, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
                       solver='ocaml',
        CPUs=1)
    generator = ecIterator(g0, train,
                           testingTasks=test,
                           **arguments)
    for result in generator:
        print('hi', result)
