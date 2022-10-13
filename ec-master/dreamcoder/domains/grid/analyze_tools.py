import os
from grid import *
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
import joblib
import functools
import scipy.cluster.hierarchy

script_dir = os.path.abspath(os.path.dirname(__file__))

@dataclass
class TaskFrontier:
    task: ...
    entries: ...

    def best_program_entry(self):
        return max(self.entries, key=lambda e: e.logPrior + e.logLikelihood)

    def is_solved(self):
        e = self.best_program_entry()
        return (e.logPrior + e.logLikelihood) > GridTask.incorrect_penalty

    def execute_best(self, **kw):
        e = self.best_program_entry()
        score, x, y = max(self.task._score_for_all_locations(e.program))
        s = execute_grid(e.program, self.task.start, (x, y), **kw)
        assert np.all(s.grid == self.task.goal) or score < GridTask.incorrect_penalty, (score, s, e.program)
        return dict(
            state=s,
            location=(x, y),
            score=score,
        )

    @classmethod
    def from_result(cls, result, task, *, iteration=-1):
        es = result.frontiersOverTime[task][iteration].entries
        if es:
            return cls(task, es)

def iter_frontiers(tasks, ecResult, *, iteration=-1):
    for task in tasks:
        tf = TaskFrontier.from_result(ecResult, task)
        if tf:
            yield task, tf

class TracingInvented(Program):
    def __init__(self, stack, invented, original_invented):
        self.stack = stack
        self.invented = invented
        self.original_invented = original_invented
    def evaluate(self, environment):
        '''
        This wrapping implementation only works for extremely simple cases of reuse
        that involve tail calls to the continuation. Cases where the continuation is
        called mid-function will not be appropriately tagged.
        TODO: analyze programs to ensure they only have explicit tail calls to continuation?
        '''
        def wrapped(k):
            returned = False
            def wrappedk(arg):
                nonlocal returned
                assert not returned, 'Assume this is only called once'
                returned = True
                v = self.stack.pop()
                assert v == self.original_invented, 'Assume this is called in a straightforward way'
                return k(arg)

            fn = self.invented.evaluate(environment)(wrappedk)

            def wrapped_state_mapper(s):
                self.stack.append(self.original_invented)
                rv = fn(s)
                assert returned, 'Should have been called by now'
                return rv

            return wrapped_state_mapper
        return wrapped
    def show(self, isFunction): return f"Trace({self.invented.show(False)})"
    def inferType(self, *a, **k): return self.invented.inferType(*a, **k)

class StackTracingRewrite(object):
    def __init__(self, stack, continuationType=CONTINUATION_TYPE):
        self.stack = stack
        self.continuationType = continuationType
    def invented(self, e):
        assert e.tp == self.continuationType, 'Only analyzing simple inventions for now'
        return TracingInvented(self.stack, Invented(e.body.visit(self)), e)
    def primitive(self, e): return e
    def index(self, e): return e
    def application(self, e): return Application(e.f.visit(self), e.x.visit(self))
    def abstraction(self, e): return Abstraction(e.body.visit(self))

def execute_grid(p, start, location, *, trace=False, set_start_with_setlocation=True):
    S = SETTINGS
    cls = GridState
    if trace:
        # This is a complete hack that's just meant to disambiguate
        # stack entries that would otherwise be identical (repeated calls to a routine)
        class ListWithCounter(list):
            def __init__(self):
                super().__init__()
                self.i = 0
            def append(self, x):
                self.i += 1
                return super().append((self.i, x))
            def pop(self):
                i, x = super().pop()
                return x
        stack = ListWithCounter()
        p = p.visit(StackTracingRewrite(stack))
        class TracingGridState(GridState):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                if self.history is not None:
                    self.history[-1]['stack'] = list(stack)
        cls = TracingGridState

    if set_start_with_setlocation:
        s = cls(start, (-1, -1), history=[], settings=S).setlocation(location)
    else:
        s = cls(start, location, history=[], settings=S)
    return executeGrid(p, s)

def execute_grid_tight_layout(program, *, program_start_shape=(7, 7), **kwargs):
    w, h = program_start_shape
    assert w == h
    mid = w//2

    # We first execute the program on an oversize grid -- we use the places
    # it marks to define how we set the start location to ensure tighter margins.
    start = np.zeros(program_start_shape)
    location = (mid, mid)
    s = execute_grid(
        program,
        start,
        location,
        set_start_with_setlocation=False,
    )
    # HACK ending early b/c below is wrong
    s.__start = start
    return s

    # Now we check to see which quadrant has marks from program execution.
    # We exclude quadrant borders.
    assert False, 'incorrect b/c it ignores middle cross (0:7,3 and 3,0:7)'
    half_first = slice(0, mid)
    half_second = slice(mid+1, w)
    assert half_first.stop - half_first.start == half_second.stop - half_second.start == mid
    has_marks = [
        np.any(s.grid[half_first][None, half_first]),
        np.any(s.grid[half_first][None, half_second]),
        np.any(s.grid[half_second][None, half_first]),
        np.any(s.grid[half_second][None, half_second]),
    ]
    if np.sum(has_marks) not in (0, 1):
        warnings.warn('The supplied program exceeds expected bounds -- it draws in multiple quadrants when centered in an oversize grid.')
        s.__start = start
        return s

    # Figure out ideal start location in a smaller grid
    mark_location = np.where(has_marks)[0]
    location = [
        (mid, mid),
        (mid, 0),
        (0, mid),
        (0, 0),
    ][mark_location[0] if len(mark_location) else 0]

    # Now execute on smaller grid and return result.
    s = mid + 1
    start = np.zeros((s, s))
    final_s = execute_grid(
        program,
        start,
        location,
        **kwargs,
    )
    final_s.__start = start
    return final_s

class InventionCounting(object):
    def __init__(self, *, recurse_into_invented=True):
        self.counts = {}
        self.recurse_into_invented = recurse_into_invented
    def invented(self, e):
        self.counts.setdefault(e, 0)
        self.counts[e] += 1
        if self.recurse_into_invented:
            e.body.visit(self)
    def primitive(self, e): pass
    def index(self, e): pass
    def application(self, e):
        e.f.visit(self)
        e.x.visit(self)
    def abstraction(self, e):
        e.body.visit(self)

def invention_counts(p, recurse_into_invented=True):
    v = InventionCounting(recurse_into_invented=recurse_into_invented)
    p.visit(v)
    return v.counts

class InventionUseContextCounting(object):
    def __init__(self):
        self.counts = {}
        self.currinv = None
    def invented(self, e):
        key = (self.currinv, e)
        self.counts.setdefault(key, 0)
        self.counts[key] += 1

        prev = self.currinv
        self.currinv = e
        e.body.visit(self)
        assert self.currinv == e, 'making sure it was reverted recursively'
        self.currinv = prev
    def primitive(self, e): pass
    def index(self, e): pass
    def application(self, e):
        e.f.visit(self)
        e.x.visit(self)
    def abstraction(self, e):
        e.body.visit(self)






# ---
# Graphics!
# ---

def generate_grid_rect(grid, facecolor="none", edgecolor="k", linewidth=0.5):
    rv = []
    for s in np.ndindex(grid.shape):
        patch = plt.Rectangle((s[0]-0.5, s[1]-0.5), 1, 1, linewidth=linewidth, facecolor=facecolor, edgecolor=edgecolor)
        # ?? making sure the reactangle is within the bounds: https://stackoverflow.com/a/60577729 patch.set_clip_path(patch)
        rv.append((s, patch))
    return rv

def get_ax_size(ax):
    fig = ax.figure
    # from stackoverflow
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width *= fig.dpi
    height *= fig.dpi
    return width, height

def plot(
    start=None, end=None, history=None,
    *,
    size=3, ax=None, trace=False, prog_to_coloridx=None,
    program=None, program_start_shape=(7, 7), program_start_location=None,
    style_plain=True,
):
    if isinstance(start, TaskFrontier):
        tf = start
        start, end, history = tf.task.start, tf.task.goal, tf.execute_best(trace=trace)['state'].history

    if program is not None:
        #location = program_start_location or tuple(coord//2 for coord in program_start_shape)
        s = execute_grid_tight_layout(
            program,
            program_start_shape=program_start_shape,
            trace=trace,
            set_start_with_setlocation=False,
        )
        start = s.__start
        end = s.grid
        history = s.history

    w, h = start.shape
    if ax is None:
        aspect_ratio = w/h
        _, ax = plt.subplots(figsize=(size*aspect_ratio, size))

    ax_w_pix, ax_h_pix = get_ax_size(ax)
    scale = .9 * min(ax_w_pix/w, ax_h_pix/h)

    ax.set(
        xticks=[],
        yticks=[],
        xlim=[-1/2, w-1/2],
        ylim=[-1/2, h-1/2],
    )
    for xy, r in generate_grid_rect(start):
        if start[xy]:
            r.set_facecolor('blue')
        elif end[xy]:
            r.set_facecolor((.8, .8, .8))
        ax.add_artist(r)

    if history and not style_plain:
        prog_to_idx = {} if prog_to_coloridx is None else prog_to_coloridx
        cmap = plt.get_cmap('tab20')

        try:
            maxdepth = max(len(curr['stack']) for curr in history if 'stack' in curr)
        except ValueError:
            maxdepth = 0

        for previ, (prev, curr) in enumerate(zip(history[:-1], history[1:])):
            curri = previ + 1
            if prev['location'] == (-1, -1):
                ax.scatter(*curr['location'], marker='*', color='red')
                continue

            xs = [prev['location'][0], curr['location'][0]]
            ys = [prev['location'][1], curr['location'][1]]

            if 'stack' not in curr:
                ax.plot(xs, ys, c='k')
                continue

            prev_stack = prev['stack']
            curr_stack = curr['stack']
            #ax.plot(xs, ys, c='k', lw=scale*(maxdepth+1) / maxdepth)
            ax.plot(xs, ys, c='k', lw=scale)

            for i, (currid, curr_invented) in enumerate(curr['stack']):
                xs_, ys_ = xs, ys
                if curr_stack[:i+1] != prev_stack[:i+1]: # comparing full stack to see if currid has changed too
                    alpha = 0.25
                    xs_ = [xs[0] * (1-alpha) + xs[1] * alpha, xs[1]]
                    ys_ = [ys[0] * (1-alpha) + ys[1] * alpha, ys[1]]

                lw = maxdepth - i # this maps i=0 to maxdepth and i=maxdepth-1 to 1
                #lw = lw / maxdepth * scale # divide to force it to (0, 1], then scale to some value
                # Adding 1 here to make sure we're including the root process
                lw = lw / (maxdepth + 1) * scale # divide to force it to (0, 1], then scale to some value
                # give this program an index (which we map to a color)
                if curr_invented not in prog_to_idx:
                    prog_to_idx[curr_invented] = len(prog_to_idx)
                #print(prog_to_idx, 'make sure same invented go to same color')
                c = cmap.colors[prog_to_idx[curr_invented] % len(cmap.colors)]
                ax.plot(xs_, ys_, c=c, lw=lw, zorder=i+3) # 3+ seems necessary to get above default z values

def plot_trace(start, history=None, *, animate=False, size=None):
    if isinstance(start, TaskFrontier):
        tf = start
        start, history = tf.task.start, tf.execute_best(trace=True)['state'].history

    def render_step(ax, i):
        h_so_far = history[:i+1]
        assert len(h_so_far) == i+1
        last_state = h_so_far[-1]
        plot(start, last_state['grid'], h_so_far, ax=ax)
        #m = ['^', '>', 'v', '<'] # actually is rotate clockwise 90deg from this
        m = ['<', '^', '>', 'v'][last_state['orientation']]
        ax.scatter(*last_state['location'], c='r', zorder=2, marker=m, s=400/start.shape[0])

    w, h = start.shape
    aspect_ratio = w/h

    if animate:
        size = size or 2
        _, ax = plt.subplots(figsize=(size*aspect_ratio, size))
        return simple_animation(len(history), render_step, interval=100, ax=ax)
    else:
        size = size or 1
        f, axes = plt.subplots(1, len(history), figsize=(size * aspect_ratio * len(history), size))
        for i, ax in enumerate(axes):
            render_step(ax, i)

def plot_to_buffer(*args, figure_kwargs={}, **kwargs):
    f = plt.figure(**figure_kwargs)
    # Important to get rid of whitespace around axis -- otherwise we have to use
    # tight_layout in savefig, but that means the figure bbox doesn't directly
    # determine image size, causing shape issues.
    ax = f.add_axes([0.,0.,1.,1.])
    plot(*args, ax=ax, **kwargs)
    im = figure_to_image_buffer(f)
    plt.close(f)
    return im

def figure_to_image_buffer(fig):
    import io
    io_buf = io.BytesIO()
    try:
        fig.savefig(io_buf, format='raw')
        io_buf.seek(0)
        return np.reshape(
            np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    finally:
        io_buf.close()

def simple_animation(n, fn, *, ax=None, filename=None, interval=100):
    '''
    This is a generic routine to make simple animations; it entirely wipes out & re-renders at every time step.
    Parameters:
    - n - number of frames.
    - fn(ax, i) - rendering callback, must take axis and rendering iteration as arguments.
    '''
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.animation as animation

    if ax is None: _, ax = plt.subplots()
    f = ax.figure

    fn(ax, 0) # Running this once to make size

    def update(t):
        for a in ax.lines + ax.collections:
            a.remove()
        fn(ax, t)
        return []

    a = FuncAnimation(
        f, update, frames=n, interval=interval, blit=True, repeat=False)
    plt.close()

    if filename is not None:
        assert filename.endswith('.gif') or filename.endswith('.mp4'), 'Only supports exporting to .gif or .mp4'
        if filename.endswith('.mp4'):
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=1000./interval, bitrate=1800)
            a.save(filename, writer=writer)
            from IPython.display import Video
            return Video(filename)
        else:
            a.save(filename, writer='imagemagick')
            from IPython.display import Image
            return Image(filename)

    from IPython.display import HTML
    return HTML(a.to_html5_video())

def savefig(fn, *, close=True, **kwargs):
    try:
        os.makedirs(os.path.dirname(fn))
    except:
        pass
    plt.savefig(fn, bbox_inches='tight', **kwargs)
    if close:
        plt.close(plt.gcf())


class TasksPlotter:
    '''
    Initially designed to render tasks in an embedded space, but also has routines
    for analysis of embeddings (like plot_explained_variance or plot_rdm).
    '''
    def __init__(self, tasks, embedding, *, explained_variance=None):
        assert embedding.shape == (len(tasks.train), 2)
        self.tasks = tasks
        self.embedding = embedding
        self.explained_variance = explained_variance

    def plot_grid_at_point(self, ax, grid, point, scale_args=(1/10,)):
        import matplotlib
        w, h = grid.shape
        pc = []
        for xy, rect in generate_grid_rect(grid):
            if grid[xy]:
                rect.set_facecolor((.5,)*3)
            else:
                rect.set_facecolor((1,)*3)
            pc.append(rect)
        pc = matplotlib.collections.PatchCollection(pc, match_original=True)
        t = (
            matplotlib.transforms.Affine2D()
            # First, center over zero
            .translate(1/2-w/2, 1/2-h/2)
            # This rescales both extents to [-.5, .5]
            .scale(1/w, 1/h)
            .scale(*scale_args)
            .translate(*point)
            + ax.transData
        )
        pc.set_transform(t)
        ax.add_artist(pc)

    def plot_task_point_pairs(self, pairs, srel=.03, show_trace=False):
        f, ax = plt.subplots(figsize=(16, 16))

        def scale_ax(points, *, padding=.1):
            min_, max_ = min(points), max(points)
            rng = max_ - min_
            pd = padding * rng
            return [min_-pd, max_+pd*2]

        points = [point for _, point, _ in pairs]
        ax.set(
            xlim=scale_ax([x for x, y in points]),
            ylim=scale_ax([y for x, y in points]),
        )

        # close but need to factor in figure too
        ax_w = ax.get_xlim()[1] - ax.get_xlim()[0]
        ax_h = ax.get_ylim()[1] - ax.get_ylim()[0]
        aspect_ratio = ax_w / ax_h

        prog_to_coloridx = self.tasks.default_prog_to_coloridx()
        for t, point, tf in pairs:
            if show_trace:
                from matplotlib.offsetbox import OffsetImage, AnnotationBbox
                i = plot_to_buffer(tf, figure_kwargs=dict(figsize=(.75, .75)), prog_to_coloridx=prog_to_coloridx)
                img = OffsetImage(i, zoom=0.5)
                ab = AnnotationBbox(img, point, xycoords='data', frameon=False)
                ax.add_artist(ab)
            else:
                self.plot_grid_at_point(ax, t.goal, point, scale_args=(ax_w * srel, ax_h * srel))

    def plot_explained_variance(self, plot_arg='-o', *plot_args, cumulative=False, **plot_kwargs):
        assert self.explained_variance is not None
        if cumulative:
            v = np.append([0], np.cumsum(self.explained_variance))
        else:
            v = self.explained_variance
        plt.plot(v, plot_arg, *plot_args, **plot_kwargs)

    def plot(self, *, samples=50, seed=None, **kwargs):
        import random
        idxs = range(len(self.tasks.train))
        if samples is not None:
            idxs = random.Random(seed).sample(idxs, k=samples)

        fs = list(self.tasks.iter_frontiers())
        self.plot_task_point_pairs([
            (fs[i][0], self.embedding[i], fs[i][1])
            for i in idxs
        ], **kwargs)

    @classmethod
    def plot_rdm(cls, data, *, linkage_input=None):
        D = nancorrcoef(data)

        kw = dict()
        if linkage_input is not None:
            linkage = scipy.cluster.hierarchy.linkage(
                nancorrcoef(linkage_input), method='average', metric='euclidean')
            kw = dict(kw, row_linkage=linkage, col_linkage=linkage)
        sns.clustermap(
            D,
            cmap='RdBu', vmin=-1, vmax=+1,
            **kw,
        )

    @classmethod
    def svd(cls, tasks, *, data=None, **kwargs):
        data = tasks.recognition_embeddings() if data is None else data
        proj, S, VT = np.linalg.svd(data, full_matrices=False)
        stddev = S**2 / (data.shape[0] - 1)
        return cls(tasks, proj[:, :2], explained_variance=stddev/np.sum(stddev), **kwargs)

    @classmethod
    def tsne(cls, tasks, *, data=None, tsne_kwargs=dict(perplexity=30), **kwargs):
        data = tasks.recognition_embeddings() if data is None else data
        from sklearn.manifold import TSNE
        proj = TSNE(
            n_components=2,
            learning_rate='auto',
            init='random',
            **tsne_kwargs,
        ).fit_transform(data)
        return cls(tasks, proj, **kwargs)

# Various utils
def pairwise_iter(it):
    for prev, curr in zip(it[:-1], it[1:]):
        yield prev, curr

def nancorrcoef(*args):
    D = np.corrcoef(*args)
    D[np.isnan(D)] = 0
    return D


# Wrapper classes

class Tasks:
    def __init__(self, train, result, arguments):
        self.train = train
        self.result = result
        self.arguments = arguments

    @classmethod
    def from_bin(cls, fn):
        r = joblib.load(fn)
        return cls(**r)

    def iter_frontiers(self):
        yield from iter_frontiers(self.train, self.result)

    def compute_hidden_state_for_task(self, task):
        rm = self.result.recognitionModel
        return rm._MLP(rm.featureExtractor.featuresOfTask(task)).detach().numpy()

    def recognition_embeddings(self, *, compute=False):
        D = np.array([
            (
                self.compute_hidden_state_for_task(t)
                if compute else
                self.result.recognitionTaskMetrics[t]['hiddenState']
            )
            for t in self.train
        ])
        return D

    def library_count_matrix(self, *, grammar=None):
        grammar = grammar or self.result.grammars[-1]
        sorted_prims = sorted(
            [p for p in grammar.primitives if not p.isPrimitive],
            key=lambda p: (len(str(p)), str(p)))
        prim_to_i = {p: i for i, p in enumerate(sorted_prims)}

        counts = np.zeros((len(self.train), len(prim_to_i)))
        for i, (task, tf) in enumerate(self.iter_frontiers()):
            p = tf.best_program_entry().program
            for prim, ct in invention_counts(p, recurse_into_invented=True).items():
                counts[i, prim_to_i[prim]] = ct
        return counts

    def sorted_library(self):
        # HACK: doing this by sorting length of routine
        ps = [p for p in self.result.grammars[-1].primitives if not p.isPrimitive]
        sorted_prims = sorted(ps, key=lambda p: (len(str(p)), str(p)))
        return sorted_prims

    def default_prog_to_coloridx(self):
        return {
            inv: i
            for i, inv in enumerate(self.sorted_library())
        }


SPs = ['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '4.0', '6.0', '8.0', '11.0', '14.0']
class Data(object):

    def _load(self, grammar, iter):
        return Tasks.from_bin(f'{script_dir}/data-server-results/{grammar}/output-iter{iter}.bin')

    @functools.cached_property
    def pen(self):
        g = 'pen'
        return [
            self._load(g, 0),
            self._load(g, 1),
            self._load(g, 2),
            self._load(g, 3),
        ]

    @functools.cached_property
    def explicit_mark(self):
        g = 'explicit_mark'
        return [
            self._load(g, 0),
            self._load(g, 1),
            self._load(g, 2),
            self._load(g, 3),
        ]

    def _load_sps(self, folder, name, primitives, *, structurePenaltyValues=None):
        rv = {}
        for sp in structurePenaltyValues or SPs:
            d = self._load(folder, name(sp))
            rv[d.arguments['structurePenalty']] = d
            assert d.arguments['structurePenalty'] == float(sp)
            assert d.result.grammars[0].primitives == primitives
        return rv

    def _structure_penalty(self, *, grammar, tag='', **kwargs):
        if grammar == 'explicit_mark':
            gshort = 'em'
            prim = primitives_explicit_mark
        else:
            gshort = 'pen'
            prim = primitives_pen
        return self._load_sps(f'structurePenalty{tag}', lambda sp: f'3-sp{sp}-{gshort}', prim, **kwargs)

    def _noConsolidation(self, *, tag=''):
        rv = {
            g: self._load(f'noConsolidation{tag}/{g}', '3')
            for g in ['pen', 'explicit_mark']
        }
        assert rv['pen'].result.grammars[0].primitives == primitives_pen
        assert rv['explicit_mark'].result.grammars[0].primitives == primitives_explicit_mark
        for v in rv.values():
            assert v.arguments['noConsolidation'] == True
        return rv

    @functools.cached_property
    def structurePenalty_pen(self):
        return self._structure_penalty(grammar='pen')

    @functools.cached_property
    def structurePenalty_explicit_mark(self):
        return self._structure_penalty(grammar='explicit_mark')

    @functools.cached_property
    def structurePenalty_pen_rerun(self):
        return self._structure_penalty(grammar='pen', tag='-rerun', structurePenaltyValues=['0.5', '1.0', '1.5'])

    @functools.cached_property
    def structurePenalty_explicit_mark_rerun(self):
        return self._structure_penalty(grammar='explicit_mark', tag='-rerun', structurePenaltyValues=['0.5', '1.0', '1.5'])

    @functools.cached_property
    def structurePenalty_pen_noDreams(self):
        return self._structure_penalty(grammar='pen', tag='-noDreams', structurePenaltyValues=['1.0', '1.5'])

    @functools.cached_property
    def structurePenalty_explicit_mark_noDreams(self):
        return self._structure_penalty(grammar='explicit_mark', tag='-noDreams', structurePenaltyValues=['1.0', '1.5'])

    @functools.cached_property
    def noConsolidation(self):
        return self._noConsolidation()

    @functools.cached_property
    def noConsolidation_noDreams(self):
        return self._noConsolidation(tag='-noDreams')

    @functools.cached_property
    def language_lowlow(self):
        x = np.load(f'{script_dir}/data-language-embed/500_gsp_samples_text_lowlowlevel_encoded.npy')
        return x.mean(axis=1)

    @functools.cached_property
    def language_low(self):
        x = np.load(f'{script_dir}/data-language-embed/500_gsp_samples_text_lowlevel_encoded.npy', allow_pickle=True)
        return np.array([arr.mean(axis=0) for arr in x])

    @functools.cached_property
    def language_high(self):
        x = np.load(f'{script_dir}/data-language-embed/500_gsp_samples_text_highlevel_encoded.npy', allow_pickle=True)
        return np.array([arr.mean(axis=0) for arr in x])

    def iter_lang(self):
        for lang_name in ['language_lowlow', 'language_low', 'language_high']:
            yield lang_name, getattr(self, lang_name)

data = Data()
