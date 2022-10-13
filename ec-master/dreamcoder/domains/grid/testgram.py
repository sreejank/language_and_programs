from dreamcoder.domains.grid.grid import *

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

if __name__ == '__main__':
    main(primitives_loc, arrow(tgrid_cont,tgrid_cont))
