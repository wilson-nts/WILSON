import networkx as nx
from datetime import datetime
import numpy

def pagerank(G, alpha=0.85, personalization=None,
             max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
             dangling=None):
    """
    a modification from networkx
    """

    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = G
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:
        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:
        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        # danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        danglesum = 0
        
        for n in x:
            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights.get(n, 0) + (1.0 - alpha) * p.get(n, 0)
        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)

############# construct date reference graph  #############

def get_date_graph(dated_sents, potential_dates):
    G = nx.DiGraph()
    for dated_sent in dated_sents:
        dt1 = dated_sent[0][0]
        for dt2, sent in dated_sent[1:]:
            
            if dt2 in potential_dates and dt1 != '2011-03-0': # ignore dataset error
                
                d1 = datetime.strptime(dt1, '%Y-%m-%d')
                d2 = datetime.strptime(dt2, '%Y-%m-%d')
                
                # edge weight
                dw = abs((d1 - d2).days)
                # dw = 1.0
                if G.has_edge(dt1, dt2):
                    G[dt1][dt2]['weight'] += dw
                else:
                    G.add_edge(dt1, dt2, weight=dw)
    Gt = nx.DiGraph()
    norm = max([G[i[0]][i[1]]['weight'] for i in G.edges()])
    for i in G.edges():
        Gt.add_edge(i[0], i[1], weight=G[i[0]][i[1]]['weight'] / norm / len(G.nodes))
    return Gt

############# recency adjusted date selection #############

def get_dates(res, date_range, dt_sents, num):
    rks = sorted(res.items(), key=lambda x: x[1], reverse=True)
    cnt = 0
    dts = []
    for i in rks:
        t = datetime.strptime(i[0], '%Y-%m-%d').date()
        if date_range[0] <= t and t <= date_range[1] and i[0] in dt_sents:
            cnt += 1
            dts.append(t)
            if cnt >= num:
                break
    return dts

def measure_uniformity(dts, date_range):
    rks = dts + [date_range[0], date_range[1]]
    rks = sorted(rks)
    diffs = []
    for i in range(1, len(rks)):
        v = (rks[i] - rks[i - 1]).days
        if v > 0:
            diffs.append(v)
    return numpy.std(diffs)

def get_dates_perso(G, beta, pred_date_range, date_range, dt_sents, DATE_NUM):
    p = {}
    for i in G.nodes():
        d1 = datetime.strptime(i, '%Y-%m-%d').date()
        dw = abs((d1 - pred_date_range[0]).days) ** beta
        p.setdefault(i, dw)
    res = pagerank(G, personalization=p)
    dts = get_dates(res, date_range, dt_sents, DATE_NUM)
    cur = measure_uniformity(dts, date_range)
    return (cur, dts)