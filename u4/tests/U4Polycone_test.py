#!/usr/bin/env python

import os, numpy as np, textwrap
from opticks.ana.fold import Fold
SIZE = np.array([1280, 720])

try:
    import pyvista as pv
except ImportError:
    pv = None
pass


def csg():
    f = Fold.Load(symbol="f")
    print(repr(f))

    #snd = f.csg.node
    sn = f._csg.sn

    EXPR = list(filter(None, textwrap.dedent(r"""
    #snd[:,:11]
    sn
    #np.all( snd[:,:11] == sn )
    """).split("\n")))

    for expr in EXPR:
        print(expr)
        if expr[0] == "#": continue
        print(eval(expr))
    pass



if __name__ == '__main__':

    a = Fold.Load("$FOLD", symbol="f")
    print(repr(a))
    title = os.environ.get("TITLE", "U4Mesh_test.sh")

    pl = pv.Plotter(window_size=SIZE*2)
    pl.add_text(title, position="upper_left")

    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'pink']

    legend_labels = []

    for i, name in enumerate(sorted(a.names)):
        f = getattr(a, name)
        color = colors[i % len(colors)]
        offset = np.array([i*200,0,0], dtype=np.float32)
        pd = pv.PolyData(f.vtx + offset, f.tpd)   # tri and/or quad
        pl.add_mesh(pd, opacity=1.0, color=color, show_edges=True, lighting=True, name=name )
        legend_labels.append([name, color])
    pass
    pl.add_legend(legend_labels)
    pl.show()
    pass
pass







pass
