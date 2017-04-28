import matplotlib.pyplot as plt

def do_plt(pc):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, aspect='equal')
    polycone_plt(ax, pc)
    plt.show()


def polycone_plt(fig, pcs, nx=4, ny=4):
    for i in range(len(pcs)):
        ax = fig.add_subplot(nx,ny,i+1, aspect='equal')
        pc = pcs[i]
        pc.plot(ax)
    pass
     

if __name__ == "__main__":

    pcs = gdml.findall_("solids//polycone")
    plt.ion()
    fig = plt.figure()
    polycone_plt(fig, pcs)
    fig.show()


