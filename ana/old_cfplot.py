
def _cf_dump( msg, val, bins, label):
    log.warning("%s for  \"%s\" " % (msg, label) ) 
    log.warning(" val   %s " % repr(val) )
    log.warning(" bins  %s " % repr(bins) )

def __cf_hist( ax, val, bins, log_, label):
    """
    ax.hist gives errors for empty histos so use np.histogram and plot counts and bins afterwards
    """
    c, b = np.histogram(val, bins=bins)
    p = ax.plot( bins[:-1], c , drawstyle="steps", label=label  )
    if log_:
        ax.set_yscale('log')
    return c, b, p

def _cf_hist( ax, val, bins, log_, label):
    c, b, p = None, None, None
    try:
        c,b,p = __cf_hist(ax, val, bins, log_, label) 
    except IndexError:
        _cf_dump("_cf_hist IndexError", val, bins, label)
    except ValueError:
        _cf_dump("_cf_hist ValueError", val, bins, label)
    pass
    return c, b, p


def _cf_plot(ax, aval, bval,  bins, labels,  log_=False):
    cnt = {}
    bns = {}
    ptc = {}

    cnt[0], bns[0], ptc[0] = __cf_hist(ax, aval, bins=bins,  log_=log_, label=labels[0])
    cnt[1], bns[1], ptc[1] = __cf_hist(ax, bval, bins=bins,  log_=log_, label=labels[1])

    return cnt, bns

def _chi2_plot(ax, _bins, counts, cut=30):
    a,b = counts[0],counts[1]

    if a is None or b is None:
        log.warning("skip chi2 plot as got None %s %s " % (repr(a), repr(b)))
        return 0  

    c2, c2n, c2c = chi2(a, b, cut=cut)
    ndf = max(c2n - 1, 1)

    c2p = c2.sum()/ndf
       
    label = "chi2/ndf %4.2f [%d]" % (c2p, ndf)

    ax.plot( _bins[:-1], c2, drawstyle='steps', label=label )

    return c2p


def old_cfplot(fig, gss, _bins, aval, bval, labels=["A","B"], log_=False, c2_cut=30, c2_ymax=10, logyfac=3., linyfac=1.3): 

    ax = fig.add_subplot(gss[0])

    cnt, bns = _cf_plot(ax, aval, bval, bins=_bins, labels=labels, log_=log_)

    ymin = 1 if log_ else 0 
    yfac = logyfac if log_ else linyfac
    
    ymax = 0 
    for k,v in cnt.items():
        if v is None:continue
        vmax = v.max()
        ymax = max(ymax, vmax) 
    pass

    ylim = [ymin,ymax*yfac]

    ax.set_ylim(ylim)
    ax.legend()
    xlim = ax.get_xlim()


    ax = fig.add_subplot(gss[1])

    c2p = _chi2_plot(ax, _bins, cnt, cut=c2_cut)  

    ax.set_xlim(xlim) 
    ax.legend()
    ax.set_ylim([0,c2_ymax]) 

    return c2p  


def old_qwns_plot(ab, qwns, irec, log_=False, c2_cut=30):

    log.info("qwns_plot(scf, \"%s\", %d, log_=%s, c2_cut=%d )" % (qwns, irec, log_ , c2_cut)) 

    fig = plt.figure()

    ab.irec = irec

    fig.suptitle(ab.suptitle)

    nx = len(qwns)

    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])

    c2ps = []
    for ix in range(nx):

        gss = [gs[ix], gs[nx+ix]]

        qwn = qwns[ix]

        bns, aval, bval, labels = ab.rqwn(qwn, irec)

        c2p = cfplot(fig, gss, bns, aval, bval, labels=labels, log_=log_, c2_cut=c2_cut )

        c2ps.append(c2p) 
    pass

    qd = odict(zip(list(qwns),c2ps))
    return qd





def old_qwn_plot(ab, qwn, irec, log_=False, c2_cut=30, c2_ymax=10):

    ab.irec = irec      

    rqwn_bins, aval, bval, labels = ab.rqwn(qwn, irec)

    fig = plt.figure()
    fig.suptitle(ab.suptitle)

    nx,ix = 1,0
    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])
    gss = [gs[ix], gs[nx+ix]]

    c2p = cfplot(fig, gss, rqwn_bins, aval, bval, labels=labels, log_=log_, c2_cut=c2_cut, c2_ymax=c2_ymax)
    c2ps = [c2p]

    #print "c2p", c2p

    qd = odict(zip(list(qwn),c2ps))
    return qd



def test_old_cfplot():

    aval = np.random.standard_normal(8000)
    bval = np.random.standard_normal(8000)
    bins = np.linspace(-4,4,200)
    log_ = False

    fig = plt.figure()
    fig.suptitle("cfplot test")

    nx = 4
    gs = gridspec.GridSpec(2, nx, height_ratios=[3,1])
    for ix in range(nx):
        gss = [gs[ix], gs[nx+ix]]
        cfplot(fig, gss, bins, aval, bval, labels=["A test", "B test"], log_=log_ )



