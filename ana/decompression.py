#!/usr/bin/env python
"""

One valued inputs that land on compression bins::

    ipdb> p aval
    A()sliced
    A([ 15.4974,  15.4974,  15.4974, ...,  15.4974,  15.4974,  15.4974])
    ipdb> p bval
    A()sliced
    A([ 15.4934,  15.4934,  15.4934, ...,  15.4934,  15.4934,  15.4934])

    ipdb> p cbins[3840:3840+10]
    array([ 15.4692,  15.4733,  15.4773,  15.4813,  15.4853,  15.4894,  15.4934,  15.4974,  15.5014,  15.5055])

"""
import numpy as np
import os, logging
log = logging.getLogger(__name__) 


class Deco(object):
    def __init__(self, cbins, debug=True, label="", binscale=100):
        self.cbins = cbins
        self.debug = debug
        self.label = label
        self.binscale = binscale

    @classmethod
    def pbins(cls, extent=5005):
        return np.linspace(-extent, extent, (1 << 16) - 1 )

    def vrange(self, vals):
        vmin = float(min(map(lambda _:_.min(), vals)))
        vmax = float(max(map(lambda _:_.max(), vals)))
        vone = map(lambda v:np.all(v[v == v[0]]), vals)
        return vmin, vmax

    def irange(self, vals):

        """


                      |               |>>>>>>>>>>>>>>>>>>>
            ..........|...............|....................
                      |               | 
                     vmin            vmax
                      |               |
            <<<<<<<<<<|               |
                     ^                 ^
                    imin              imax

            In [33]: np.where(cbins < vmin)
            Out[33]: (array([    0,     1,     2, ..., 65467, 65468, 65469]),)

            In [32]: np.where(cbins > vmax)
            Out[32]: 
            (array([65470, 65471, 65472, 65473,


        """
        vmin, vmax = self.vrange(vals)   
        assert vmax >= vmin
        a = vals[0]
        if len(vals) > 1:
            b = vals[1]
        else:
            b = None

        cmin = self.cbins[0]
        cmax = self.cbins[-1]
        assert cmax >= cmin

        if vmin < cmin:
            log.warning("[%s] vmin < cmin : %s < %s " % (self.label, vmin, cmin ))
            log.warning("a[a < cmin] %r " % a[a < cmin] )
            log.warning("b[b < cmin] %r " % b[b < cmin] )
            imin = 0 
        else:
            imin = np.where(self.cbins<=vmin)[0][-1]
        pass
        if vmax > cmax:
            log.warning("[%s] vmax > cmax : %s > %s " % (self.label, vmax, cmax ))
            log.warning("a[a > cmax] %r " % a[a > cmax] )
            log.warning("b[b > cmax] %r " % b[b > cmax] )
            imax = len(self.cbins) - 1
        else: 
            imax = np.where(self.cbins>=vmax)[0][0]
        pass

        assert imax >= imin
        return imin, imax


    def bins(self, vals, mibin=10):
        """
        :param vals: list of value arrays
        :param mibin: minimum number of bins to grow to
        :return bins: array of bin edges
        """
        imin, imax = self.irange(vals)
        nbin = len(self.cbins)
        umin = imin
        umax = imax
        inc = self.binscale/10  # when inc too big cannot grow when near the edge
        while True:
            umin = max(0, umin - inc )
            umax = min( nbin - 1, umax + inc )
            bins = self.cbins[umin:umax+1][::self.binscale]
            lenb = len(bins) 
            if self.debug:
                log.info("umin %d umax %d lenb %d bins %r " % (umin, umax, lenb, bins))
            if lenb > mibin or (umin == 0 and umax == nbin - 1) : 
                break
            pass
        pass
        return bins



def decompression_bins(cbins, vals, debug=False, label="", binscale=100):
    """
    :param cbins: full range decompressed bins 
    :param vals:

    Compression can be considered to be a very early (primordial) binning.  
    To avoid artifacts all subsequent binning needs to be
    use bins that correspond to these originals. 

    This provides a subset of full range decompression bins, 
    corresponding to a value range.

    ::

        In [3]: cbins = cf.a.pbins()

        In [4]: cbins
        Out[4]: array([ -24230.625 ,  -24242.3772,  -24254.1294, ..., -794375.8706, -794387.6228, -794399.375 ])

        In [15]: np.where( cbins <=  -60254.1294 )
        Out[15]: (array([ 3066,  3067,  3068, ..., 65532, 65533, 65534]),)


    """
       
    dc = Deco(cbins, debug=debug, label=label, binscale=binscale)
    return dc.bins(vals)



def test_decompression_bins_(cbins, avals, bvals, binscale=1):
    dbins = decompression_bins(cbins, [avals, bvals], debug=True, binscale=binscale)
    # assert len(dbins) > 1 
    log.info("avals : %s " % repr(avals))
    log.info("bvals : %s " % repr(bvals))
    log.info("dbins : %s " % repr(dbins))

def test_decompression_bins_0():
    cbins = np.linspace(-300,300,10)
    avals = np.repeat(300,1000)
    bvals = np.repeat(300,1000)
    test_decompression_bins_(cbins, avals, bvals, binscale=1)

def test_decompression_bins_1():
    cbins = np.arange(0, 132.001, 0.004, dtype=np.float32)
    avals = np.repeat( cbins[3840], 100)
    bvals = np.repeat( cbins[3841], 100)
    test_decompression_bins_(cbins, avals, bvals, binscale=100)

def test_decompression_bins_2():
    cbins = np.arange(0, 132.001, 0.004, dtype=np.float32)
    avals = np.repeat( cbins[-1], 100)
    bvals = np.repeat( cbins[-1], 100)
    test_decompression_bins_(cbins, avals, bvals, binscale=100)





if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    #test_decompression_bins_0() 
    #test_decompression_bins_1() 

    cbins = Deco.pbins(extent=5005)
    avals = np.repeat( cbins[-1], 100)
    bvals = np.repeat( cbins[-1], 100)

    dc = Deco(cbins, debug=True)

    bins = dc.bins([avals, bvals])

    print "cbins",cbins 
    print avals 
    print bvals 
    print bins


 
