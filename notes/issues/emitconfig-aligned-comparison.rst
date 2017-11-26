Emitconfig aligned comparison
===============================


Overview
-----------

With emitconfig input photons the two sims are not indep,  

* see :doc:`emitconfig-cfg4-chisq-too-good-as-not-indep-samples`

Hence need to step-by-step compare photons in a history aligned way.
So need way to seqhis select a category with additional requirement that 
the photons in each sim that are aligned.



::

    simon:ana blyth$ tboolean-;tboolean-box-ip
    args: /opt/local/bin/ipython -i -- /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    [2017-11-24 19:31:56,600] p33909 {/Users/blyth/opticks/ana/base.py:316} INFO - envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'concentric'} 
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    ok.smry 1 
    [2017-11-24 19:31:56,602] p33909 {/Users/blyth/opticks/ana/tboolean.py:27} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython True 
    [2017-11-24 19:31:56,603] p33909 {/Users/blyth/opticks/ana/ab.py:81} INFO - AB.load START smry 1 
    [2017-11-24 19:31:56,774] p33909 {/Users/blyth/opticks/ana/ab.py:108} INFO - AB.load DONE 
    [2017-11-24 19:31:56,778] p33909 {/Users/blyth/opticks/ana/ab.py:150} INFO - AB.init_point START
    [2017-11-24 19:31:56,780] p33909 {/Users/blyth/opticks/ana/ab.py:152} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171124-1909 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171124-1909 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         4.44/5 =  0.89  (pval:0.488 prob:0.512)  
    0000      55321     55303             0.00  TO SA
    0001      39222     39231             0.00  TO BT BT SA
    0002       2768      2814             0.38  TO BR SA
    0003       2425      2369             0.65  TO BT BR BT SA
    0004        151       142             0.28  TO BT BR BR BT SA
    0005         54        74             3.12  TO SC SA
    0006         13        16             0.00  TO BT BT SC SA
    0007         12         8             0.00  TO BT BR BR BR BT SA

    ...

    In [10]: np.where(ab.a.seqhis != ab.b.seqhis)[0].shape
    Out[10]: (9945,)

    In [11]: np.where(ab.a.seqhis == ab.b.seqhis)[0].shape    # aligned dominates, this is before any BR cheating 
    Out[11]: (90055,)


    In [14]: ab.a.flvana.seq_or?
    Type:       instancemethod
    String Form:<bound method SeqAna.seq_or of <opticks.ana.seq.SeqAna object at 0x10554c290>>
    File:       /Users/blyth/opticks/ana/seq.py
    Definition: ab.a.flvana.seq_or(self, sseq)
    Docstring:
    :param sseq: list of sequence labels including source, eg "TO BR SA" "TO BR AB"
    :return psel: selection boolean array of photon length

    Selection of photons with any of the sequence arguments


    In [1]: seq = ab.a.flvana.seq_or(["TO BT BT SA"])   # select the straight thru 

    In [2]: seq.shape
    Out[2]: (100000,)

    In [3]: align = ab.a.seqhis == ab.b.seqhis

    In [4]: align.shape
    Out[4]: (100000,)


    In [5]: seqalign = np.logical_and( seq, align )

    In [6]: seqalign.shape
    Out[6]: (100000,)

    In [7]: np.where( seq == True )[0].shape
    Out[7]: (39222,)

    In [8]: np.where( seqalign == True )[0].shape
    Out[8]: (34492,)

    ## random BR in either sim looses alignment        

    In [10]: ab.a.psel = seqalign
    In [11]: ab.b.psel = seqalign

    In [13]: ab.a.rpost_(slice(0,4)).shape
    Out[13]: (34492, 4, 4)

    In [14]: ab.b.rpost_(slice(0,4)).shape
    Out[14]: (34492, 4, 4)



Aligned value comparisons : reveals possible GPU/CPU compression rounding difference
--------------------------------------------------------------------------------------

::

    In [17]: av = ab.a.rpost_(slice(0,4))

    In [18]: bv = ab.b.rpost_(slice(0,4))


    In [27]: np.allclose( av[:900], bv[:900] )
    Out[27]: True


    In [33]: dv = np.abs( av - bv )

    In [34]: dv.shape
    Out[34]: (34492, 4, 4)


    In [35]: np.where( dv > 0.1 )
    Out[35]: 
    (A()sliced
    A([], dtype=int64),
     A()sliced
    A([], dtype=int64),
     A()sliced
    A([], dtype=int64))

    In [36]: np.where( dv > 0.01 )
    Out[36]: 
    (A()sliced
    A([  940,   940,   940,   940,  8019,  8019,  8019,  8019, 13879, 13879, 13879, 13879, 16210, 16210, 16210, 16210, 17710, 17710, 17710, 17710, 18238, 18238, 18238, 18238, 20476, 20476, 20476,
           20476, 21314, 21314, 21314, 21314, 22018, 22018, 22018, 22018, 22343, 22343, 22343, 22343, 22524, 22524, 22524, 22524, 23088, 23088, 23088, 23088, 23805, 23805, 23805, 23805, 30057, 30057,
           30057, 30057, 30162, 30162, 30162, 30162, 32709, 32709, 32709, 32709, 33596, 33596, 33596, 33596, 33881, 33881, 33881, 33881]),
     A()sliced
    A([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
           0, 1, 2, 3, 0, 1, 2, 3]),
     A()sliced
    A([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
           1, 1, 1, 1, 0, 0, 0, 0]))


    In [38]: discrep = np.where( dv > 0.01 )

    In [39]: av[discrep]
    Out[39]: 
    A()sliced
    A([ -47.4853,  -47.4853,  -47.4853,  -47.4853,   30.4181,   30.4181,   30.4181,   30.4181,  116.2219,  116.2219,  116.2219,  116.2219,  -78.4815,  -78.4815,  -78.4815,  -78.4815,   76.3894,
             76.3894,   76.3894,   76.3894, -135.3812, -135.3812, -135.3812, -135.3812,  138.5194,  138.5194,  138.5194,  138.5194,  120.5988,  120.5988,  120.5988,  120.5988,   78.3713,   78.3713,
             78.3713,   78.3713, -107.4131, -107.4131, -107.4131, -107.4131,   -9.4971,   -9.4971,   -9.4971,   -9.4971,  118.3691,  118.3691,  118.3691,  118.3691,  -99.5401,  -99.5401,  -99.5401,
            -99.5401, -135.2711, -135.2711, -135.2711, -135.2711,   94.42  ,   94.42  ,   94.42  ,   94.42  ,  104.4676,  104.4676,  104.4676,  104.4676, -131.3622, -131.3622, -131.3622, -131.3622,
            -81.3994,  -81.3994,  -81.3994,  -81.3994])

    In [40]: bv[discrep]
    Out[40]: 
    A()sliced
    A([ -47.499 ,  -47.499 ,  -47.499 ,  -47.499 ,   30.4319,   30.4319,   30.4319,   30.4319,  116.2357,  116.2357,  116.2357,  116.2357,  -78.4952,  -78.4952,  -78.4952,  -78.4952,   76.4031,
             76.4031,   76.4031,   76.4031, -135.395 , -135.395 , -135.395 , -135.395 ,  138.5331,  138.5331,  138.5331,  138.5331,  120.6126,  120.6126,  120.6126,  120.6126,   78.3851,   78.3851,
             78.3851,   78.3851, -107.4268, -107.4268, -107.4268, -107.4268,   -9.5108,   -9.5108,   -9.5108,   -9.5108,  118.3829,  118.3829,  118.3829,  118.3829,  -99.5539,  -99.5539,  -99.5539,
            -99.5539, -135.2849, -135.2849, -135.2849, -135.2849,   94.4338,   94.4338,   94.4338,   94.4338,  104.4814,  104.4814,  104.4814,  104.4814, -131.3759, -131.3759, -131.3759, -131.3759,
            -81.4132,  -81.4132,  -81.4132,  -81.4132])

    In [41]: 



Huh hows that possible, different x (perhaps some precision edge effect)::

    In [41]: av[940]
    Out[41]: 
    A()sliced
    A([[ -47.4853,   16.7093, -449.8989,    0.2002],
           [ -47.4853,   16.7093,  -99.9944,    1.3672],
           [ -47.4853,   16.7093,   99.9944,    2.5788],
           [ -47.4853,   16.7093,  449.9952,    3.7465]])

    In [42]: bv[940]
    Out[42]: 
    A()sliced
    A([[ -47.499 ,   16.7093, -449.8989,    0.2002],
           [ -47.499 ,   16.7093,  -99.9944,    1.3672],
           [ -47.499 ,   16.7093,   99.9944,    2.5788],
           [ -47.499 ,   16.7093,  449.9952,    3.7465]])

    In [43]: 



Source data is persisted::

    1661 void OpticksEvent::saveSourceData()
    1662 {
    1663     // source data originates CPU side, and is INPUT_ONLY to GPU side
    1664     NPY<float>* so = getSourceData();
    1665     if(so) so->save("so", m_typ,  m_tag, m_udet);
    1666 }



Yep deviations all same size, domain compression artifact::


    In [50]: av[discrep] - bv[discrep]
    Out[50]: 
    A()sliced
    A([ 0.0138,  0.0138,  0.0138,  0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138,  0.0138,  0.0138,  0.0138,  0.0138, -0.0138, -0.0138, -0.0138, -0.0138,  0.0138,
            0.0138,  0.0138,  0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138,  0.0138,  0.0138,  0.0138,  0.0138,  0.0138,  0.0138,
            0.0138,  0.0138, -0.0138, -0.0138, -0.0138, -0.0138,  0.0138,  0.0138,  0.0138,  0.0138,  0.0138,  0.0138,  0.0138,  0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138, -0.0138,
           -0.0138,  0.0138,  0.0138,  0.0138,  0.0138,  0.0138,  0.0138,  0.0138,  0.0138])



This geom is using auto sizing::

     710 tboolean-box--(){ cat << EOP 
     711 import logging
     712 log = logging.getLogger(__name__)
     713 from opticks.ana.base import opticks_main
     714 from opticks.analytic.polyconfig import PolyConfig
     715 from opticks.analytic.csg import CSG  
     716 
     717 args = opticks_main(csgpath="$TMP/$FUNCNAME")
     718 
     719 emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75" 
     720 
     721 CSG.kwa = dict(poly="IM",resolution="20", verbosity="0",ctrl="0", containerscale="3", emitconfig=emitconfig  )
     722 
     723 container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container="1" )  # no param, container="1" switches on auto-sizing
     724 
     725 box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2" )
     726 
     727 CSG.Serialize([container, box], args )
     728 EOP
     729 }


fdom::

    In [51]: ab.a.fdom
    Out[51]: 
    A(torch,1,tboolean-box)(metadata) 3*float4 domains of position, time, wavelength (used for compression)
    A([[[   0.,    0.,    0.,  451.]],

           [[   0.,   20.,   20.,    0.]],

           [[  60.,  820.,   20.,  760.]]], dtype=float32)


What is compression granularity for extent 451. ?:: 

    In [54]: 32767./451.
    Out[54]: 72.65410199556541

    In [55]: 1./(32767./451.)
    Out[55]: 0.013763847773674733

    In [58]: np.arange(0, 32767, 1, dtype=np.float64)*1./(32767./451.)
    Out[58]: array([   0.    ,    0.0138,    0.0275, ...,  450.9587,  450.9725,  450.9862])


    In [59]: vv = np.arange(0, 32767, 1, dtype=np.float64)*1./(32767./451.)         


Discrepancies are all float compression rounded to nextdoor ints.::

    In [71]: vv[3450:3453]
    Out[71]: array([ 47.4853,  47.499 ,  47.5128])



Rounding
----------

* https://mathematica.stackexchange.com/questions/2116/why-round-to-even-integers



Can the compression be made more agreeable between GPU/CPU ?
---------------------------------------------------------------

* http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__CAST.html#group__CUDA__MATH__INTRINSIC__CAST_1ga0223a729c7bda6096fc7fc212df32cd

::

    __device__ int __float2int_rn ( float  x )
    Convert a float to a signed integer in round-to-nearest-even mode.

::

    083 
     84 __device__ short shortnorm( float v, float center, float extent )
     85 {
     86     // range of short is -32768 to 32767
     87     // Expect no positions out of range, as constrained by the geometry are bouncing on,
     88     // but getting times beyond the range eg 0.:100 ns is expected
     89     //
     90     int inorm = __float2int_rn(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
     91     return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
     92 }

    102 __device__ void rsave( Photon& p, State& s, optix::buffer<short4>& rbuffer, unsigned int record_offset, float4& center_extent, float4& time_domain )
    103 {
    104     //  pack position and time into normalized shorts (4*16 = 64 bits)
    105     //
    106     //  TODO: use a more vectorized approach, ie
    107     // 
    108     //  * combine position and time domains into single float4 on the host 
    109     //  * after verification can dispense with the fit checking for positions, just do time
    110     //        
    111     //  * adopt p.position_time  maybe p.polarization_wavelength
    112     //  * simularly with domains of those ?
    113     // 
    114     rbuffer[record_offset+0] = make_short4(    // 4*int16 = 64 bits 
    115                     shortnorm(p.position.x, center_extent.x, center_extent.w),
    116                     shortnorm(p.position.y, center_extent.y, center_extent.w),
    117                     shortnorm(p.position.z, center_extent.z, center_extent.w),
    118                     shortnorm(p.time      , time_domain.x  , time_domain.y  )
    119                     );



CWriter::

     32 
     33 #define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)
     34 #define iround(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))
     35 
     36 short CWriter::shortnorm( float v, float center, float extent )  // static 
     37 {
     38     // range of short is -32768 to 32767
     39     // Expect no positions out of range, as constrained by the geometry are bouncing on,
     40     // but getting times beyond the range eg 0.:100 ns is expected
     41     //  
     42     int inorm = iround(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
     43     return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
     44 }
     45 
     46 unsigned char CWriter::my__float2uint_rn( float f ) // static
     47 {
     48     return iround(f);
     49 }


* http://en.cppreference.com/w/cpp/numeric/math/nearbyint
* http://en.cppreference.com/w/cpp/numeric/fenv/FE_round



All discrep are all the way thru (every point) difference in x/y::

    In [45]: discrep[0].reshape(-1,4)
    Out[45]: 
    A()sliced
    A([[  940,   940,   940,   940],
           [ 8019,  8019,  8019,  8019],
           [13879, 13879, 13879, 13879],
           [16210, 16210, 16210, 16210],
           [17710, 17710, 17710, 17710],
           [18238, 18238, 18238, 18238],
           [20476, 20476, 20476, 20476],
           [21314, 21314, 21314, 21314],
           [22018, 22018, 22018, 22018],
           [22343, 22343, 22343, 22343],
           [22524, 22524, 22524, 22524],
           [23088, 23088, 23088, 23088],
           [23805, 23805, 23805, 23805],
           [30057, 30057, 30057, 30057],
           [30162, 30162, 30162, 30162],
           [32709, 32709, 32709, 32709],
           [33596, 33596, 33596, 33596],
           [33881, 33881, 33881, 33881]])

    In [46]: discrep[1].reshape(-1,4)
    Out[46]: 
    A()sliced
    A([[0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3],
           [0, 1, 2, 3]])

    In [47]: discrep[2].reshape(-1,4)
    Out[47]: 
    A()sliced
    A([[0, 0, 0, 0],
           [1, 1, 1, 1],
           [0, 0, 0, 0],
           [1, 1, 1, 1],
           [0, 0, 0, 0],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [0, 0, 0, 0],
           [1, 1, 1, 1],
           [0, 0, 0, 0],
           [1, 1, 1, 1],
           [1, 1, 1, 1],
           [0, 0, 0, 0]])








ana/evt.py::

     794     # *psel* provides low level selection control via  boolean array 
     795     def _get_psel(self):
     796         return self._psel
     797     def _set_psel(self, psel):
     798         self._init_selection(psel)
     799     psel = property(_get_psel, _set_psel)
     800 

     823     # *sel* provides high level selection control using slices, labels, hexint etc
     824     def _get_sel(self):
     825         return self._sel
     826     def _set_sel(self, arg):
     827         log.debug("Evt._set_sel %s " % repr(arg))
     828 
     829         if arg is None:
     830             sel = None
     831         else:
     832             sel = self._parse_sel(arg)
     833         pass
     834         self._sel = sel
     835 
     836         psel = self.make_selection(sel, False)
     837         self._init_selection(psel)
     838     sel = property(_get_sel, _set_sel)




::

    delta:tests blyth$ float2intTest
       47.4165 3445      47.4165 3445      47.4165 3445       47.4165  3445 
       47.4302 3446      47.4302 3446      47.4302 3446       47.4302  3446 
       47.4440 3447      47.4440 3447      47.4440 3447       47.4440  3447 
       47.4577 3448      47.4577 3448      47.4577 3448       47.4577  3448 
       47.4715 3449      47.4715 3449      47.4715 3449       47.4715  3449 
       47.4853 3450      47.4853 3450      47.4853 3450       47.4853  3450 
       47.4990 3451      47.4990 3451      47.4990 3451       47.4990  3451 
       47.5128 3452      47.5128 3452      47.5128 3452       47.5128  3452 
       47.5266 3453      47.5266 3453      47.5266 3453       47.5266  3453 
       47.5403 3454      47.5403 3454      47.5403 3454       47.5403  3454 
    delta:tests blyth$ 

    delta:tests blyth$ boost_numeric_converter_Test
     i       3440   v  47.347637 iv       3440
     i       3441   v  47.361401 iv       3441
     i       3442   v  47.375164 iv       3442
     i       3443   v  47.388927 iv       3443
     i       3444   v  47.402691 iv       3444
     i       3445   v  47.416454 iv       3445
     i       3446   v  47.430218 iv       3446
     i       3447   v  47.443985 iv       3447
     i       3448   v  47.457748 iv       3448
     i       3449   v  47.471512 iv       3449
     i       3450 * v  47.485275 iv       3450
     i       3451 * v  47.499039 iv       3451
     i       3452 * v  47.512802 iv       3452
     i       3453   v  47.526566 iv       3453
     i       3454   v  47.540329 iv       3454
     i       3455   v  47.554092 iv       3455
     i       3456   v  47.567860 iv       3456
     i       3457   v  47.581623 iv       3457
     i       3458   v  47.595387 iv       3458
     i       3459   v  47.609150 iv       3459
    delta:tests blyth$ 



