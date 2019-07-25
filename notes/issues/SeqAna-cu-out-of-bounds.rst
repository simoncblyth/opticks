SeqAna-cu-out-of-bounds
=========================


Issue with event loading Reported by Sam
--------------------------------------------

::

    Hi Simon,

    That delt with the first error, but the test stills fails with:

    args: /home/opc/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show
    [2019-07-24 18:09:02,701] p29765 {<module>            :tboolean.py:30} INFO     - pfx tboolean-box tag 1 src torch det tboolean-box c2max [1.5, 2.0, 2.5] ipython False 
    [2019-07-24 18:09:02,702] p29765 {load                :ab.py     :274} INFO     - [ False 
    [2019-07-24 18:09:02,703] p29765 {__init__            :evt.py    :173} INFO     - [ A 
    [2019-07-24 18:09:02,703] p29765 {__init__            :metadata.py:44} INFO     - path /tmp/opc/opticks/tboolean-box/evt/tboolean-box/torch/1/DeltaTime.ini 
    [2019-07-24 18:09:02,703] p29765 {__init__            :metadata.py:62} INFO     - path /tmp/opc/opticks/tboolean-box/evt/tboolean-box/torch/1/OpticksEvent_launch.ini does not exist 
    [2019-07-24 18:09:02,723] p29765 {check_ox_fdom       :evt.py    :535} WARNING  -  t :   0.000   9.020 : tot 100000 over 1 0.000  under 0 0.000 : mi      0.291 mx      9.805  
    [2019-07-24 18:09:02,853] p29765 {__init__            :evt.py    :233} INFO     - ] A 
    [2019-07-24 18:09:02,853] p29765 {__init__            :evt.py    :173} INFO     - [ B 
    [2019-07-24 18:09:02,854] p29765 {__init__            :metadata.py:44} INFO     - path /tmp/opc/opticks/tboolean-box/evt/tboolean-box/torch/-1/DeltaTime.ini 
    [2019-07-24 18:09:02,854] p29765 {__init__            :metadata.py:62} INFO     - path /tmp/opc/opticks/tboolean-box/evt/tboolean-box/torch/-1/OpticksEvent_launch.ini does not exist 
    [2019-07-24 18:09:02,866] p29765 {check_ox_fdom       :evt.py    :535} WARNING  -  t :   0.000   9.020 : tot 100000 over 1 0.000  under 0 0.000 : mi      0.291 mx      9.805  
    Traceback (most recent call last):
      File "/home/opc/opticks/ana/tboolean.py", line 32, in <module>
        ab = AB(ok)
      File "/home/opc/opticks/ana/ab.py", line 250, in __init__
        self.load()
      File "/home/opc/opticks/ana/ab.py", line 287, in load
        b = Evt(tag=btag, src=args.src, det=args.det, pfx=args.pfx, args=args, nom="B", smry=args.smry)
      File "/home/opc/opticks/ana/evt.py", line 228, in __init__
        self.init() 
      File "/home/opc/opticks/ana/evt.py", line 246, in init
        self.init_hits()
      File "/home/opc/opticks/ana/evt.py", line 559, in init_hits
        self.hflags_ana = SeqAna( self.hflags, self.hismask, cnames=[self.cn], dbgseq=self.dbgmskhis, dbgzero=self.dbgzero, cmx=self.cmx, smry=self.smry)
      File "/home/opc/opticks/ana/seq.py", line 548, in __init__
        cu = count_unique_sorted(aseq)
      File "/home/opc/opticks/ana/nbase.py", line 103, in count_unique_sorted
        cu = cu[np.argsort(cu[:,1])[::-1]]  # descending frequency order
    IndexError: index 0 is out of bounds for axis 0 with size 0
    2019-07-24 18:09:03.305 INFO  [29731] [SSys::run@72] tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show   rc_raw : 256 rc : 1
    2019-07-24 18:09:03.305 ERROR [29731] [SSys::run@79] FAILED with  cmd tboolean.py --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show   RC 1

    So still the event isn't being read?
    Do I need to have something else defined?

    Cheers,
    Sam
    _._,_._,_



For me the below code works fine when there are no hits, but seemingly not for Sam.
This could be a NumPy version difference.

ana/evt.py::

     541     def init_hits(self):
     542         log.debug("init_hits")
     543         ht = self.aload("ht",optional=True)
     544         self.ht = ht
     545         self.desc['ht'] = "(hits) surface detect SD final photon steps"
     546 
     547         if ht.missing:return
     548 
     549         hwl = ht[:,2,W]
     550         hc4 = ht[:,3,2].copy().view(dtype=[('x',np.uint8),('y',np.uint8),('z',np.uint8),('w',np.uint8)]).view(np.recarray)
     551 
     552         self.hwl = hwl
     553         self.hpost = ht[:,0]
     554         self.hdirw = ht[:,1]
     555         self.hpolw = ht[:,2]
     556         self.hflags = ht.view(np.uint32)[:,3,3]
     557         self.hc4 = hc4
     558 
     559         self.hflags_ana = SeqAna( self.hflags, self.hismask, cnames=[self.cn], dbgseq=self.dbgmskhis, dbgzero=self.dbgzero, cmx=self.cmx, smry=self.smry)
     560 
     561         self.desc['hwl'] = "(hits) wavelength"
     562         self.desc['hpost'] = "(hits) final photon step: position, time"
     563         self.desc['hdirw'] = "(hits) final photon step: direction, weight "
     564         self.desc['hpolw'] = "(hits) final photon step: polarization, wavelength "
     565         self.desc['hflags'] = "(hits) final photon step: flags "
     566         self.desc['hc4'] = "(hits) final photon step: dtype split uint8 view of ox flags"
     567 


ana/seq.py::

    513 class SeqAna(object):
    514     """
    515     Canonical usage is from evt with::
    516 
    517         self.seqhis_ana = SeqAna(self.seqhis, self.histype) 
    518         self.seqmat_ana = SeqAna(self.seqmat, self.mattype)   
    519 
    520     In addition to holding the SeqTable instance SeqAna provides
    521     methods to make boolean array selections using the aseq and
    522     form labels. 
    523 
    524     SeqAna and its contained SeqTable exist within a particular selection, 
    525     ie changing selection entails recreation of SeqAna and its contained SeqTable
    526 
    527     """
    528     @classmethod
    529     def for_evt(cls, af, tag="1", src="torch", det="dayabay", pfx="source", offset=0):
    530         ph = A.load_("ph",src,tag,det, pfx=pfx)
    531         aseq = ph[:,0,offset]
    532         return cls(aseq, af, cnames=[tag])
    533 
    534     def __init__(self, aseq, af, cnames=["noname"], dbgseq=0, dbgmsk=0, dbgzero=False, cmx=0, smry=False):
    535         """
    536         :param aseq: photon length sequence array 
    537         :param af: instance of SeqType subclass, which knows what the codes mean 
    538 
    539         ::
    540 
    541             In [10]: sa.aseq
    542             A([  9227469,   9227469, 147639405, ...,   9227469,   9227469,     19661], dtype=uint64)
    543 
    544             In [11]: sa.aseq.shape
    545             Out[11]: (1000000,)
    546 
    547         """
    548         cu = count_unique_sorted(aseq)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //             FAILS WHEN aseq is EMPTY FOR Sam  

    549         self.smry = smry
    550         self.af = af
    551         self.dbgseq = dbgseq
    552         self.dbgmsk = dbgmsk
    553         self.dbgzero = dbgzero
    554         self.cmx = cmx
    555 
    556         self.table = SeqTable(cu, af, cnames=cnames, dbgseq=self.dbgseq, dbgmsk=self.dbgmsk, dbgzero=self.dbgzero, cmx=self.cmx, smry=self.smry)
    557 




Jump into ipython with these events: a (Opticks) and b (Geant4)::

    [blyth@localhost ana]$ t ta
    ta is a function
    ta () 
    { 
        LV=$1 tboolean.sh --ip ${@:2}
    }

    [blyth@localhost ana]$ ta box
    ====== /home/blyth/opticks/bin/tboolean.sh --ip ====== PWD /home/blyth/opticks/ana =================
    tboolean-lv --ip
    === tboolean-lv : tboolean-box
    Python 2.7.15 |Anaconda, Inc.| (default, May  1 2018, 23:32:55) 
    Type "copyright", "credits" or "license" for more information.

    IPython 5.7.0 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.
    [2019-07-25 09:46:58,529] p450817 {<module>            :tboolean.py:30} INFO     - pfx tboolean-box tag 1 src torch det tboolean-box c2max [1.5, 2.0, 2.5] ipython True 
    [2019-07-25 09:46:58,530] p450817 {load                :ab.py     :274} INFO     - [ False 
    [2019-07-25 09:46:58,530] p450817 {__init__            :evt.py    :173} INFO     - [ A 
    [2019-07-25 09:46:58,531] p450817 {__init__            :metadata.py:44} INFO     - path /home/blyth/local/opticks/tmp/tboolean-box/evt/tboolean-box/torch/1/DeltaTime.ini 
    [2019-07-25 09:46:58,551] p450817 {__init__            :evt.py    :233} INFO     - ] A 
    [2019-07-25 09:46:58,551] p450817 {__init__            :evt.py    :173} INFO     - [ B 
    [2019-07-25 09:46:58,552] p450817 {__init__            :metadata.py:44} INFO     - path /home/blyth/local/opticks/tmp/tboolean-box/evt/tboolean-box/torch/-1/DeltaTime.ini 
    [2019-07-25 09:46:58,567] p450817 {__init__            :evt.py    :233} INFO     - ] B 
    [2019-07-25 09:46:58,568] p450817 {load                :ab.py     :308} INFO     - ] 
    [2019-07-25 09:46:58,779] p450817 {compare             :ab.py     :383} INFO     - [
    [2019-07-25 09:46:58,781] p450817 {__init__            :ab.py     :68} INFO     - [ rpost_dv 
    [2019-07-25 09:46:58,842] p450817 {__init__            :ab.py     :70} INFO     - ]
    [2019-07-25 09:46:58,842] p450817 {__init__            :ab.py     :72} INFO     - [ rpol_dv 
    [2019-07-25 09:46:58,860] p450817 {__init__            :ab.py     :74} INFO     - ]
    [2019-07-25 09:46:58,860] p450817 {__init__            :ab.py     :76} INFO     - [ ox_dv 
    [2019-07-25 09:46:58,863] p450817 {__init__            :ab.py     :78} INFO     - ]
    [2019-07-25 09:46:58,863] p450817 {compare             :ab.py     :393} INFO     - ]
    [2019-07-25 09:46:58,866] p450817 {save                :absmry.py :139} INFO     - saving to /home/blyth/local/opticks/tmp/tboolean-box/evt/tboolean-box/torch/1/ABSmry.json
    ...


    In [1]: a.ht
    Out[1]: 
    A(torch,1,tboolean-box)-
    A([[[  58.5492,  170.2793,  450.    ,    3.8635],
        [   0.4733,    0.1947,    0.8591,    1.    ],
        [  -0.6335,    0.7529,    0.1783,  380.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ]],

       [[-191.6261, -240.3635,  450.    ,    5.366 ],
        [  -0.6321,   -0.301 ,    0.714 ,    1.    ],
        [  -0.1712,    0.9529,    0.2502,  380.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ]],
    ...

    In [2]: b.ht         
    Out[2]: 
    A(torch,-1,tboolean-box)-
    A([], shape=(0, 4, 4), dtype=float32)
    ...


Lack of hits from Geant4 side is due to my not having implemented it,  
thats a higher level problem. In anycase the code needs to handle not seeing any hits.

::

    In [3]: from opticks.ana.nbase import count_unique

    In [4]: t = np.array( [1,2,2,3,3,3,4,4,4,4,5,5,5,5,5], dtype=np.uint32 )

    In [5]: count_unique(t)
    Out[5]: 
    array([[1, 1],
           [2, 2],
           [3, 3],
           [4, 4],
           [5, 5]], dtype=uint64)

    In [6]: from opticks.ana.nbase import count_unique_sorted

    In [7]: count_unique_sorted(t)
    Out[7]: 
    array([[5, 5],
           [4, 4],
           [3, 3],
           [2, 2],
           [1, 1]], dtype=uint64)


    In [8]: e = np.array( [], dtype=np.uint32 )

    In [9]: count_unique(e)
    Out[9]: array([], shape=(0, 2), dtype=uint64)

    In [10]: count_unique_sorted(e)
    Out[10]: array([], shape=(0, 2), dtype=uint64)



I added some tests to ana/nbase.py that do similar to the above. Run them as shown below::

    [blyth@localhost ana]$ nbase.py 
    INFO:__main__:np.__version__ 1.14.3 
    INFO:__main__:test_count_unique_2
    INFO:__main__:test_count_unique_sorted
    INFO:__main__:test_count_unique_sorted_empty
    [blyth@localhost ana]$ 

I am expecting the "test_count_unique_sorted_empty" to fail for Sam. 





