py-analysis-too-slow-for-big-events  FIXED 
==================================================

Context
----------

* :doc:`lifting-the-3M-photon-limitation`


Reproduce
-----------

::


    ts box --generateoverride -1    # 1M
    ta box



ISSUE : changes make working with 1M fast enough, but 10M and its too slow again : AVOIDED BY MMAP_MODE SLICED ARRAY LOAD
-------------------------------------------------------------------------------------------------------------------------------

DONE: restrict analysis to configurable slice using --msli option 

* uses np.load with mmap_mode="r"
* this avoids slow loading of very big arrays 

* https://stackoverflow.com/questions/34540585/how-to-partial-load-an-array-saved-with-numpy-save-in-python

::

    In [8]: a = np.arange(1000000)

    In [9]: np.save("a.npy", a)

    In [10]: a2 = np.load("a.npy", mmap_mode="r")

    In [11]: a2[:10]
    Out[11]: memmap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    In [12]: a2[-10:]
    Out[12]: memmap([999990, 999991, 999992, 999993, 999994, 999995, 999996, 999997, 999998, 999999])




FIXED : by rethinking how to do deviation checking with large numbers of photons
-------------------------------------------------------------------------------------

* fix replaces ana/dv.py with ana/qdv.py 

* Looping over single line selections just to give the same shape (number
  of record points) is too slow once going beyond 100k photons. 

* So instead invert loop order, first calculate the deviations for 
  all photons using a simple subtraction.  
  The empty record points just give zero, so they cause no problem.

* Then aggregate again the per photon max deviations into per history max deviations
  for presentation.    


::

    In [8]: dv = np.abs(a.rposta - b.rposta)    ## could also split pos and t at this juncture

    In [16]: dv.shape
    Out[16]: (100000, 10, 4)

    In [17]: dv.max(0)      ## collapse dimension 0, the photons    
    Out[17]: 
    A()sliced
    A([[0.0138, 0.0138, 0.    , 0.    ],
       [0.0138, 0.0138, 0.    , 0.    ],
       [0.0138, 0.0138, 0.    , 0.    ],
       [0.0138, 0.0138, 0.    , 0.    ],
       [0.    , 0.0138, 0.    , 0.    ],
       [0.    , 0.    , 0.    , 0.    ],
       [0.    , 0.    , 0.    , 0.    ],
       [0.    , 0.    , 0.    , 0.0003],
       [0.    , 0.    , 0.0138, 0.    ],
       [0.    , 0.    , 0.    , 0.    ]])


    ## this is fine when agreeing, but normally start with disagreement
    ## and need to chase photons

    In [24]: dv.max(axis=(1,2)).shape     ## collapse dimensions 1,2 ie aggregate over all rpost for single photons
    Out[24]: (100000,)

    In [25]: dv.max(axis=(1,2)).max()
    Out[25]: 
    A(0.0138)

    In [27]: np.unique(dv.max(axis=(1,2)))
    Out[27]: 
    A()sliced
    A([0.    , 0.0003, 0.0138, 0.0138])


    In [6]: a.phosel
    Out[6]: 
    A([1, 1, 2, ..., 1, 1, 1], dtype=uint8)

    In [7]: a.phosel.shape
    Out[7]: (100000,)

    In [8]: np.where(a.phosel == 1)
    Out[8]: (array([    0,     1,     3, ..., 99997, 99998, 99999]),)

    In [9]: np.where(a.phosel == 1)[0].shape
    Out[9]: (87782,)




Huh::

    In [12]: np.where( a.phosel != b.phosel )
    Out[12]: (array([  595,  1230,  9041, 14510, 18921, 25113, 30272, 45629, 58189, 58609, 64663, 65850, 69653, 76467, 77962, 90322, 92353, 97887]),)

    In [13]: np.where( a.seqhis != b.seqhis )
    Out[13]: (array([], dtype=int64),)

Ahhh yes, the category orders will not be the same between events in the category tail.  Hence it
is better to use the absolute seqhis history approach.


::

    In [26]: np.all( np.where( a.phosel == 2)[0] == np.where( a.seqhis == 2237)[0] )
    Out[26]: True

    In [27]: np.all( np.where( a.phosel == 1)[0] == np.where( a.seqhis == 36045)[0] )
    Out[27]: True




ISSUE: 1M+ running : py analysis too slow for comfort : mostly from deviation comparisons for every line selection 
-------------------------------------------------------------------------------------------------------------------

* https://pypi.org/project/memory-profiler/
* https://medium.com/zendesk-engineering/hunting-for-memory-leaks-in-python-applications-6824d0518774


The dv for each sel is whats taking the time

* given that the tail of the sel has very few entries, this is kinda surprising

  * NOT really : the psel is still a boolean mask over all photons even with few entries


Obvious way to improve it is to make all psel selected arrays lazily 
provided. So can then quickly switch selection without incurring penalties
until actually access the data.   This makes lots of sense for rpost, rpol deviations
where can avoid reselecting everything when just want to see eg rpost in 
different selections.  This might give a factor of 5.


::

    args: /home/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 100 --det tboolean-box --pfx tboolean-box --src torch
    [2019-07-09 22:57:30,728] p248164 {<module>            :tboolean.py:63} INFO     - pfx tboolean-box tag 100 src torch det tboolean-box c2max [1.5, 2.0, 2.5] ipython False 
    [2019-07-09 22:57:30,728] p248164 {__init__            :ab.py     :171} INFO     - [
    [2019-07-09 22:57:31,244] p248164 {check_ox_fdom       :evt.py    :446} WARNING  -  t :   0.000   9.020 : tot 4000000 over 42 0.000  under 0 0.000 : mi      0.021 mx     11.205  
    [2019-07-09 22:57:36,688] p248164 {check_ox_fdom       :evt.py    :446} WARNING  -  t :   0.000   9.020 : tot 4000000 over 41 0.000  under 0 0.000 : mi      0.021 mx     11.205  
    [2019-07-09 22:57:43,011] p248164 {check_alignment     :ab.py     :264} INFO     - [
    [2019-07-09 22:57:43,080] p248164 {check_alignment     :ab.py     :266} INFO     - ]
    [2019-07-09 22:57:43,081] p248164 {compare             :ab.py     :270} INFO     - [
    [2019-07-09 22:57:43,081] p248164 {_get_cf             :ab.py     :492} INFO     - [ ab.ahis 
    [2019-07-09 22:57:43,088] p248164 {_get_cf             :ab.py     :501} INFO     - ] ab.ahis 
    [2019-07-09 22:57:43,088] p248164 {_get_cf             :ab.py     :492} INFO     - [ ab.amat 
    [2019-07-09 22:57:43,091] p248164 {_get_cf             :ab.py     :501} INFO     - ] ab.amat 
    [2019-07-09 22:57:43,091] p248164 {__init__            :ab.py     :58} INFO     - [
    [2019-07-09 22:57:43,091] p248164 {_make_dv            :ab.py     :413} INFO     - [ rpost_dv 
    [2019-07-09 22:57:43,092] p248164 {__init__            :dv.py     :278} INFO     - [ rpost_dv 
    [2019-07-09 22:57:54,083] p248164 {dv_                 :dv.py     :400} INFO     - [
    [2019-07-09 22:57:56,533] p248164 {dv_                 :dv.py     :421} INFO     - ]
    [2019-07-09 22:58:02,638] p248164 {dv_                 :dv.py     :400} INFO     - [
    [2019-07-09 22:58:02,775] p248164 {dv_                 :dv.py     :421} INFO     - ]
    [2019-07-09 22:58:07,792] p248164 {dv_                 :dv.py     :400} INFO     - [
    ...
    [2019-07-09 23:01:55,006] p248164 {dv_                 :dv.py     :421} INFO     - ]
    [2019-07-09 23:01:58,702] p248164 {dv_                 :dv.py     :400} INFO     - [
    [2019-07-09 23:01:58,703] p248164 {dv_                 :dv.py     :421} INFO     - ]
    [2019-07-09 23:02:01,755] p248164 {dv_                 :dv.py     :400} INFO     - [
    [2019-07-09 23:02:01,759] p248164 {dv_                 :dv.py     :421} INFO     - ]
    [2019-07-09 23:02:05,486] p248164 {__init__            :dv.py     :322} INFO     - ] rpost_dv 
    [2019-07-09 23:02:05,487] p248164 {_make_dv            :ab.py     :422} INFO     - ] rpost_dv 
    [2019-07-09 23:02:05,487] p248164 {_make_dv            :ab.py     :413} INFO     - [ rpol_dv 
    [2019-07-09 23:02:05,487] p248164 {__init__            :dv.py     :278} INFO     - [ rpol_dv 
    [2019-07-09 23:02:12,621] p248164 {dv_                 :dv.py     :400} INFO     - [
    [2019-07-09 23:02:14,004] p248164 {dv_                 :dv.py     :421} INFO     - ]
    [2019-07-09 23:02:18,832] p248164 {dv_                 :dv.py     :400} INFO     - [
    [2019-07-09 23:02:18,879] p248164 {dv_                 :dv.py     :421} INFO     - ]
     ...
    [2019-07-09 23:03:35,286] p248164 {dv_                 :dv.py     :421} INFO     - ]
    [2019-07-09 23:03:38,205] p248164 {dv_                 :dv.py     :400} INFO     - [
    [2019-07-09 23:03:38,205] p248164 {dv_                 :dv.py     :421} INFO     - ]
    [2019-07-09 23:03:41,383] p248164 {dv_                 :dv.py     :400} INFO     - [
    [2019-07-09 23:03:41,384] p248164 {dv_                 :dv.py     :421} INFO     - ]



Interrupt profiling, suggests _init_selection is taking the time::


    /home/blyth/opticks/ana/ab.pyc in _set_aselhis(self, sel)
        576         self._set_sel( sel, nom="selhis")
        577     def _set_aselhis(self, sel):
    --> 578         self._set_sel( sel, nom="aselhis")
        579     def _set_selflg(self, sel):
        580         self._set_sel( sel, nom="selflg")

    /home/blyth/opticks/ana/ab.pyc in _set_sel(self, sel, nom)
        552             self.align = "seqhis"
        553             self.a.selhis = sel
    --> 554             self.b.selhis = sel
        555         elif nom == "selmat":
        556             self.align = None

    /home/blyth/opticks/ana/evt.pyc in _set_selhis(self, arg)
       1014     def _set_selhis(self, arg):
       1015         self.flv = "seqhis"
    -> 1016         self.sel = arg
       1017     selhis = property(_get_sel, _set_selhis)
       1018 

    /home/blyth/opticks/ana/evt.pyc in _set_sel(self, arg)
       1001 
       1002         psel = self.make_selection(sel, False)
    -> 1003         self._init_selection(psel)
       1004     sel = property(_get_sel, _set_sel)
       1005 

    /home/blyth/opticks/ana/evt.pyc in _init_selection(self, psel)
        912         self.c4 = self.c4_[psel]
        913         self.wl = self.wl_[psel]
    --> 914         self.rx = self.rx_[psel]
        915 
        916         if not self.so_.missing:

    KeyboardInterrupt: 
    > /home/blyth/opticks/ana/evt.py(914)_init_selection()
        912         self.c4 = self.c4_[psel]
        913         self.wl = self.wl_[psel]
    --> 914         self.rx = self.rx_[psel]
        915 
        916         if not self.so_.missing:

    ipdb> p psel
    A()sliced
    A([ True,  True,  True, ...,  True,  True,  True])
    ipdb> p psel.shape
    (1000000,)
    ipdb> 


::

   LV=box python -m cProfile -o tboolean.cProfile tboolean.py 
   # huh file contains gibberish 


* https://docs.python.org/2/library/profile.html

ncalls
    for the number of calls,
tottime
    for the total time spent in the given function (and excluding time made in calls to sub-functions)
percall
    is the quotient of tottime divided by ncalls
cumtime
    is the cumulative time spent in this and all subfunctions (from invocation till exit). This figure is accurate even for recursive functions.
percall
    is the quotient of cumtime divided by primitive calls
filename:lineno(function)
    provides the respective data of each function


::

    [blyth@localhost ana]$ LV=box python -m cProfile -s time tboolean.py
    ...
    2019-07-11 21:21:32,372] p232641 {<module>            :tboolean.py:75} INFO     - early exit as non-interactive
             1275267 function calls (1234686 primitive calls) in 22.170 seconds

       Ordered by: internal time

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
          130   13.797    0.106   18.254    0.140 evt.py:878(_init_selection)
          407    3.387    0.008    3.387    0.008 {method 'sort' of 'numpy.ndarray' objects}
          126    0.440    0.003    0.448    0.004 records.py:504(__getitem__)
          411    0.386    0.001    0.395    0.001 seq.py:70(seq2msk)
           20    0.367    0.018    0.446    0.022 evt.py:1573(rpost_)
         1585    0.359    0.000    0.359    0.000 {method 'astype' of 'numpy.ndarray' objects}
          408    0.273    0.001    0.273    0.001 {method 'flatten' of 'numpy.ndarray' objects}
            2    0.253    0.126    0.289    0.145 evt.py:614(init_npoint)
           20    0.245    0.012    0.245    0.012 {numpy.core.multiarray.fromfile}
           30    0.216    0.007    0.282    0.009 dv.py:141(__init__)
          408    0.180    0.000    3.977    0.010 arraysetops.py:256(_unique1d)
           65    0.166    0.003    0.166    0.003 ab.py:592(_set_align)
          651    0.159    0.000    0.159    0.000 {method 'reduce' of 'numpy.ufunc' objects}
           64    0.089    0.001    0.089    0.001 {method 'copy' of 'numpy.ndarray' objects}
           60    0.087    0.001    0.089    0.001 seq.py:579(<lambda>)
            2    0.078    0.039    0.937    0.468 evt.py:537(init_sequence)
            2    0.072    0.036    0.138    0.069 evt.py:413(check_ox_fdom)
          403    0.064    0.000    0.064    0.000 {method 'nonzero' of 'numpy.ndarray' objects}
         1340    0.063    0.000    0.063    0.000 {numpy.core.multiarray.concatenate}
         6194    0.057    0.000    0.099    0.000 seq.py:373(line)
           20    0.051    0.003    0.164    0.008 evt.py:1403(rpolw_)
    20680/2153    0.048    0.000    0.342    0.000 {map}
            1    0.042    0.042    0.042    0.042 qt_compat.py:2(<module>)
           73    0.036    0.000    0.036    0.000 {numpy.core.multiarray.where}
     2871/646    0.030    0.000    0.083    0.000 sre_parse.py:414(_parse)
          404    0.028    0.000    4.530    0.011 seq.py:530(__init__)
            1    0.026    0.026    0.070    0.070 backend_qt5.py:1(<module>)
           82    0.022    0.000    0.027    0.000 collections.py:305(namedtuple)
            2    0.021    0.011    0.033    0.016 __init__.py:27(<module>)
          409    0.021    0.000    0.323    0.001 seq.py:251(__init__)
     4449/599    0.019    0.000    0.048    0.000 sre_compile.py:64(_compile)
          297    0.019    0.000    0.051    0.000 doccer.py:12(docformat)
        28339    0.018    0.000    0.021    0.000 sre_parse.py:194(__next)
         8369    0.017    0.000    0.030    0.000 {filter}
          284    0.015    0.000    0.015    0.000 {method 'read' of 'file' objects}
        34828    0.014    0.000    0.017    0.000 seq.py:226(<lambda>)
         5568    0.013    0.000    0.068    0.000 seq.py:178(label)
       146848    0.013    0.000    0.013    0.000 {method 'append' of 'list' objects}
    58021/57454    0.013    0.000    0.021    0.000 {isinstance}
           37    0.012    0.000    0.169    0.005 __init__.py:1(<module>)
          126    0.012    0.000    0.129    0.001 evt.py:707(make_selection_)
    157128/155373    0.012    0.000    0.012    0.000 {len}
            1    0.012    0.012    0.012    0.012 extensions.py:25(ExtensionManager)
            1    0.011    0.011    0.011    0.011 {posix.read}
         1181    0.011    0.000    0.016    0.000 sre_compile.py:256(_optimize_charset)
        34828    0.010    0.000    0.010    0.000 seq.py:223(<lambda>)
    5519/1752    0.009    0.000    0.011    0.000 sre_parse.py:152(getwidth)
         7642    0.009    0.000    0.009    0.000 {method 'expandtabs' of 'str' objects}
         1273    0.008    0.000    0.009    0.000 {method 'sub' of '_sre.SRE_Pattern' objects}
    24816/22726    0.008    0.000    0.023    0.000 {method 'join' of 'str' objects}
          574    0.008    0.000    0.008    0.000 {method 'search' of '_sre.SRE_Pattern' objects}
        19445    0.008    0.000    0.008    0.000 {method 'split' of 'str' objects}
          403    0.008    0.000    0.009    0.000 function_base.py:1851(diff)
          404    0.008    0.000    4.186    0.010 nbase.py:97(count_unique_sorted)
          262    0.007    0.000    0.007    0.000 {method 'split' of '_sre.SRE_Pattern' objects}
          293    0.007    0.000    0.011    0.000 doccer.py:172(indentcount_lines)

