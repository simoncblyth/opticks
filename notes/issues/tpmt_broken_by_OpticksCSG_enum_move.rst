tpmt broken by OpticksCSG enum move
======================================

* shape/operator enum unification to use sysrap/OpticksCSG.{h,py} is incomplete
* tpmt broken due to mis-interpretation of part buffer


overview
----------

* PMT serialization ignoring new enum   



symptom
--------

::

    tpmt--   

    2017-03-15 20:48:44.712 INFO  [829428] [OContext::close@219] OContext::close numEntryPoint 2
    ##hemi-pmt.cu:bounds primIdx 0 is_partlist:0 min  -101.1682  -101.1682   -23.8382 max   101.1682   101.1682    56.0000 
    ##hemi-pmt.cu:bounds primIdx 1 is_partlist:0 min   -98.1428   -98.1428    56.0000 max    98.1428    98.1428    98.0465 
    ##hemi-pmt.cu:bounds primIdx 2 is_partlist:0 min   -98.0932   -98.0932    55.9934 max    98.0932    98.0932    98.0128 
    ##hemi-pmt.cu:bounds primIdx 3 is_partlist:0 min   -27.5000   -27.5000  -164.5000 max    27.5000    27.5000     1.5000 
    ##hemi-pmt.cu:bounds primIdx 4 is_partlist:0 min  -300.0100  -300.0100  -300.0100 max   300.0100   300.0100   300.0100 
    2017-03-15 20:48:45.342 INFO  [829428] [OPropagator::prelaunch@149] 1 : (0;500000,1) prelaunch_times vali,comp,prel,lnch  0.0000 0.2694 0.2364 0.0000
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit
    evaluative_csg primIdx_ 1 numParts 4 perfect tree fullHeight 4294967295 exceeds current limit

review of analytic PMT serialization
--------------------------------------

* ana/pmt/analytic.py 

Recreate the analytic PMT from detdecs parse with

::

   pmt-analytic-tmp   # writing to $TMP/GPmt/0/GPmt.npy
   pmt-analytic       # writing to $IDPATH/GPmt/0/GPmt.npy

Actual one in use is from opticksdata repo $OPTICKS_DATA/export/DayaBay/GPmt/0/  


Comparing existing serializations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three look effectively the same, with no influence from new enum so far::

    simon:pmt blyth$ l /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GPmt/0/
    total 48
    -rw-r--r--  1 blyth  staff   848 Mar 15 16:27 GPmt.npy
    -rw-r--r--  1 blyth  staff   289 Mar 15 16:27 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  staff  1168 Mar 15 16:27 GPmt_csg.npy
    -rw-r--r--  1 blyth  staff    74 Mar 15 16:27 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  staff    47 Mar 15 16:27 GPmt_materials.txt
    -rw-r--r--  1 blyth  staff    74 Mar 15 16:27 GPmt_pvnames.txt
    simon:pmt blyth$ 
    simon:pmt blyth$ 
    simon:pmt blyth$ l $TMP/GPmt/0/
    total 48
    -rw-r--r--  1 blyth  wheel   848 Mar 15 17:31 GPmt.npy
    -rw-r--r--  1 blyth  wheel   289 Mar 15 17:31 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  wheel  1168 Mar 15 17:31 GPmt_csg.npy
    -rw-r--r--  1 blyth  wheel    74 Mar 15 17:31 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  wheel    47 Mar 15 17:31 GPmt_materials.txt
    -rw-r--r--  1 blyth  wheel    74 Mar 15 17:31 GPmt_pvnames.txt
    simon:pmt blyth$ diff -r --brief $IDPATH/GPmt/0 $TMP/GPmt/0
    simon:pmt blyth$ 
    simon:pmt blyth$ 
    simon:pmt blyth$ l /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/
    total 80
    -rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt.npy
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt.txt
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_boundaries.txt
    -rw-r--r--  1 blyth  staff   848 Jul  5  2016 GPmt_check.npy
    -rw-r--r--  1 blyth  staff   289 Jul  5  2016 GPmt_check.txt
    -rw-r--r--  1 blyth  staff  1168 Jul  5  2016 GPmt_csg.npy
    -rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_csg.txt
    -rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_lvnames.txt
    -rw-r--r--  1 blyth  staff    47 Jul  5  2016 GPmt_materials.txt
    -rw-r--r--  1 blyth  staff    74 Jul  5  2016 GPmt_pvnames.txt

    simon:pmt blyth$ echo $OPTICKS_DATA
    /usr/local/opticks/opticksdata
    simon:pmt blyth$ 
    simon:pmt blyth$ diff -r --brief $OPTICKS_DATA/export/DayaBay/GPmt/0/ $TMP/GPmt/0/
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt.txt
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt_check.npy
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt_check.txt
    Only in /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0/: GPmt_csg.txt





