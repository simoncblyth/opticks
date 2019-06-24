tboolean-g4-TO-NA-MI
================================

::

   ts truncate
   ta truncate
   tv truncate

   ts cone   # similar "TO NA MI" problem


ISSUES
-----------

1. OpticksEventAna::checkPointExcursions found some
2. All G4 photons are  "TO NA MI"  NA:NAN-ABORT MI:MISS


ts truncate::

    ...
    2019-06-24 16:43:57.541 INFO  [31414] [OpticksEvent::makeReport@1688] tagdir /tmp/blyth/opticks/tboolean-truncate/evt/tboolean-truncate/torch/1
    2019-06-24 16:43:57.545 INFO  [31414] [GGeoTest::anaEvent@804]  dbgnode -1 numTrees 1 evt 0xa66cb60
    2019-06-24 16:43:57.545 INFO  [31414] [OpticksEvent::getTestConfig@706]  gtc autoseqmap=TO:0,SR:1,SA:0_name=tboolean-truncate_outerfirst=1_analytic=1_csgpath=/tmp/blyth/opticks/tboolean-truncate_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x3f,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75_autocontainer=Rock//perfectAbsorbSurface/Vacuum
    2019-06-24 16:43:57.546 INFO  [31414] [OpticksEventInstrument::CreateRecordsNPY@36] OpticksEventInstrument::CreateRecordsNPY  shape 100000,10,2,4
    2019-06-24 16:43:57.546 INFO  [31414] [OpticksEventAna::initOverride@67]  autoseqmap TO:0,SR:1,SA:0
    2019-06-24 16:43:57.549 INFO  [31414] [OpticksEventAna::checkPointExcursions@108]  seqmap TO:0,SR:1,SA:0 seqmap_his              8ad seqmap_val              121
     p  0 abbrev TO val1  1 tree 0 count 0 dist (      0.000     0.000     0.000       0.000) xdist (     -0.100    -0.100    -0.100      -0.100) df 0.1000000015 expected
     p  1 abbrev SR val1  2 tree 1 count 3276275712 dist (   -200.000   200.000  -150.000    -200.000) xdist (      0.000     0.000     0.000       0.000) df 200.0000000000 EXCURSION
     p  2 abbrev SA val1  1 tree 0 count 0 dist (      0.000     0.000     0.000       0.000) xdist (      0.000     0.000     0.000       0.000) df 0.0000000000 expected
    2019-06-24 16:43:57.549 FATAL [31414] [OpticksEventAna::checkPointExcursions@157]  num_excursions 1
    2019-06-24 16:43:57.549 FATAL [31414] [Opticks::dumpRC@202]  rc 202 rcmsg : OpticksEventAna::checkPointExcursions found some
    2019-06-24 16:43:57.549 INFO  [31414] [OpticksEventAna::dump@232] GGeoTest::anaEvent OpticksEventAna pho 100000,4,4 seq 100000,1,2
    2019-06-24 16:43:57.549 INFO  [31414] [GGeoTest::anaEvent@804]  dbgnode -1 numTrees 1 evt 0xac2c230
    2019-06-24 16:43:57.549 INFO  [31414] [OpticksEvent::getTestConfig@706]  gtc autoseqmap=TO:0,SR:1,SA:0_name=tboolean-truncate_outerfirst=1_analytic=1_csgpath=/tmp/blyth/opticks/tboolean-truncate_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x3f,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75_autocontainer=Rock//perfectAbsorbSurface/Vacuum
    2019-06-24 16:43:57.550 INFO  [31414] [OpticksEventInstrument::CreateRecordsNPY@36] OpticksEventInstrument::CreateRecordsNPY  shape 100000,10,2,4
    2019-06-24 16:43:57.550 INFO  [31414] [OpticksEventAna::initOverride@67]  autoseqmap TO:0,SR:1,SA:0
    2019-06-24 16:43:57.553 INFO  [31414] [OpticksEventAna::checkPointExcursions@108]  seqmap TO:0,SR:1,SA:0 seqmap_his              8ad seqmap_val              121
     p  0 abbrev TO val1  1 tree 0 count 0 dist (      0.000     0.000     0.000       0.000) xdist (     -0.100    -0.100    -0.100      -0.100) df 0.1000000015 expected
     p  1 abbrev SR val1  2 tree 1 count 3276275712 dist (   -200.000   200.000  -150.000    -200.000) xdist (      0.000     0.000     0.000       0.000) df 200.0000000000 EXCURSION
     p  2 abbrev SA val1  1 tree 0 count 0 dist (      0.000     0.000     0.000       0.000) xdist (      0.000     0.000     0.000       0.000) df 0.0000000000 expected
    2019-06-24 16:43:57.553 FATAL [31414] [OpticksEventAna::checkPointExcursions@157]  num_excursions 1
    2019-06-24 16:43:57.553 FATAL [31414] [Opticks::dumpRC@202]  rc 202 rcmsg : OpticksEventAna::checkPointExcursions found some
    2019-06-24 16:43:57.553 INFO  [31414] [OpticksEventAna::dump@232] GGeoTest::anaEvent OpticksEventAna pho 100000,4,4 seq 100000,1,2
    2019-06-24 16:43:57.553 INFO  [31414] [OpticksAna::run@70]  anakey tboolean enabled Y

    args: /home/blyth/opticks/ana/tboolean.py --tagoffset 0 --tag 1 --det tboolean-truncate --pfx tboolean-truncate --src torch
    [2019-06-24 16:43:58,404] p31868 {tboolean.py:63} INFO     - pfx tboolean-truncate tag 1 src torch det tboolean-truncate c2max [1.5, 2.0, 2.5] ipython False 
    [2019-06-24 16:43:58,412] p31868 {evt.py    :446} WARNING  -  x : -200.000 200.000 : tot 100000 over 10 0.000  under 10 0.000 : mi   -200.000 mx    200.000  
    [2019-06-24 16:43:58,413] p31868 {evt.py    :446} WARNING  -  y : -200.000 200.000 : tot 100000 over 1 0.000  under 5 0.000 : mi   -200.000 mx    200.000  
    [2019-06-24 16:43:58,415] p31868 {evt.py    :446} WARNING  -  z : -200.000 200.000 : tot 100000 over 11 0.000  under 14 0.000 : mi   -200.000 mx    200.000  
    [2019-06-24 16:43:58,416] p31868 {evt.py    :446} WARNING  -  t :   0.000   4.000 : tot 100000 over 100000 1.000  under 0 0.000 : mi      6.181 mx     12.239  
    [2019-06-24 16:43:58,458] p31868 {evt.py    :596} WARNING  - init_records tboolean-truncate/tboolean-truncate/torch/  1 :  finds too few (ph)seqmat uniques : 1 : EMPTY HISTORY
    ab.cfm
    nph:  100000 A:    0.0117 B:    9.3984 B/A:     802.0 INTEROP_MODE ALIGN non-reflectcheat 
    ab.a.metadata:/tmp/blyth/opticks/tboolean-truncate/evt/tboolean-truncate/torch/1 ox:c38f1bd703797b74e0396028b7912809 rx:9c8e93970c6237f9ca465d276eb38933 np: 100000 pr:    0.0117 INTEROP_MODE
    ab.b.metadata:/tmp/blyth/opticks/tboolean-truncate/evt/tboolean-truncate/torch/-1 ox:c1612472d50a26b8d5f2b0bf2d6d526c rx:7d8577bba5bc33b2311aa65a800cc21e np: 100000 pr:    9.3984 INTEROP_MODE
    WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_ALIGN_DEV_DEBUG WITH_LOGDOUBLE 
    {u'containerscale': 3.0, u'ctrl': 0, u'verbosity': 0, u'poly': u'IM', u'jsonLoadPath': u'/tmp/blyth/opticks/tboolean-truncate/0/meta.json', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1', u'resolution': 20, u'emit': -1}
    .
    ab.mal
    aligned        0/ 100000 : 0.0000 :  
    maligned  100000/ 100000 : 1.0000 : 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 
    slice(0, 25, None)
          0      0 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          1      1 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          2      2 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          3      3 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          4      4 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          5      5 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          6      6 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          7      7 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          8      8 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
          9      9 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
         10     10 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
         11     11 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 
         12     12 : * :                      TO SR SR SR SR SR SR SR SR SR                                           TO NA MI 

