Production Performance
========================

With hit saving (for some hit checking, normally no need to save hits to file)

::

    tpmt-;tpmt--  --compute --multievent 10  --production


    2016-09-22 18:39:08.006 INFO  [689834] [TimesTable::dump@103] Opticks::postpropagate filter: NONE

              2.645           2.645          2.645      30482.000      30482.000 : OpticksRun::createEvent.BEG_0
              0.012           2.656          0.012      30485.000          3.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_0
              0.371           3.027          0.371      30657.000        172.000 : OPropagator::prelaunch_0
              0.070           3.098          0.070      30657.000          0.000 : OPropagator::launch_0
              0.027           3.125          0.027      30664.000          7.000 : OEvent::downloadHits_0
              0.023           3.148          0.023      30664.000          0.000 : OpticksRun::saveEvent.END_0
              0.004           3.152          0.004      30664.000          0.000 : OpticksRun::resetEvent.END_0

              0.004           3.156          0.004      30664.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_1
              0.070           3.227          0.070      30664.000          0.000 : OPropagator::launch_1
              0.027           3.254          0.027      30671.000          7.000 : OEvent::downloadHits_1
              0.020           3.273          0.020      30671.000          0.000 : OpticksRun::saveEvent.END_1
              0.004           3.277          0.004      30671.000          0.000 : OpticksRun::resetEvent.END_1

              0.004           3.281          0.004      30671.000          0.000 : OpticksRun::createEvent.END_2
              0.070           3.352          0.070      30671.000          0.000 : OPropagator::launch_2
              0.027           3.379          0.027      30678.000          7.000 : OEvent::downloadHits_2
              0.020           3.398          0.020      30678.000          0.000 : OpticksRun::saveEvent.END_2
              0.008           3.406          0.008      30678.000          0.000 : OpticksRun::resetEvent.END_2

              0.078           3.484          0.078      30678.000          0.000 : OPropagator::launch_3
              0.027           3.512          0.027      30685.000          7.000 : OEvent::downloadHits_3
              0.020           3.531          0.020      30685.000          0.000 : OpticksRun::saveEvent.END_3
              0.008           3.539          0.008      30685.000          0.000 : OpticksRun::resetEvent.END_3

              0.004           3.543          0.004      30685.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_4
              0.074           3.617          0.074      30685.000          0.000 : OPropagator::launch_4
              0.020           3.637          0.020      30692.000          7.000 : OEvent::downloadHits_4
              0.023           3.660          0.023      30692.000          0.000 : OpticksRun::saveEvent.END_4
              0.004           3.664          0.004      30692.000          0.000 : OpticksRun::resetEvent.END_4

              0.004           3.668          0.004      30692.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_5
              0.070           3.738          0.070      30692.000          0.000 : OPropagator::launch_5
              0.023           3.762          0.023      30699.000          7.000 : OEvent::downloadHits_5
              0.020           3.781          0.020      30699.000          0.000 : OpticksRun::saveEvent.END_5
              0.008           3.789          0.008      30699.000          0.000 : OpticksRun::resetEvent.END_5

              0.070           3.859          0.070      30699.000          0.000 : OPropagator::launch_6
              0.031           3.891          0.031      30706.000          7.000 : OEvent::downloadHits_6
              0.020           3.910          0.020      30706.000          0.000 : OpticksRun::saveEvent.END_6
              0.008           3.918          0.008      30706.000          0.000 : OpticksRun::resetEvent.END_6

              0.070           3.988          0.070      30706.000          0.000 : OPropagator::launch_7
              0.027           4.016          0.027      30713.000          7.000 : OEvent::downloadHits_7
              0.023           4.039          0.023      30713.000          0.000 : OpticksRun::saveEvent.END_7
              0.004           4.043          0.004      30713.000          0.000 : OpticksRun::resetEvent.END_7

              0.004           4.047          0.004      30713.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_8
              0.070           4.117          0.070      30713.000          0.000 : OPropagator::launch_8
              0.023           4.141          0.023      30719.000          6.000 : OEvent::downloadHits_8
              0.020           4.160          0.020      30719.000          0.000 : OpticksRun::saveEvent.END_8
              0.008           4.168          0.008      30719.000          0.000 : OpticksRun::resetEvent.END_8

              0.070           4.238          0.070      30719.000          0.000 : OPropagator::launch_9
              0.027           4.266          0.027      30726.000          7.000 : OEvent::downloadHits_9
              0.020           4.285          0.020      30726.000          0.000 : OpticksRun::saveEvent.END_9
              0.008           4.293          0.008      30726.000          0.000 : OpticksRun::resetEvent.END_9
    2016-09-22 18:39:08.007 INFO  [689834] [OpticksProfile::dump@145]  npy 174,1,4 /tmp/blyth/opticks/evt/PmtInBox/torch/Opticks.npy
    2016-09-22 18:39:08.007 INFO  [689834] [OpticksProfile::dump@140] Opticks::postpropagate dir /tmp/blyth/opticks/evt/PmtInBox/torch name Opticks.npy num_stamp 174
    2016-09-22 18:39:08.007 INFO  [689834] [TimesTable::dump@103] Opticks::postpropagate filter: OPropagator::launch
              3.098           3.098          0.070      30657.000          0.000 : OPropagator::launch_0
              0.129           3.227          0.070      30664.000          0.000 : OPropagator::launch_1
              0.125           3.352          0.070      30671.000          0.000 : OPropagator::launch_2
              0.133           3.484          0.078      30678.000          0.000 : OPropagator::launch_3
              0.133           3.617          0.074      30685.000          0.000 : OPropagator::launch_4
              0.121           3.738          0.070      30692.000          0.000 : OPropagator::launch_5
              0.121           3.859          0.070      30699.000          0.000 : OPropagator::launch_6
              0.129           3.988          0.070      30706.000          0.000 : OPropagator::launch_7
              0.129           4.117          0.070      30713.000          0.000 : OPropagator::launch_8
              0.121           4.238          0.070      30719.000          0.000 : OPropagator::launch_9
    2016-09-22 18:39:08.008 INFO  [689834] [OpticksProfile::dump@145]  npy 174,1,4 /tmp/blyth/opticks/evt/PmtInBox/torch/Opticks.npy
    Opticks::postpropagate
       OptiXVersion :            3080
    /Users/blyth/opticks/bin/op.sh RC 0



Hit only evt needs some changes::

    ipython -i $(which tevt.py) -- --tag 10

    tevt.py --multievent 10 --terse     ## dump just mask tables of the saved hits, have to manully scrub folder ahead to get just hits



