profiling_opticks_running_with_SProf
=====================================


Throughout Opticks code there are lines like the below using the static ``SProf::Add`` method from sysrap/SProf.hh::

    479     int64_t t_LBEG = SProf::Add("QSim__simulate_LBEG");


For example see ``QSim::simulate``

* https://github.com/simoncblyth/opticks/blob/master/qudarap/QSim.cc


When the below envvar is set::

   export SProf__WRITE=1   # enable the time and memory profiling

Running Opticks writes an SProf.txt file in the invoking directory, with contents like::

    A[blyth@localhost detsim]$ cat /data1/blyth/tmp/j/zhenning_double_muon/detsim/OJ_LOCAL_Dec04_ok1_hit_seed42_evtmax1_run/SProf.txt
    SEvt__Init_RUN_META:1764835645334942,1049192,505280
    CSGOptiX__Create_HEAD:1764835676090415,6547500,1401592
    CSGOptiX__Create_TAIL:1764835677382106,8713792,2002452
    junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_HEAD:1764835751800461,11365936,4651516
    A000_QSim__simulate_HEAD:1764835751800530,11365936,4651516
    A000_SEvt__BeginOfRun:1764835751800565,11365936,4651516
    A000_SEvt__beginOfEvent_FIRST_EGPU:1764835751800685,11365936,4651516
    A000_SEvt__setIndex:1764835751800711,11365936,4651516
    A000_QSim__simulate_LBEG:1764835751906377,11420680,4706016
    A000_QSim__simulate_PRUP:1764835751906414,11420680,4706016
    A000_QSim__simulate_PREL:1764835751969031,29475848,4706912
    A000_QSim__simulate_POST:1764835774965272,29475848,4711840
    A000_QSim__simulate_DOWN:1764835776914594,31192844,6430056
    A000_QSim__simulate_LEND:1764835776914640,31192844,6430056
    A000_QSim__simulate_PCAT:1764835776914663,31192844,6430056
    A000_QSim__simulate_BRES:1764835776914707,31192844,6430056 # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928
    A000_QSim__simulate_TAIL:1764835776914712,31192844,6430056
    A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_HEAD:1764835776914892,31192844,6430056
    A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_cpumerged_HEAD:1764835776914956,31192844,6430056
    A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_cpumerged_TAIL:1764835967360544,32940884,8163292 # hit=27471928,merged=21062899,save=6409029
    A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_TAIL:1764835967360611,32940884,8163292
    A000_junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_TAIL:1764835967360619,32940884,8163292
    A000_QSim__reset_HEAD:1764835967360999,32940884,8163292
    A000_SEvt__endIndex:1764835967361333,32886140,8108636
    A000_SEvt__EndOfRun:1764835967365389,31169144,6391640


The label before the ":" identifies locations in the code where ``SProf::Add`` was called
and the 16 digit numbers are microseconds since epoch timestamps. The other numbers are vm and rss memory in kb.
Opticks has bash and python scripts such as ``SProf.sh`` that find and parse multiple of these ``SProf.txt``
from different runs and presents results, such as::


    A[blyth@localhost opticks]$ cd /data1/blyth/tmp/j/zhenning_double_muon/detsim/
    A[blyth@localhost detsim]$ T0="2025-12-04 16:00" T1="2025-12-04 18:00" SProf.sh proi
    Identity                                             Start(s)  Duration(s)     Sub PREL→POST(s)     Sub POST→DOWN(s)    Sub TAIL→RESET(s)    VM(MB)  RSS(MB)  Comment
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    hit/A000_QSim                                         106.466   215.560469            22.996241             1.949322           190.446287   32168.83  7971.96  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928
    hitlite/A000_QSim                                     106.462   170.226838            23.108319             0.484299           146.471936   32499.61  5361.86  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928
    hitlitemerged/A000_QSim                               107.263    23.835564            23.097717             0.181023             0.404235   87284.29  4988.84  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=6409029
    hitmerged/A000_QSim                                   106.770    30.400825            22.988498             0.543559             6.713401   88292.85  6634.07  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=6409029  # slice=1,max_slot_M=150



The above numbers are close to those presented on https://simoncblyth.github.io/env/presentation/opticks_20260122_wuhan.html?p=8
The above commandline is using installed scripts accessed from from the PATH.  It is also possible to use the scripts
from the source tree eg: ``~/opticks/bin/SProf.sh`` (o is symbolic link).



To dump details of the profiling::

    A[blyth@localhost detsim]$ REPORT=1 T0="2025-12-04 16:00" T1="2025-12-04 18:00" SProf.sh prof
    → /data1/blyth/tmp/j/zhenning_double_muon/detsim/OJ_LOCAL_Dec04_ok1_hit_seed42_evtmax1_run/SProf.txt
      Delta(s)   Δprev(s) DateTime (+08)        VM(MB)  RSS(MB) Label / Comment
    --------------------------------------------------------------------------------------------
      0.000000   0.000000 2025-12-04 16:07:25.334  1024.60   493.44 SEvt__Init_RUN_META
     30.755473  30.755473 2025-12-04 16:07:56.090  6394.04  1368.74 CSGOptiX__Create_HEAD
     32.047164   1.291691 2025-12-04 16:07:57.382  8509.56  1955.52 CSGOptiX__Create_TAIL
    106.465519  74.418355 2025-12-04 16:09:11.800 11099.55  4542.50 junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_HEAD
    106.465588   0.000069 2025-12-04 16:09:11.800 11099.55  4542.50 A000_QSim__simulate_HEAD
    106.465623   0.000035 2025-12-04 16:09:11.800 11099.55  4542.50 A000_SEvt__BeginOfRun
    106.465743   0.000120 2025-12-04 16:09:11.800 11099.55  4542.50 A000_SEvt__beginOfEvent_FIRST_EGPU
    106.465769   0.000026 2025-12-04 16:09:11.800 11099.55  4542.50 A000_SEvt__setIndex
    106.571435   0.105666 2025-12-04 16:09:11.906 11153.01  4595.72 A000_QSim__simulate_LBEG
    106.571472   0.000037 2025-12-04 16:09:11.906 11153.01  4595.72 A000_QSim__simulate_PRUP
    106.634089   0.062617 2025-12-04 16:09:11.969 28785.01  4596.59 A000_QSim__simulate_PREL
    129.630330  22.996241 2025-12-04 16:09:34.965 28785.01  4601.41 A000_QSim__simulate_POST
    131.579652   1.949322 2025-12-04 16:09:36.914 30461.76  6279.35 A000_QSim__simulate_DOWN
    131.579698   0.000046 2025-12-04 16:09:36.914 30461.76  6279.35 A000_QSim__simulate_LEND
    131.579721   0.000023 2025-12-04 16:09:36.914 30461.76  6279.35 A000_QSim__simulate_PCAT
    131.579765   0.000044 2025-12-04 16:09:36.914 30461.76  6279.35 A000_QSim__simulate_BRES # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928
    131.579770   0.000005 2025-12-04 16:09:36.914 30461.76  6279.35 A000_QSim__simulate_TAIL
    131.579950   0.000180 2025-12-04 16:09:36.914 30461.76  6279.35 A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_HEAD
    131.580014   0.000064 2025-12-04 16:09:36.914 30461.76  6279.35 A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_cpumerged_HEAD
    322.025602 190.445588 2025-12-04 16:12:47.360 32168.83  7971.96 A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_cpumerged_TAIL # hit=27471928,merged=21062899,save=6409029
    322.025669   0.000067 2025-12-04 16:12:47.360 32168.83  7971.96 A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_TAIL
    322.025677   0.000008 2025-12-04 16:12:47.360 32168.83  7971.96 A000_junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_TAIL
    322.026057   0.000380 2025-12-04 16:12:47.360 32168.83  7971.96 A000_QSim__reset_HEAD
    322.026391   0.000334 2025-12-04 16:12:47.361 32115.37  7918.59 A000_SEvt__endIndex
    322.030447   0.004056 2025-12-04 16:12:47.365 30438.62  6241.84 A000_SEvt__EndOfRun
    Identity                                             Start(s)  Duration(s)     Sub PREL→POST(s)     Sub POST→DOWN(s)    Sub TAIL→RESET(s)    VM(MB)  RSS(MB)  Comment
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    hit/A000_QSim                                         106.466   215.560469            22.996241             1.949322           190.446287   32168.83  7971.96  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928
    → /data1/blyth/tmp/j/zhenning_double_muon/detsim/OJ_LOCAL_Dec04_ok1_hitlite_seed42_evtmax1_run/SProf.txt
      Delta(s)   Δprev(s) DateTime (+08)        VM(MB)  RSS(MB) Label / Comment
    --------------------------------------------------------------------------------------------
      0.000000   0.000000 2025-12-04 16:15:54.245  1023.13   496.45 SEvt__Init_RUN_META
     29.961488  29.961488 2025-12-04 16:16:24.206  6393.96  1375.03 CSGOptiX__Create_HEAD
     31.212086   1.250598 2025-12-04 16:16:25.457  8509.47  1960.28 CSGOptiX__Create_TAIL
    106.462371  75.250285 2025-12-04 16:17:40.707 11099.45  4545.73 junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_HEAD
    106.462438   0.000067 2025-12-04 16:17:40.707 11099.45  4545.73 A000_QSim__simulate_HEAD
    ...



To understand in detail, look at the code:

* https://github.com/simoncblyth/opticks/blob/master/bin/SProf.sh
* https://github.com/simoncblyth/opticks/blob/master/bin/SProf.py

For usage notes::

    SProf.sh help




