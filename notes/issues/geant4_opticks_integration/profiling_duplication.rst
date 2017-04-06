Profiling Duplication
=========================

TODO
------

OpticksProfile uses Timer internally, but other bare usage remains.

* move persisted OpticksProfile contents into a folder named OpticksProfile at top level
  and duplicate that within datestamped folders, following the old pattern of Timer

* get rid of legacy direct usage of npy/Timer


OpticksProfile
----------------

Canonical m_profile instance resides in Opticks instance, and is created with it.
Used from Opticks via OK_PROFILE::

     39 #define OK_PROFILE(s) \
     40     { \
     41        if(m_ok)\
     42        {\
     43           m_ok->profile((s)) ;\
     44        }\
     45     }

     215 template <typename T>
     216 void Opticks::profile(T label)
     221 void Opticks::dumpProfile(const char* msg, const char* startswith, const char* spacewith, double tcut)
     225 void Opticks::saveProfile()

::

     217 template <typename T>
     218 void Opticks::profile(T label)
     219 {
     220     m_profile->stamp<T>(label, m_tagoffset);
     221    // m_tagoffset is set by Opticks::makeEvent
     222 }





    simon:opticks blyth$ opticks-find OK_PROFILE | wc -l
          38

Profile stamps are persisted in ini and npy, holding the same information::

    /tmp/blyth/opticks/evt/boolean/torch
    simon:torch blyth$ l
    total 40
    drwxr-xr-x  12 blyth  wheel  408 Mar  4 14:09 -1
    drwxr-xr-x  23 blyth  wheel  782 Mar  4 14:09 1
    -rw-r--r--   1 blyth  wheel  701 Mar  4 14:09 DeltaTime.ini
    -rw-r--r--   1 blyth  wheel  573 Mar  4 14:09 DeltaVM.ini
    -rw-r--r--   1 blyth  wheel  480 Mar  4 14:09 Opticks.npy
    -rw-r--r--   1 blyth  wheel  776 Mar  4 14:09 Time.ini
    -rw-r--r--   1 blyth  wheel  656 Mar  4 14:09 VM.ini

Looks like this is currently only done at top level, not 
in the date stamped folders. So only the last invokation times
are available.


Related : npy-/Timer
-------------------------


Other time info is stored with the saved events in t_delta.ini and t_absolute.ini::

    simon:torch blyth$ cat 1/20170304_140910/t_delta.ini 
    _seqhisMakeLookup=0.72714199999973061
    seqhisMakeLookup=0.010654000001522945
    seqhisApplyLookup=2.0999999833293259e-05
    _seqmatMakeLookup=9.9999670055694878e-07
    seqmatMakeLookup=0.006623000001127366
    seqmatApplyLookup=1.5000001440057531e-05
    indexSequenceInterop=0.03241399999751593
    indexBoundaries=0.034935000003315508
    _save=0.11633399999846006
    save=0.15342500000042492

These are default column names from::

    opticksnpy/Timer.cpp

Canonical Timer instance::

    m_timer = new Timer("Opticks::");   


::

    simon:opticks blyth$ opticks-find m_timer
    ./optickscore/Opticks.cc:       m_timer(NULL),
    ./optickscore/Opticks.cc:    m_timer = new Timer("Opticks::");
    ./optickscore/Opticks.cc:    m_timer->setVerbose(true);
    ./optickscore/Opticks.cc:    m_timer->start();
    ./optickscore/Opticks.cc:    return m_timer ; 
    ./optickscore/OpticksEvent.cc:       if(m_timer)\
    ./optickscore/OpticksEvent.cc:          Timer& t = *(m_timer) ;\
    ./optickscore/OpticksEvent.cc:          m_timer(NULL),
    ./optickscore/OpticksEvent.cc:    return m_timer ;
    ./optickscore/OpticksEvent.cc:    m_timer = new Timer("OpticksEvent"); 
    ./optickscore/OpticksEvent.cc:    m_timer->setVerbose(false);
    ./optickscore/OpticksEvent.cc:    m_timer->start();
    ./optickscore/OpticksEvent.cc:    (*m_timer)("_save");
    ./optickscore/OpticksEvent.cc:    (*m_timer)("save");
    ./optickscore/OpticksEvent.cc:    m_timer->stop();
    ./optickscore/OpticksEvent.cc:    m_ttable = m_timer->makeTable();
    ./optickscore/OpticksEvent.cc:    (*m_timer)("load");
    ./optixrap/OScene.cc:       (*m_timer)((s)); \
    ./optixrap/OScene.cc:      m_timer(new Timer("OScene::")),
    ./optixrap/OScene.cc:    m_timer->setVerbose(true);
    ./optixrap/OScene.cc:    m_timer->start();
    ./optickscore/Opticks.hh:       Timer*               m_timer ; 
    ./optickscore/OpticksEvent.hh:       Timer*                m_timer ;
    ./optickscore/OpticksProfile.hh:    m_timer = new Timer("Opticks::");   
    ./optixrap/OScene.hh:       Timer*               m_timer ;
    simon:opticks blyth$ 

::

    1563 void OpticksEvent::makeReport(bool verbose)
    1564 {
    1565     LOG(info) << "OpticksEvent::makeReport " << getTagDir()  ;
    1566 
    1567     if(verbose)
    1568     m_parameters->dump();
    1569 
    1570     m_timer->stop();
    1571 
    1572     m_ttable = m_timer->makeTable();
    1573     if(verbose)
    1574     m_ttable->dump("OpticksEvent::makeReport");
    1575 
    1576     m_report->add(m_parameters->getLines());
    1577     m_report->add(m_ttable->getLines());
    1578 }
    1579 
    1580 
    1581 void OpticksEvent::saveReport()
    1582 {
    1583     std::string tagdir = getTagDir();
    1584     saveReport(tagdir.c_str());
    1585 
    1586     std::string anno = getTimeStamp() ;
    1587     std::string tagdir_ts = getTagDir(anno.c_str());
    1588     saveReport(tagdir_ts.c_str());
    1589 }



