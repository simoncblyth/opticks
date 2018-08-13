OKCORE
========


Opticks
    commandline collector, holds m_run 

OpticksRun
    holds g4 and ok events, resident of Opticks  

OpticksCfg
    options parsing  

OpticksDbg
    helper for commandline lists of indices etc.. 

OpticksQuery
    geometry selection 

OpticksResource
    dispenses file system paths    

OpticksEvent
    holds CPU side buffers 


OpticksEvent Access
---------------------

OpticksRun.m_run is resident of Opticks (visitor to OpticksHub) 
so there is no reason to go up to OpticksHub level for OpticksEvent
access ... get your OpticksRun from Opticks and OpticksEvent 
from there 



Others
---------

::

    Composition.hh
    CompositionCfg.hh
    Animator.hh
    Bookmarks.hh
    Camera.hh
    CameraCfg.hh
    Clipper.hh
    ClipperCfg.hh
    View.hh
    ViewCfg.hh
    InterpolatedView.hh
    OrbitalView.hh
    TrackView.hh
    Light.hh
    Trackball.hh
    TrackballCfg.hh

    Demo.hh
    DemoCfg.hh
    Indexer.hh

    OpticksActionControl.hh
    OpticksAna.hh
    OpticksApp.hh
    OpticksAttrSeq.hh

    OpticksBufferControl.hh
    OpticksBufferSpec.hh

    OpticksColors.hh
    OpticksConst.hh
    OpticksDomain.hh
    OpticksEntry.hh

    OpticksEventAna.hh
    OpticksEventCompare.hh
    OpticksEventDump.hh
    OpticksEventInstrument.hh
    OpticksEventSpec.hh
    OpticksEventStat.hh

    OpticksFlags.hh
    OpticksIndexer.hh
    OpticksMode.hh
    OpticksProfile.hh
    Sparse.hh





