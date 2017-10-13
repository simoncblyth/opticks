PLOG Logging is out of control
=====================================


Issue
-------

Embedded mode revealed a deficiency in PLOG logging, 
there was no way to control the logging ...

Expedient workaround from Tao was adding some static structs 
that initialize PLOG, turning on overly verbose logging.

* https://bitbucket.org/simoncblyth/opticks/commits/edb3ae130a97d9a371292f81863e74d95888e567

::

    +static struct OXRAPPLOGInit {
    +    OXRAPPLOGInit() {
    +
    +        PLOG_(0, 0);
    +
    +    }
    +} s_oxrapploginit;
    +
     
::

    simon:issues blyth$ opticks-find LOGInit 
    ./okop/OpMgr.cc:static struct OpMgrPLOGInit {
    ./okop/OpMgr.cc:    OpMgrPLOGInit() {
    ./optickscore/OpticksEvent.cc:static struct OKCLOGInit {
    ./optickscore/OpticksEvent.cc:    OKCLOGInit() {
    ./optixrap/OPropagator.cc:static struct OXRAPPLOGInit {
    ./optixrap/OPropagator.cc:    OXRAPPLOGInit() {
    simon:opticks blyth$ 

::

    simon:opticks blyth$ opticks-find PLOG_\(0
    ./okop/OpMgr.cc:        PLOG_(0, 0);
    ./optickscore/OpticksEvent.cc:        PLOG_(0, 0);
    ./optixrap/OPropagator.cc:        PLOG_(0, 0);
    simon:opticks blyth$ 


Temporary Fix : Hide Behind ELOG_WORKAROUND macro
--------------------------------------------------


Current Logging Setup 
----------------------

* executable main sets up the loggers via macros like "NPY_LOG__"  
* peculiar preprocessor arrangement plants loggers into every lib listed 
* BUT for convenient use in embedded manner, do not have access to main


Idea for proper fix
---------------------

Arrange for loggers to come into existance automatically 
via static structs... and then be configured from main 
(or elsewhere when operating in embedded mode)



Reproduce
-----------

Any command logs far too much

::

    tboolean-
    tboolean-torus











 
