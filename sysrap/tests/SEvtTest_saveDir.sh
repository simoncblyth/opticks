#!/bin/bash -l 

usage(){ cat << EOU
SEvtTest_saveDir.sh
=====================

Note that when needing to override the default directory it 
preferable to use TMP envvar rather than OPTICKS_OUT_FOLD envvar  
as most of the automatic bookkeeping will still be done.

  
+--------------------------------------------+-----------------------------------------------------------+
|  OPTICKS_OUT_FOLD envvar                   |  SEvt saveDir                                             | 
+============================================+===========================================================+
|   undefined                                |   /tmp/blyth/opticks/GEOM/Pasta/SEvtTest/ALL              |
+--------------------------------------------+-----------------------------------------------------------+
|   /tmp/$USER/opticks                       |   /tmp/blyth/opticks/ALL                                  |
+--------------------------------------------+-----------------------------------------------------------+
|   /tmp/$USER/opticks/GEOM/$GEOM/SEvtTest   |   /tmp/blyth/opticks/GEOM/Pasta/SEvtTest/ALL              |
+--------------------------------------------+-----------------------------------------------------------+


+--------------------------------------------+-----------------------------------------------------------+
|   TMP envvar                               |  SEvt saveDir                                             | 
+============================================+===========================================================+
|    undefined                               |   /tmp/blyth/opticks/GEOM/Pasta/SEvtTest/ALL              |
+--------------------------------------------+-----------------------------------------------------------+
|   /tmp/$USER/opticks                       |   /tmp/blyth/opticks/GEOM/Pasta/SEvtTest/ALL              | 
+--------------------------------------------+-----------------------------------------------------------+

* see SEvt.cc SEvt::save 
 

EOU
}


export GEOM=Pasta 

#export OPTICKS_OUT_FOLD=/tmp/$USER/opticks/GEOM/$GEOM/SEvtTest

tmpbase=${LOCAL_TMPBASE:-/tmp}
export TMP=$tmpbase/$USER/opticks

SEvtTest

