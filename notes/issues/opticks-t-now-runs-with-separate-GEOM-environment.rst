opticks-t-now-runs-with-separate-GEOM-environment
====================================================

NB tests now run within customized OPTICKS_T_GEOM env
--------------------------------------------------------

So to reproduce the same fails that *opticks-t* or *om-test* testing gives from 
the commandline running of test executables, it is necessary to first 
use "om-testenv-push" and then exit the terminal session after running tests 
to avoid confusion from non-standard environment. 

::

    epsilon:ggeo blyth$ t om-test
    om-test () 
    { 
        local rc=0;
        om-testenv-push;
        om-one-or-all test $*;
        rc=$?;
        om-testenv-pop;
        return $rc
    }


    epsilon:ggeo blyth$ t om-testenv-push 
    om-testenv-push () 
    { 
        om-testenv-dump "[push";
        export OM_KEEP_GEOM=$GEOM;
        export GEOM=${OPTICKS_T_GEOM:-$GEOM};
        source $(dirname $BASH_SOURCE)/bin/GEOM_.sh;
        om-testenv-dump "]push"
    }
    epsilon:ggeo blyth$ 


    epsilon:g4cx blyth$ t om-testenv-push
    om-testenv-push () 
    { 
        om-testenv-dump "[push";
        export OM_KEEP_GEOM=$GEOM;
        export GEOM=${OPTICKS_T_GEOM:-$GEOM};
        source $(dirname $BASH_SOURCE)/bin/GEOM_.sh;
        om-testenv-dump "]push"
    }
    epsilon:g4cx blyth$ echo $OPTICKS_T_GEOM
    J004
    epsilon:g4cx blyth$ 



