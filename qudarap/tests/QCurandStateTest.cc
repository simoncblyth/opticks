/**
QCurandStateTest.cc
=====================

Used at install time via::

    qudarap-prepare-installation () 
    { 
        local sizes=$(qudarap-prepare-sizes);
        local size;
        local seed=${QUDARAP_RNG_SEED:-0};
        local offset=${QUDARAP_RNG_OFFSET:-0};
        for size in $sizes;
        do
            QCurandState_SPEC=$size:$seed:$offset ${OPTICKS_PREFIX}/lib/QCurandStateTest;
            rc=$?;
            [ $rc -ne 0 ] && return $rc;
        done;
        return 0
    }


**/

#include "OPTICKS_LOG.hh"
#include "QCurandState.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    QCurandState* cs = QCurandState::Create() ; 
    LOG(info) << cs->desc() ;

    return 0 ; 
}
