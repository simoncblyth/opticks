eliminate_opticks_dependency
==============================

All look easy to replace Opticks with SOpticksResource as just used for resource access::

    epsilon:tests blyth$ grep Opticks.hh *.*

    QCerenkovIntegralTest.cc:#include "Opticks.hh"
    QPropTest.cc:#include "Opticks.hh"
    QScintTest.cc:#include "Opticks.hh"
    QSimWithEventTest.cc:#include "Opticks.hh"


