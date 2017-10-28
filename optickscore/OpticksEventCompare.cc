#include "OpticksEvent.hh"
#include "OpticksEventStat.hh"
#include "OpticksEventCompare.hh"

#include "PLOG.hh"

OpticksEventCompare::OpticksEventCompare(OpticksEvent* a, OpticksEvent* b)
    :
    m_ok(a->getOpticks()),
    m_a(a),
    m_b(b),
    m_as(new OpticksEventStat(a,0)),   
    m_bs(new OpticksEventStat(b,0))   
{
}


void OpticksEventCompare::dump(const char* msg)
{
    LOG(info) << msg ;

    m_as->dump("A");
    m_bs->dump("B");

}




