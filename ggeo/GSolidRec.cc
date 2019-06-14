#include <iomanip>
#include <sstream>

#include "BStr.hh"
#include "NNode.hpp"
#include "NCSG.hpp"
#include "GSolidRec.hh"

GSolidRec::GSolidRec( const nnode* raw_, const nnode* balanced_, const NCSG* csg_, unsigned soIdx_, unsigned lvIdx_ )
    :
    raw(raw_),
    balanced(balanced_),
    csg(csg_),
    soIdx(soIdx_),
    lvIdx(lvIdx_)
{
}

GSolidRec::~GSolidRec()
{
}


std::string GSolidRec::desc() const 
{
    std::stringstream ss ; 
    ss
        << " so:" << BStr::utoa(soIdx, 3, true)  
        << " lv:" << BStr::utoa(lvIdx, 3, true)  
        << " rmx:" << BStr::utoa(raw->maxdepth(), 2, true )  
        << " bmx:" << BStr::utoa(balanced->maxdepth(), 2, true )  
        << " soName: " << csg->get_soname() 
        ;

    return ss.str(); 
}



