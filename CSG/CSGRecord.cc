#ifdef DEBUG_RECORD

#include <sstream>
#include <string>

#include "SSys.hh"
#include "NP.hh"
#include "PLOG.hh"
#include "scuda.h"
#include "squad.h"

#include "OpticksCSG.h"
#include "csg_classify.h"

#include "CSGRecord.h"

std::vector<quad4> CSGRecord::record = {} ;     

bool CSGRecord::ENABLED = SSys::getenvbool("CSGRecord_ENABLED") ;  
void CSGRecord::SetEnabled(bool enabled)  // static
{   
    ENABLED = enabled ; 
}

/**
CSGRecord::Dump
-----------------


              +-------------------------+
              |                      C  |
              |                         |
              |                         |
     +-----[5]|--+                      |
     |       \|  |                      |
     |        4  |                      |
     |        |\ |                      |
     |        | \|                      |
     |        |  3                      |
     |        |  |\                     |
     |        |  | \                    |
     | B      |  |  \                   |
     +--------|--+   \                  |
              |       \                 |
              |      +-2---------+      | 
              |      |  \        |      |
              +------|---1-------|------+       
                     |    \      |
                     |     0     |
                     | A         |
                     +-----------+  

Getting inkling of source of spurious intersects 
when have lots of of constituents because the binary 
comparisons are not being redone.

**/

void CSGRecord::Dump(const char* msg) // static 
{
    LOG(info) << msg << " CSGRecord::record.size " << record.size() << "IsEnabled " << ENABLED  ; 
    for(unsigned i=0 ; i < record.size() ; i++) std::cout << Desc(record[i], i, "rec"); 
} 

std::string CSGRecord::Desc_q2( const quad4& rec )  // static
{
    int typecode = rec.q2.i.x ; 
    bool primitive = typecode >= CSG_SPHERE ; 

    std::stringstream ss ; 
    ss 
        << std::setw(30) << " rec.q2.i.x tc/l/r/lic " 
        << "(" 
        << std::setw(10) << rec.q2.i.x 
        << std::setw(10) << rec.q2.i.y
        << std::setw(10) << rec.q2.i.z
        << std::setw(10) << rec.q2.i.w
        << ")" 
        << std::setw(15) << CSG::Name(rec.q2.i.x) 
        << " " 
        ;

    if( primitive )
    {
        ss << ( rec.q2.i.y == -1 ? "-1" : IntersectionState::Name((IntersectionState_t)rec.q2.i.y)  ) ; 
    }
    else
    {
        ss   
            << ( rec.q2.i.y == -1 ? "-1" : IntersectionState::Name((IntersectionState_t)rec.q2.i.y)  ) << "," 
            << ( rec.q2.i.z == -1 ? "-1" : IntersectionState::Name((IntersectionState_t)rec.q2.i.z)  ) << "," 
            << ( rec.q2.i.w == -1 ? "-1" : ( rec.q2.i.w == 1 ? "leftIsCloser" : "rightIsCloser" ))  
            ;
    }

    ss << std::endl ; 
    std::string s = ss.str() ; 
    return s ; 
}


std::string CSGRecord::Desc_q3( const quad4& rec )  // static
{
    int ctrl = rec.q3.i.z ;
    std::stringstream ss ; 
    ss 
         << std::setw(30) << " rec.q3.i.x tr/nodeIdx/ctrl" 
         << "(" 
         << std::setw(10) << rec.q3.i.x 
         << std::setw(10) << rec.q3.i.y 
         << std::setw(10) << ctrl
         << std::setw(10) << rec.q3.i.w 
         << ")" 
         << " tr "      << std::setw(4) << rec.q3.i.x
         << " nodeIdx " << std::setw(4) << rec.q3.i.y
         << " ctrl " <<  ( ctrl == -1 ? "-1" : CTRL::Name(ctrl) ) 
         << std::endl 
         ;

    std::string s = ss.str() ; 
    return s ; 
}


std::string CSGRecord::Desc( const quad4& rec, unsigned irec, const char* label  )  // static
{
    std::stringstream ss ; 
    ss 
         << " irec " << std::setw(10) << irec << " label " << label 
         << std::endl 
         << std::setw(30) << " rec.q0.f isect  " 
         << "(" 
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q0.f.x 
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q0.f.y
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q0.f.z 
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q0.f.w 
         << ")" 
         << std::endl 
         << std::setw(30) << " rec.q1.f pos  " 
         << "(" 
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q1.f.x 
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q1.f.y
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q1.f.z 
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q1.f.w 
         << ")" 
         << std::endl 
         << Desc_q2(rec)
         << Desc_q3(rec) 
         ;

    std::string s = ss.str() ; 
    return s ; 
}

void CSGRecord::Clear()  // static
{
    record.clear(); 
}
void CSGRecord::Save(const char* dir)  // static
{
    LOG(info) << " dir " << dir ; 
    NP::Write<float>(dir, "CSGRecord.npy", (float*)record.data(),  record.size(), 4, 4 );  
}

#endif
