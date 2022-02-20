#ifdef DEBUG_RECORD

#include <sstream>
#include <string>

#include "SSys.hh"
#include "NP.hh"
#include "PLOG.hh"
#include "scuda.h"
#include "squad.h"

#include "sc4u.h"
#include "sbibit.h"
#include "sbit_.h"

#include "OpticksCSG.h"
#include "csg_classify.h"

#include "CSGRecord.h"

std::vector<quad4> CSGRecord::record = {} ;     


CSGRecord::CSGRecord( const quad4& r_ )
    :
    r(r_),
    typecode(CSG_ZERO),
    l_state(State_Miss),
    r_state(State_Miss),
    leftIsCloser(false),
    l_promote_miss(false),
    l_complement(false),
    l_unbounded(false),
    l_false(false),
    r_promote_miss(false),
    r_complement(false),
    r_unbounded(false),
    r_false(false),
    zero0(0),
    zero1(0),
    ctrl(0)
{
    U4U uu ; 
    uu.u = r.q2.u.x ; 
    ctrl = r.q2.u.y ; 

    typecode     =                      sbibit_UNPACK4_0( uu.u4.x ); 
    l_state      = (IntersectionState_t)sbibit_UNPACK4_1( uu.u4.x ); 
    r_state      = (IntersectionState_t)sbibit_UNPACK4_2( uu.u4.x ); 
    leftIsCloser =                      sbibit_UNPACK4_3( uu.u4.x ); 

    l_promote_miss = sbit_UNPACK8_0( uu.u4.y );
    l_complement   = sbit_UNPACK8_1( uu.u4.y );
    l_unbounded    = sbit_UNPACK8_2( uu.u4.y );
    l_false        = sbit_UNPACK8_3( uu.u4.y );

    r_promote_miss = sbit_UNPACK8_4( uu.u4.y );
    r_complement   = sbit_UNPACK8_5( uu.u4.y );
    r_unbounded    = sbit_UNPACK8_6( uu.u4.y );
    r_false        = sbit_UNPACK8_7( uu.u4.y );

    assert( l_false == false ); 
    assert( r_false == false ); 

    zero0 = uu.u4.z ; 
    zero1 = uu.u4.w ; 

    assert( zero0 == 0 ); 
    assert( zero1 == 0 ); 
}

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

std::string CSGRecord::Desc(const quad4& r, unsigned irec, const char* label  )  // static
{
    CSGRecord rec(r); 
    return rec.desc(irec, label); 
}

std::string CSGRecord::desc(unsigned irec, const char* label  ) const 
{
    std::stringstream ss ; 
    ss 
         << " irec " << std::setw(10) << irec << " label " << std::setw(90) << label << " " << CSG::Name(typecode)  
         << std::endl 
         << std::setw(30) << " r.q0.f left  " 
         << "(" 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q0.f.x 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q0.f.y
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q0.f.z 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q0.f.w 
         << ") " 
         << IntersectionState::Name(l_state)
         << " "
         << " " << ( l_promote_miss ? "l_promote_miss" : "-" )
         << " " << ( l_complement ?   "l_complement" : "-" )
         << " " << ( l_unbounded ?   "l_unbounded" : "-" )
         << " "
         << ( leftIsCloser ? "leftIsCloser" : " " )
         << std::endl 
         << std::setw(30) << " r.q1.f right  " 
         << "(" 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q1.f.x 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q1.f.y
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q1.f.z 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q1.f.w 
         << ") " 
         << IntersectionState::Name(r_state)
         << " "
         << " " << ( r_promote_miss ? "r_promote_miss" : "-" )
         << " " << ( r_complement ?   "r_complement" : "-" )
         << " " << ( r_unbounded ?   "r_unbounded" : "-" )
         << " "
         << ( leftIsCloser ? " " : "rightIsCloser" )
         << std::endl 
         << desc_q2()
         << desc_q3() 
         << std::endl 
         ;

    std::string s = ss.str() ; 
    return s ; 
}


std::string CSGRecord::desc_q2() const 
{
    std::stringstream ss ; 
    ss 
        << std::setw(30) << " r.q2.i.x tc/l/r/lic " 
        << "(" 
        << std::setw(10) << r.q2.i.x 
        << std::setw(10) << r.q2.i.y
        << std::setw(10) << r.q2.i.z
        << std::setw(10) << r.q2.i.w
        << ")" 
        << " " 
        << std::endl 
        ;

    std::string s = ss.str() ; 
    return s ; 
}


std::string CSGRecord::desc_q3() const 
{
    std::stringstream ss ; 
    ss 
         << std::setw(30) << " rec.q3.i.x tr/nodeIdx/ctrl" 
         << "(" 
         << std::setw(10) << r.q3.i.x 
         << std::setw(10) << r.q3.i.y 
         << std::setw(10) << ctrl
         << std::setw(10) << r.q3.i.w 
         << ")" 
         << " tr "      << std::setw(4) << r.q3.i.x
         << " nodeIdx " << std::setw(4) << r.q3.i.y
         << " ctrl " <<  CTRL::Name(ctrl) 
         << std::endl 
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
    if( !ENABLED)
    {
        LOG(error) << "CSGRecord::ENABLED is not set, define envvar CSGRecord_ENABLED to do so  " ; 
        return ;  
    }

    unsigned num_record = record.size() ;  
    LOG(info) << " dir " << dir << " num_record " << num_record ; 

    if( num_record > 0)
    {
        NP::Write<float>(dir, "CSGRecord.npy", (float*)record.data(),  num_record, 4, 4 );  
    }
    else
    {
        LOG(error) << "not writing as no records" ; 
    }
}

#endif
