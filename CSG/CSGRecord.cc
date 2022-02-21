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

std::vector<quad6> CSGRecord::record = {} ;     


CSGRecord::CSGRecord( const quad6& r_ )
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
    tloop(0),
    nodeIdx(0),
    ctrl(0),
    tmin(0.f),
    t_min(0.f),
    tminAdvanced(0.f)
{
    unpack(r.q2.u.x); 

    ctrl = r.q2.u.y ;   // lots of spare bits in here

    tmin = r.q3.f.x ; 
    t_min = r.q3.f.y ; 
    tminAdvanced = r.q3.f.z ; 
}

void CSGRecord::unpack(unsigned packed )
{
    U4U uu ; 
    uu.u = packed ; 

    typecode     =                      sbibit_UNPACK4_0( uu.u4.x ); 
    l_state      = (IntersectionState_t)sbibit_UNPACK4_1( uu.u4.x ); 
    r_state      = (IntersectionState_t)sbibit_UNPACK4_2( uu.u4.x ); 
    leftIsCloser =                      sbibit_UNPACK4_3( uu.u4.x ); 

    l_promote_miss = sbit_rUNPACK8_0( uu.u4.y );
    l_complement   = sbit_rUNPACK8_1( uu.u4.y );
    l_unbounded    = sbit_rUNPACK8_2( uu.u4.y );
    l_false        = sbit_rUNPACK8_3( uu.u4.y );

    r_promote_miss = sbit_rUNPACK8_4( uu.u4.y );
    r_complement   = sbit_rUNPACK8_5( uu.u4.y );
    r_unbounded    = sbit_rUNPACK8_6( uu.u4.y );
    r_false        = sbit_rUNPACK8_7( uu.u4.y );

    assert( l_false == false ); 
    assert( r_false == false ); 

    tloop = uu.u4.z ; 
    nodeIdx = uu.u4.w ; 
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
comparisons are not being redone with looping on 
balanced trees. 

**/

void CSGRecord::Dump(const char* msg) // static 
{
    LOG(info) << msg << " CSGRecord::record.size " << record.size() << "IsEnabled " << ENABLED  ; 
    for(unsigned i=0 ; i < record.size() ; i++) std::cout << Desc(record[i], i, "rec"); 
} 

std::string CSGRecord::Desc(const quad6& r, unsigned irec, const char* label  )  // static
{
    CSGRecord rec(r); 
    return rec.desc(irec, label); 
}

std::string CSGRecord::desc(unsigned irec, const char* label  ) const 
{
    std::stringstream ss ; 
    ss 
         << " tloop " << std::setw(4) << tloop 
         << " nodeIdx " << std::setw(4) << nodeIdx
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
         << " ctrl " << CTRL::Name(ctrl)
         << std::endl 
         << std::setw(30) << " r.q3.f tmin/t_min  " 
         << "(" 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q3.f.x 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q3.f.y
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q3.f.z 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q3.f.w 
         << ") " 
         << " tmin "
         << std::setw(10) << std::fixed << std::setprecision(4) << tmin
         << " t_min "
         << std::setw(10) << std::fixed << std::setprecision(4) << t_min
         << " tminAdvanced "
         << std::setw(10) << std::fixed << std::setprecision(4) << tminAdvanced
         << std::endl 
         << std::setw(30) << " r.q4.f result  " 
         << "(" 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q4.f.x 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q4.f.y
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q4.f.z 
         << std::setw(10) << std::fixed << std::setprecision(4) << r.q4.f.w 
         << ") " 
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
        NP::Write<float>(dir, "CSGRecord.npy", (float*)record.data(),  num_record, 6, 4 );  
    }
    else
    {
        LOG(error) << "not writing as no records" ; 
    }
}

#endif
