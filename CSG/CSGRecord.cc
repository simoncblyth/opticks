#include <sstream>
#include <string>

#include "NP.hh"
#include "PLOG.hh"
#include "scuda.h"
#include "squad.h"
#include "CSGRecord.h"

std::vector<quad4> CSGRecord::record = {} ;     


void CSGRecord::Dump(const char* msg) // static 
{
    LOG(info) << msg << " CSGRecord::record.size " << record.size() ; 
    for(unsigned i=0 ; i < record.size() ; i++) std::cout << Desc(record[i], i, "rec"); 
} 

std::string CSGRecord::Desc( const quad4& rec, unsigned irec, const char* label  )  // static
{
    const float4& isect = rec.q0.f ; 
    std::stringstream ss ; 
    ss 
         << " irec " << std::setw(10) << irec << " label " << label 
         << std::endl 
         << std::setw(30) << " rec.q0.f isect  " 
         << "(" 
         << std::setw(10) << std::fixed << std::setprecision(4) << isect.x 
         << std::setw(10) << std::fixed << std::setprecision(4) << isect.y
         << std::setw(10) << std::fixed << std::setprecision(4) << isect.z 
         << std::setw(10) << std::fixed << std::setprecision(4) << isect.w 
         << ")" 
         << std::endl 
         << std::setw(30) << " rec.q1 nodeIdx csg.curr " 
         << "(" 
         << std::setw(10) << rec.q1.u.x 
         << std::setw(10) << rec.q1.i.y
         << std::setw(10) << rec.q1.u.z
         << std::setw(10) << rec.q1.u.w
         << ")" 
         << std::endl 
         << std::setw(30) << " rec.q2.f pos  " 
         << "(" 
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q2.f.x 
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q2.f.y
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q2.f.z 
         << std::setw(10) << std::fixed << std::setprecision(4) << rec.q2.f.w 
         << ")" 
         << std::endl 
         ;

    std::string s = ss.str() ; 
    return s ; 
}

void CSGRecord::Save(const char* dir)  // static
{
    LOG(info) << " dir " << dir ; 
    NP::Write<float>(dir, "CSGRecord.npy", (float*)record.data(),  record.size(), 4, 4 );  
}


