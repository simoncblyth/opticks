#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include "scuda.h"
#include "SBuf.hh"
#include "QSeed.hh"


void ExpectedSeeds(std::vector<int>& seeds,  const std::vector<int>& counts )
{
    unsigned total = 0 ;  
    for(unsigned i=0 ; i < counts.size() ; i++)  total += counts[i] ; 

    unsigned ni = counts.size(); 
    for(unsigned i=0 ; i < ni ; i++)
    {
        int np = counts[i] ; 
        for(int p=0 ; p < np ; p++) seeds.push_back(i) ; 
    }
    assert( seeds.size() == total );  
}


SBuf<quad6> UploadFakeGensteps(const std::vector<int>& counts)
{
    std::vector<quad6> gs ; 
    unsigned ni = counts.size(); 

    for(unsigned i=0 ; i < ni ; i++)
    {   
        quad6 qq ; 
        qq.q0.i.x = -1 ;   qq.q0.i.y = -1 ;   qq.q0.i.z = -1 ;   qq.q0.i.w = counts[i] ; 
        qq.q1.i.x = -1 ;   qq.q1.i.y = -1 ;   qq.q1.i.z = -1 ;   qq.q1.i.w = -1 ; 
        qq.q2.i.x = -1 ;   qq.q2.i.y = -1 ;   qq.q2.i.z = -1 ;   qq.q2.i.w = -1 ; 
        qq.q3.i.x = -1 ;   qq.q3.i.y = -1 ;   qq.q3.i.z = -1 ;   qq.q3.i.w = -1 ; 
        qq.q4.i.x = -1 ;   qq.q4.i.y = -1 ;   qq.q4.i.z = -1 ;   qq.q4.i.w = -1 ; 
        qq.q5.i.x = -1 ;   qq.q5.i.y = -1 ;   qq.q5.i.z = -1 ;   qq.q5.i.w = -1 ; 
        gs.push_back(qq); 
    }   
    return SBuf<quad6>::Upload(gs) ; 
}

int main(int argc, char** argv)
{

    std::vector<int> counts = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    std::vector<int> xseeds ; 
    ExpectedSeeds(xseeds, counts); 

    SBuf<quad6> gs = UploadFakeGensteps(counts) ; 
    SBuf<int> se = QSeed::CreatePhotonSeeds(gs); 
    se.download_dump("QSeed::CreatePhotonSeeds"); 

    std::vector<int> seeds ; 
    se.download(seeds); 

    assert( seeds.size() == xseeds.size() ); 
    for(unsigned i=0 ; i < seeds.size() ; i++) assert( seeds[i] == xseeds[i] ); 

    return 0 ;  
}




