#pragma once

struct NP ; 
struct quad4 ; 
struct quad6 ; 
struct storch ; 
struct uint4 ; 
struct sframe ; 

template <typename T> struct Tran ;

#include <vector>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"


struct SYSRAP_API SEvent
{
    static const plog::Severity LEVEL ; 

    static NP* MakeDemoGensteps(const char* config=nullptr);  

    static NP* MakeTorchGensteps();  
    static NP* MakeCerenkovGensteps();  
    static NP* MakeScintGensteps();  
    static NP* MakeCarrierGensteps(); 
    static NP* MakeGensteps(int gentype); 

    template <typename T> 
    static void FillGensteps( NP* gs, unsigned numphoton_per_genstep ) ; 


    static NP* MakeSeed( const NP* gs ); 
 





    static NP* MakeCountGensteps(const char* config=nullptr, int* total=nullptr);
    static unsigned SumCounts(const std::vector<int>& counts); 

    static void ExpectedSeeds(std::vector<int>& seeds, const std::vector<int>& counts );
    static int  CompareSeeds( const int* seeds, const int* xseeds, int num_seed ); 
    static std::string DescSeed( const int* seed, int num_seed, int edgeitems ); 


    static NP* MakeCountGensteps(const std::vector<int>& photon_counts_per_genstep, int* total );




};



