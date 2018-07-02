#include <cassert>
#include <vector>

#include "NPY.hpp"

#include "NTreeBalance.hpp"
#include "NTreePositive.hpp"
#include "NTreeAnalyse.hpp"
#include "NTreeProcess.hpp"

#include "PLOG.hh"

template <typename T>
unsigned NTreeProcess<T>::MaxHeight0 = 3 ;   // was discrepantly 4 previously   

template <typename T>
std::vector<unsigned>* NTreeProcess<T>::LVList = NULL ;  

template <typename T>
NPY<unsigned>* NTreeProcess<T>::ProcBuffer = NULL ;  

template <typename T>
void NTreeProcess<T>::SaveBuffer(const char* path)
{
    ProcBuffer->save(path); 
}

template <typename T>
T* NTreeProcess<T>::Process( T* root_ , unsigned soIdx, unsigned lvIdx )  // static
{
    //if( LVList == NULL )
    //     LVList = new std::vector<unsigned> {25,  26,  29,  60,  65,  68,  75,  77,  81,  85, 131, 140} ;
    if( LVList == NULL )
         LVList = new std::vector<unsigned> {25,  26,  29,  60,  68,  75,  77,  81,  85, 131};
 

    if( ProcBuffer == NULL ) ProcBuffer = NPY<unsigned>::make(0,4) ; 

    bool listed = std::find(LVList->begin(), LVList->end(), lvIdx ) != LVList->end() ; 

    if(listed) LOG(info) << "before\n" << NTreeAnalyse<T>::Desc(root_) ; 
     // dump it here, prior to the inplace positivization 
 
    unsigned height0 = root_->maxdepth(); 

    NTreeProcess<T> proc(root_); 

    assert( height0 == proc.balancer->height0 ); 

    T* result = proc.result ; 

    unsigned height1 = result->maxdepth();   

    if(listed) 
    {
         LOG(info) << "after\n" << NTreeAnalyse<T>::Desc(result) ; 
         LOG(info) 
         << " soIdx " << soIdx
         << " lvIdx " << lvIdx
         << " height0 " << height0
         << " height1 " << height1
         << " " << ( listed ? "### LISTED" : "" ) 
         ;
    }

    if(ProcBuffer) ProcBuffer->add(soIdx, lvIdx, height0, height1);   

    return result ; 
} 
 

template <typename T>
NTreeProcess<T>::NTreeProcess( T* root_ ) 
   :
   root(root_),
   balanced(NULL),
   result(NULL),
   balancer(new NTreeBalance<T>(root_)),    // writes depth, subdepth to all nodes
   positiver(NULL)
{
   init();
}

template <typename T>
void NTreeProcess<T>::init()
{
    if(balancer->height0 > MaxHeight0 )
    {
        positiver = new NTreePositive<T>(root) ; 
        balanced = balancer->create_balanced() ;  
        result = balanced ; 
    }
    else
    {
        result = root ; 
    }
}

#include "No.hpp"
#include "NNode.hpp"

template struct NTreeProcess<no> ; 
template struct NTreeProcess<nnode> ; 


