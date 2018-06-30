#include <cassert>

#include "NTreeBalance.hpp"
#include "NTreePositive.hpp"
#include "NTreeProcess.hpp"

#include "PLOG.hh"

template <typename T>
unsigned NTreeProcess<T>::MaxHeight0 = 4 ;  

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


