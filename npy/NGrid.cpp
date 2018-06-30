
#include <cassert>
#include <cstring>
#include <sstream>
#include <iomanip>

#include "NGrid.hpp"

template <typename T>
NGrid<T>::NGrid(unsigned nr_, unsigned nc_, unsigned width_, const char* unset_ , const char* rowjoin_ )
   :
   nr(nr_),
   nc(nc_),
   width(width_),
   unset(strdup(unset_)),
   rowjoin(strdup(rowjoin_)),
   grid(new const T*[nr*nc])      // linear array of nr*nc pointers to instances of T, pointers default initialized to zero
{
    init();
}

template <typename T>
NGrid<T>::~NGrid()
{
    clear();
    delete [] grid ; 
}


template <typename T>
unsigned NGrid<T>::idx(unsigned r, unsigned c) const 
{
   assert( r < nr && c < nc );
   return r*nc + c ; 
}

template <typename T>
void NGrid<T>::set(unsigned r, unsigned c, const T* ptr)
{
    grid[idx(r,c)] = ptr ; 
}

template <typename T>
const T* NGrid<T>::get(unsigned r, unsigned c) const 
{
    const T* ptr = grid[idx(r,c)] ; 
    return ptr ;
}

template <typename T>
void NGrid<T>::init()
{
   // There is no need to set them to NULL, as "new T*[nc*nr]" 
   // invokes the default initializer of T* which sets  
   // the pointers all to zero already.
   //
   // In principal yes, but in practice this relies on cleanliness
   // of the delete : to make sure stale memory is cleared.

   for(unsigned r=0 ; r < nr ; r++){
   for(unsigned c=0 ; c < nc ; c++){

       //assert( get(r,c) == NULL ); 
       set(r,c, NULL) ;  
   }
   }
}

template <typename T>
void NGrid<T>::clear()
{
   for(unsigned r=0 ; r < nr ; r++){
   for(unsigned c=0 ; c < nc ; c++){
       set(r,c, NULL) ;  
   }
   }
}

template <typename T>
std::string NGrid<T>::desc() 
{
    std::stringstream ss ; 
    for(unsigned r=0 ; r < nr ; r++)
    {
        for(unsigned c=0 ; c < nc ; c++)
        {        
            const T* ptr = get(r,c) ;
            //const char* label = ptr ? ptr->label : unset ; 
            std::string id = ptr ? ptr->id() : unset ; 
            ss << std::setw(width) << id ; 
        }
        ss << rowjoin ; 
    }
    return ss.str();
}


#include "No.hpp"
#include "NNode.hpp"

template struct NGrid<nnode> ; 
template struct NGrid<no> ; 


