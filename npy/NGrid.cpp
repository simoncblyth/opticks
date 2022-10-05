/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


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
std::string NGrid<T>::desc(bool label) const 
{
    std::stringstream ss ; 
    ss << rowjoin ; 
    ss << ruler(true) << rowjoin ; 
    ss << ruler(false)  << rowjoin ; 

    for(unsigned r=0 ; r < nr ; r++)
    {
        for(unsigned c=0 ; c < nc ; c++)
        {        
            const T* ptr = get(r,c) ;

            std::string out = unset ; 
            if(ptr)
            {  
                out = ptr->id() ; 
                if(label && ptr->label && strcmp(ptr->label, out.c_str()) != 0)
                {
                    out += " " ; 
                    out += ptr->label ; 
                } 
             }

            ss << std::setw(width) << out ; 
        }
        ss << rowjoin ; 
    }

    ss << rowjoin ; 
    ss << ruler(false) << rowjoin ; 
    ss << ruler(true)  << rowjoin ; 
    ss << labelindex()  << rowjoin ; 

    return ss.str();
}

template<typename T>
const char* NGrid<T>::RULER_MARKS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" ; 

template <typename T>
std::string NGrid<T>::ruler(bool mark) const 
{
    std::stringstream ss ; 
    for(unsigned c=0 ; c < nc ; c++)
    {   
        unsigned num_in_column = 0 ; 
        for(unsigned r=0 ; r < nr ; r++) if(get(r,c)) num_in_column += 1 ; 
        assert( num_in_column < 2 ); 

        char mkr = mark ? RULER_MARKS[c%strlen(RULER_MARKS)] : '.' ;  
        ss << std::setw(width) << ( num_in_column > 0 ? mkr : ' ' ) ; 
    }
    return ss.str();
}

template <typename T>
std::string NGrid<T>::labelindex() const 
{
    std::stringstream ss ; 
    for(unsigned c=0 ; c < nc ; c++)
    {   
        unsigned num_in_column = 0 ; 
        const T* first = nullptr ;  
        for(unsigned r=0 ; r < nr ; r++) 
        {
            const T* ptr = get(r,c) ; 
            if(ptr && first == nullptr) 
            {
                first = ptr ; 
                num_in_column += 1 ; 
            }
            assert( num_in_column < 2 ); 
        }

        if(first && first->label)
        {
            char mkr = RULER_MARKS[c%strlen(RULER_MARKS)] ;  
            ss << std::setw(width) << mkr << " " << first->label << std::endl ; 
        } 
    }
    return ss.str();
}
 







#include "No.hpp"
#include "NNode.hpp"

template struct NGrid<nnode> ; 
template struct NGrid<no> ; 


