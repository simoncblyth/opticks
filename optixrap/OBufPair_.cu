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

#include "OBufPair.hh"
#include "OBuf.hh"

#include "THRAP_HEAD.hh"
#include "strided_range.h"
#include "iexpand.h"
#include "THRAP_TAIL.hh"

template <typename T>
OBufPair<T>::OBufPair(CBufSlice src, CBufSlice dst ) 
   :
   m_src(src),
   m_dst(dst)
{
}


template <typename T>
void OBufPair<T>::seedDestination()
{
    typedef typename thrust::device_vector<T>::iterator Iterator;

    thrust::device_ptr<T> psrc = thrust::device_pointer_cast((T*)m_src.dev_ptr) ; 
    thrust::device_ptr<T> pdst = thrust::device_pointer_cast((T*)m_dst.dev_ptr) ; 

    strided_range<Iterator> si( psrc + m_src.begin, psrc + m_src.end, m_src.stride );
    strided_range<Iterator> di( pdst + m_dst.begin, pdst + m_dst.end, m_dst.stride );

    iexpand( si.begin(), si.end(), di.begin(), di.end() );

#ifdef DEBUG
    std::cout << "OBufPair<T>::seedDestination " << std::endl ; 
    thrust::copy( di.begin(), di.end(), std::ostream_iterator<T>(std::cout, " ") ); 
    std::cout << "OBufPair<T>::seedDestination " << std::endl ; 
#endif

}


template class OXRAP_API OBufPair<unsigned int> ;

