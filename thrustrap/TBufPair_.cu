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

#include "TBufPair.hh"
#include "TBuf.hh"
#include "strided_range.h"
#include "iexpand.h"

template <typename T>
TBufPair<T>::TBufPair(CBufSlice src, CBufSlice dst, bool verbose )
   :
   m_src(src),
   m_dst(dst),
   m_verbose(verbose)
{
}

template <typename T>
void TBufPair<T>::seedDestination()
{  
    if(m_verbose)
    {
        m_src.Summary("TBufPair<T>::seedDestination (CBufSlice)src");
        m_dst.Summary("TBufPair<T>::seedDestination (CBufSlice)dst");
    }

    typedef typename thrust::device_vector<T>::iterator Iterator;
      
    thrust::device_ptr<T> psrc = thrust::device_pointer_cast((T*)m_src.dev_ptr) ;
    thrust::device_ptr<T> pdst = thrust::device_pointer_cast((T*)m_dst.dev_ptr) ;
    
    strided_range<Iterator> si( psrc + m_src.begin, psrc + m_src.end, m_src.stride );
    strided_range<Iterator> di( pdst + m_dst.begin, pdst + m_dst.end, m_dst.stride );

    iexpand( si.begin(), si.end(), di.begin(), di.end() );

//#define DEBUG 1   
#ifdef DEBUG
    std::cout << "TBufPair<T>::seedDestination " << std::endl ;
    thrust::copy( di.begin(), di.end(), std::ostream_iterator<T>(std::cout, " ") );
    std::cout << "TBufPair<T>::seedDestination " << std::endl ;
#endif

}

template class THRAP_API TBufPair<unsigned int> ;

