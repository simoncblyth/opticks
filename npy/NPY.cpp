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

#include <iomanip>
#include <algorithm>
#include <iterator>
#include <limits>  

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#include "SSys.hh"
#include "BFile.hh"
#include "BOpticksEvent.hh"
#include "BStr.hh"
#include "BBufSpec.hh"

#include "NSlice.hpp"
#include "NGLM.hpp"

#include "NGLMExt.hpp"
#include "nmat4triple.hpp"
#include "nmat4pair.hpp"

#include "NPYSpec.hpp"
#include "NPY.hpp"
#include "NP.hh"

#include "PLOG.hh"


// ctor takes ownership of a copy of the inputs 
template <typename T>
const plog::Severity NPY<T>::LEVEL = PLOG::EnvLevel("NPY", "DEBUG") ; 

template <typename T>
NPY<T>::NPY(const std::vector<int>& shape, const std::vector<T>& data_, std::string& metadata) 
    :
    NPYBase(shape, sizeof(T), type, metadata, data_.size() > 0),
    m_data(data_),      // copies the vector
    m_unset_item(NULL),
    m_bufspec(NULL),
    m_msk(NULL)
{
    setBasePtr(m_data.data());
} 

template <typename T>
NPY<T>::NPY(const std::vector<int>& shape, const T* data_, std::string& metadata) 
    :
    NPYBase(shape, sizeof(T), type, metadata, data_ != NULL),
    m_data(),      
    m_unset_item(NULL),
    m_bufspec(NULL),
    m_msk(NULL)
{
    if(data_) 
    {
        setData(data_);
    }
}


template <typename T>
void NPY<T>::setData(const T* data_)
{
    assert(data_);
    allocate();
    read(data_);
}

template <typename T>
T* NPY<T>::fill( T value)
{
    allocate();
    std::fill(m_data.begin(), m_data.end(), value);
    return m_data.data(); 
}



template <typename T>
void NPY<T>::zero()
{
    T* data_ = allocate();
    memset( data_, 0, getNumBytes(0) );
}

template <typename T>
void NPY<T>::fillIndexFlat(T offset)
{
    zero();
    unsigned nv = getNumValues(0) ; 
    LOG(info) << " nv " << nv ; 
    for(unsigned idx=0 ; idx < nv ; idx++)  setValueFlat(idx, T(idx) + offset);
}

template <typename T>
int NPY<T>::compareWithIndexFlat()
{
    unsigned nv = getNumValues(0) ; 
    unsigned mismatch = 0 ;  
    for(unsigned idx=0 ; idx < nv ; idx++)  
    {
        T value = getValueFlat(idx); 
        bool match = value == T(idx) ; 
        if(!match)
        {
            mismatch += 1 ; 
            LOG(info) 
                << " idx " << idx 
                << " value " << value 
                << " mismatch " << mismatch 
                ;
        }
    }
    LOG(info) << " total mismatch " << mismatch ; 
    return mismatch ; 
}


/**
NPY<T>::allocate()
--------------------

std::vector *reserve* vs *resize*

*reserve* just allocates without changing size, does that matter ? 
most NPY usage just treats m_data as a buffer so not greatly however
there is some use of m_data.size() so using resize

**/

template <typename T>
T* NPY<T>::allocate()
{
    //assert(m_data.size() == 0);  tripped when indexing a loaded event
    setHasData(true);
    m_data.resize(getNumValues(0));
    setBasePtr(m_data.data());
    return m_data.data();
}

template <typename T>
void NPY<T>::deallocate()
{
    setHasData(false);
    m_data.clear();
    m_data.shrink_to_fit(); 
    setBasePtr(NULL); 
    setNumItems( 0 );
}


/**
NPY::reset
-----------

* clears data reducing NumItems to zero 

**/

template <typename T>
void NPY<T>::reset()
{
    deallocate();
}

template <typename T>
unsigned NPY<T>::capacity() const 
{
    return m_data.capacity(); 
}


template <typename T>
void NPY<T>::read(const void* src)
{
    if(m_data.size() == 0)
    {
        unsigned int nv0 = getNumValues(0) ; 
        LOG(debug) << "NPY<T>::read allocating space now (deferred from earlier) for NumValues(0) " << nv0 ; 
        allocate();
    }
    memcpy(m_data.data(), src, getNumBytes(0) );
}









/**
NPY<T>::write
---------------

See also NPYBase::write_ perhaps can remove this
in favor of that.

**/

template <typename T> 
void NPY<T>::write(void* dst ) const 
{
    memcpy( dst, m_data.data(), getNumBytes(0) ); 
}


template <typename T> 
void NPY<T>::writeItem(void* dst, unsigned item)
{
    unsigned itemVals = getNumValues(1); 
    unsigned itemValsOffset = itemVals*item ;  
    unsigned itemBytes = getNumBytes(1) ; 

    LOG(info) 
        << " itemVals " << itemVals
        << " itemValsOffset " << itemValsOffset
        << " itemBytes " << itemBytes
        ;

    memcpy( dst, m_data.data() + itemValsOffset , itemBytes ); 
}

/**
NPY::grow
------------

* CAUTION this often causes a change to the base ptr after reallocation


About std::vector resize which this invokes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Resizes the container to contain count elements.
If the current size is greater than count, the container is reduced to its first count elements.

If the current size is less than count,
1) additional default-inserted elements are appended
2) additional copies of value are appended.




**/

template <typename T>
T* NPY<T>::grow(unsigned nitems)
{
    setHasData(true);

    unsigned int origvals = m_data.size() ; 
    unsigned int itemvals = getNumValues(1); 
    unsigned int growvals = nitems*itemvals ; 

    LOG(debug)
        << "with space for"
        << " nitems " << nitems
        << " itemvals " << itemvals
        << " origvals " << origvals   
        << " growvals " << growvals
        ;

    m_data.resize(origvals + growvals);  

    //void* old_base_ptr  = getBasePtr();   
    void* new_base_ptr = (void*)m_data.data() ; 
    setBasePtr(new_base_ptr);

    //if(old_base_ptr != new_base_ptr) std::cout << "NPY<T>::grow base_ptr shift " << std::endl ; 

    return m_data.data() + origvals ;
}

/**
NPY::reserve
--------------

About std::vector reserve which this invokes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Increase the capacity of the vector to a value that's greater or equal to new_cap. 
  If new_cap is greater than the current capacity(), new storage is allocated, 
  otherwise the method does nothing. 

* reserve() does not change the size of the vector.  

* If new_cap is greater than capacity(), all iterators, including the past-the-end iterator, 
  and all references to the elements are invalidated. 
  Otherwise, no iterators or references are invalidated.

Thoughts
~~~~~~~~~~

This can improve performance by avoiding the need to reallocate 
so often as the vector grows.

**/

template <typename T>
void NPY<T>::reserve(unsigned items)
{
    unsigned itemvals = getNumValues(1); 
    unsigned vals = items*itemvals ; 
    unsigned cap0 = m_data.capacity(); 

    m_data.reserve(vals); 

    unsigned cap1 = m_data.capacity(); 

    LOG(LEVEL)
        << " reserve " 
        << " items " << items
        << " itemvals " << itemvals
        << " vals " << vals
        << " cap0 " << cap0
        << " cap1 " << cap1
        ; 
    
}







template <typename T>
void NPY<T>::updateDigests()
{
    unsigned ni = getNumItems() ;
    if(m_digests.size() == ni) return ; 

    typedef std::vector<std::string> VS ; 
    for(unsigned i=0 ; i < ni ; i++) 
    {
        std::string item_digest = getItemDigestString(i) ;

        VS::const_iterator begin_ = m_digests.begin();
        VS::const_iterator end_   = m_digests.end();
        VS::const_iterator prior_ = std::find(begin_, end_, item_digest) ;

        if( prior_ == end_ )
        { 
            m_digests.push_back(item_digest) ;
        }
        else
        {
            LOG(fatal) << "NPY<T>::updateDigests finds duplicated items in buffer, MUST start addItemUnique from a unique buffer eg start from empty" ;
            assert(0); 
        }
    }
}


template <typename T>
unsigned NPY<T>::addItemUnique(NPY<T>* other, unsigned item)
{
    // Add single item from other buffer only if the item is not already present in this buffer, 
    // 
    // * the other buffer must have the same itemsize as this buffer
    // * the other buffer may of course only contain a single item, in which case item will be 0 
    //
    // returns the 0-based index of the newly added item if unique or the preexisting 
    // item if not unique
    //
 
    assert(item < other->getNumItems() );

    updateDigests();

    std::string item_digest = other->getItemDigestString(item);

    typedef std::vector<std::string> VS ; 
    VS::const_iterator begin_ = m_digests.begin();
    VS::const_iterator end_   = m_digests.end();
    VS::const_iterator prior_ = std::find(begin_, end_, item_digest) ;

    int index = prior_ == end_ ? -1 : std::distance( begin_, prior_ ) ;

    if( index  == -1 )
    {
        unsigned ni0 = getNumItems();

        addItem(other, item);

        unsigned ni1 = getNumItems();
        assert( ni1 == ni0 + 1 );

        index = ni1 - 1 ; 

        std::string digest = getItemDigestString( index  );
        assert( strcmp( item_digest.c_str(), digest.c_str() ) == 0 );

        m_digests.push_back(digest); 
    }
    return index ; 

}


/*
// moved to base
template <typename T>
bool NPY<T>::hasSameItemSize(NPY<T>* a, NPY<T>* b)
{
    unsigned aItemValues = a->getNumValues(1) ;
    unsigned bItemValues = b->getNumValues(1) ;

    bool same = aItemValues == bItemValues ;  
    if(!same)
    {
    LOG(fatal) << "NPY<T>::hasSameItemSize MISMATCH "
              << " aShape " << a->getShapeString()
              << " bShape " << b->getShapeString()
              << " aItemValues " << aItemValues 
              << " bItemValues " << bItemValues 
              ;
    } 
 
    return same ; 
}
*/



/**
NPY<T>::addItem
-----------------

Add one item from an-other array to this increasing the item count by 1.
This array and the other array must have same itemsize (ie size after 1st dimension).

**/

template <typename T>
void NPY<T>::addItem(NPY<T>* other, unsigned item)    
{
    assert( item < other->getNumItems() );
    unsigned orig = getNumItems();
    unsigned extra = 1 ; 

    bool same= HasSameItemSize(this, other);

    if(!same) LOG(fatal) << "HasSameItemSize FAIL"
                         << " other " << other->getShapeString()
                         << " this " << this->getShapeString()
                         ;

    assert(same && "addItem requires the same item size for this and other");

    unsigned itemNumBytes = getNumBytes(1) ; 
    char* itemBytes = (char*)other->getBytes() + itemNumBytes*item; 

    memcpy( grow(extra), itemBytes, itemNumBytes );

    setNumItems( orig + extra );
}

/**
NPY::expand
------------

Grow the array to hold additional extra_items, 
returns the new total number of array items. 

**/

template <typename T>
unsigned NPY<T>::expand(unsigned extra_items)    
{
    unsigned orig = getNumItems();
    grow(extra_items); 
    setNumItems( orig + extra_items );

    unsigned ni = getNumItems(); 
    return ni ; 
}

 
/**
NPY<T>::add
-------------

Add all items from an-other array to this array.
This array and the other array must have same itemsize (ie size after 1st dimension).

**/

template <typename T>
void NPY<T>::add(const NPY<T>* other)      
{
    unsigned orig = getNumItems();
    unsigned extra = other->getNumItems() ;

    bool same= HasSameItemSize(this, other);
    if(!same) 
        LOG(fatal) << "HasSameItemSize FAIL"
                   << " other " << other->getShapeString()
                   << " this " << this->getShapeString()
                   ;

    assert(same);

    memcpy( grow(extra), other->getBytes(), other->getNumBytes(0) );

    setNumItems( orig + extra );
}




/**
NPY<T>::nasty_old_concat
--------------------------

OOPS: this changes the supposedly const argument array 

**/

template <typename T>
NPY<T>* NPY<T>::nasty_old_concat(const std::vector<const NPYBase*>& comps) // static 
{
    LOG(fatal) << " this old concat has some mis-behaviour  " ; 
    assert(0); 

    unsigned num_comps = comps.size() ; 
    assert( num_comps > 0 ); 

    for(unsigned i=0 ; i < num_comps ; i++)
    {
        const NPYBase* a = comps[i];
        LOG(info) << i << " " << a->getShapeString() ; 
    }

    NPY<T>* comb = (NPY<T>*)comps[0];  

    for(unsigned i=1 ; i < num_comps ; i++)
    {
        const NPY<T>* other = (const NPY<T>*)comps[i] ; 
        comb->add(other); 
    }
    return comb ; 
}

/**
NPY<T>::old_concat
--------------------

Although old_concat is correct and creates the same array content as concat 
it has different memory allocation behaviour due to the use of "add" 
which has been observed to cause problems with subarray image reading 
and textures. This could be due to a stale base ptr, but thats not been 
confirmed.  

Have now made a change which might possibly fix this issue, by 
updating the BasePtr in grow() after resizing m_data.

**/

template <typename T>
NPY<T>* NPY<T>::old_concat(const std::vector<const NPYBase*>& comps) // static 
{
    unsigned num_comps = comps.size() ; 
    LOG(LEVEL) << "[" ; 
    assert( num_comps > 0 ); 


    // check all comps have the same item shape  
    const NPYSpec* spec = NULL ; 
    for(unsigned i=0 ; i < num_comps ; i++)
    {
        const NPYBase* a = comps[i];
        if(spec == NULL){
            spec = a->getShapeSpec();  
        } else {
            assert(spec->isSameItemShape(a->getShapeSpec())); 
        }
        LOG(LEVEL) << i << " " << a->getShapeString() << " spec " << spec->desc() ; 
    }

    // create new array with same item shape as inputs, but zero items
    NPY<T>* comb = NPY<T>::make(0, spec);  
    LOG(LEVEL) << " comb " << comb->getShapeString(); 


    // add all input arrays into the combination array 
    for(unsigned i=0 ; i < num_comps ; i++)
    {
        const NPY<T>* other = (const NPY<T>*)comps[i] ; 
        comb->add(other); 
    }
    LOG(LEVEL) << "]" ; 
    return comb ; 
}


template <typename T>
NPY<T>* NPY<T>::concat(const std::vector<const NPYBase*>& comps) // static 
{
    LOG(LEVEL) << "[" ; 
    unsigned item_size = 0 ; 
    assert( comps.size() > 0 ); 
    std::vector<int> shape(comps[0]->getShapeVector()); 

    unsigned tot_items(0); 

    for(unsigned i=0 ; i < comps.size() ; i++)
    {
        const NPYBase* a = comps[i] ; 
        tot_items += a->getShape(0);  

        if(item_size == 0)
        {
            item_size = a->getNumBytes(1);  
        }
        else
        {
            assert( item_size == a->getNumBytes(1));
        }
        LOG(LEVEL) << i << " " << a->getShapeString() << " item_size " << item_size ; 
    }

    shape[0] = tot_items ; 

    NPY<T>* c = NPY<T>::make(shape);  
    c->zero(); 
    for(unsigned i=0 ; i < comps.size() ; i++)
    {
        const NPYBase* a = comps[i] ; 
        c->read_item_( a->getBytes(), i ); 
    }
    LOG(LEVEL) << "]" ; 

    return c ; 
}



/**
NPY::add T* values, unsigned nvals
-------------------------------------

Used for example by genstep collection.

**/


template <typename T>
void NPY<T>::add(const T* values, unsigned int nvals)  
{
    unsigned int orig = getNumItems();

    int reservation = getReservation(); 
    if(orig == 0 && hasReservation())
    {
        LOG(info) << "adding on empty : setting reservation " << reservation ; 
        reserve(getReservation()); 
    }

    unsigned int itemsize = getNumValues(1) ;
    //LOG(info) << " orig " << orig << " nvals " << nvals << " itemsize " << itemsize ; 

    assert( nvals % itemsize == 0 && "values adding is restricted to integral multiples of the item size");
    unsigned int extra = nvals/itemsize ; 
    memcpy( grow(extra), values, nvals*sizeof(T) );
    setNumItems( orig + extra );
}


template <typename T>
void NPY<T>::addString(const char* s)  
{
    unsigned nl = getNumValues(1) ;
  
    char* cc = new char[nl] ; 
    for( unsigned l=0 ; l < nl ; l++ ) cc[l] = l < strlen(s) ? s[l] : '\0' ; 

    add( cc, nl ) ; 

    delete [] cc ; 
}




template <typename T>
void NPY<T>::add(void* bytes, unsigned int nbytes)  
{
    unsigned int orig = getNumItems();
    unsigned int itembytes = getNumBytes(1) ;
    assert( nbytes % itembytes == 0 && "bytes adding is restricted to integral multiples of itembytes");
    unsigned int extra = nbytes/itembytes ; 
    memcpy( grow(extra), bytes, nbytes );

    setNumItems( orig + extra );
}


template <typename T>
void NPY<T>::add(const glm::vec4& v)
{
    add(v.x, v.y, v.z, v.w);
}







template <typename T>
void NPY<T>::add(const glm::uvec4& u)
{
    add(u.x, u.y, u.z, u.w);
}

template <typename T>
void NPY<T>::add(const glm::ivec4& i)
{
    add(i.x, i.y, i.z, i.w);
}



template <typename T>
void NPY<T>::add(const glm::mat4& m)
{
    const T* values = reinterpret_cast<const T*>(glm::value_ptr(m));    // expect gibberish when not float 
    add((void*)values, 16*sizeof(float) );    
}

template <typename T>
void NPY<T>::add(const glm::vec4& v0, const glm::vec4& v1)
{
    T* vals = new T[8] ;

    vals[0] = v0.x ; 
    vals[1] = v0.y ; 
    vals[2] = v0.z ; 
    vals[3] = v0.w ; 

    vals[4] = v1.x ; 
    vals[5] = v1.y ; 
    vals[6] = v1.z ; 
    vals[7] = v1.w ; 
  
    add((void*)vals, 8*sizeof(T) );    
}



template <typename T>
void NPY<T>::add(T x, T y, T z, T w)  
{
    unsigned int orig = getNumItems();
    unsigned int itemsize = getNumValues(1) ;
    assert( itemsize == 4 && "quad adding is restricted to quad sized items");
    unsigned int extra = 1 ;

    T* vals = new T[4] ;
    vals[0] = x ; 
    vals[1] = y ; 
    vals[2] = z ; 
    vals[3] = w ; 
 
    memcpy( grow(extra), vals, 4*sizeof(T) );

    delete [] vals ; 

    setNumItems( orig + extra );
}



template <typename T>
void NPY<T>::minmax(T& mi_, T& mx_) const 
{
    unsigned int nv = getNumValues(0);
    const T* vv = getValuesConst();

    //T mx(std::numeric_limits<T>::min()); 
    T mx(std::numeric_limits<T>::lowest()); 
    T mi(std::numeric_limits<T>::max()); 

    for(unsigned i=0 ; i < nv ; i++)
    {
        T v = *(vv+i) ; 
        if(v > mx) mx = v ; 
        if(v < mi) mi = v ; 
    }
    mi_ = mi ; 
    mx_ = mx ; 
}

template <typename T>
void NPY<T>::minmax_strided(T& mi_, T& mx_, unsigned stride, unsigned offset) const 
{
    unsigned int nv = getNumValues(0);
    assert( nv % stride == 0);
    assert( offset < stride);

    const T* vv = getValuesConst();

    //T mx(std::numeric_limits<T>::min()); 
    T mx(std::numeric_limits<T>::lowest()); 
    T mi(std::numeric_limits<T>::max()); 

    unsigned ns = nv/stride ; 

    for(unsigned s=0 ; s < ns ; s++)
    {
        unsigned i = s*stride + offset ; 

        T v = *(vv+i) ; 
        if(v > mx) mx = v ; 
        if(v < mi) mi = v ; 
    }
    mi_ = mi ; 
    mx_ = mx ; 
}


template <typename T>
void NPY<T>::minmax3(ntvec3<T>& mi_, ntvec3<T>& mx_) const 
{
    minmax_strided( mi_.x , mx_.x,  3, 0 );
    minmax_strided( mi_.y , mx_.y,  3, 1 );
    minmax_strided( mi_.z , mx_.z,  3, 2 );
}

template <typename T>
void NPY<T>::minmax4(ntvec4<T>& mi_, ntvec4<T>& mx_) const 
{
    minmax_strided( mi_.x , mx_.x,  4, 0 );
    minmax_strided( mi_.y , mx_.y,  4, 1 );
    minmax_strided( mi_.z , mx_.z,  4, 2 );
    minmax_strided( mi_.w , mx_.w,  4, 3 );
}


template <typename T>
void NPY<T>::minmax(std::vector<T>& min_,  std::vector<T>& max_) const 
{
    unsigned nelem = getNumElements(); 
    min_.resize(nelem); 
    max_.resize(nelem); 

    for(unsigned i=0 ; i < nelem ; i++)
    {
        minmax_strided( min_[i] , max_[i],  nelem, i );
    }
}



template <typename T>
ntrange3<T> NPY<T>::minmax3() const 
{
    ntrange3<T> r ; 
    minmax3( r.min , r.max );
    return r ; 
}

template <typename T>
ntrange4<T> NPY<T>::minmax4() const 
{
    ntrange4<T> r ; 
    minmax4( r.min , r.max );
    return r ; 
}




template <typename T>
bool NPY<T>::isConstant(T val) const 
{
    T mi(0);
    T mx(0);
    minmax(mi, mx) ;
    
    bool yes = val == mi && val == mx ;

    LOG(info) << "NPY<T>::isConstant " 
              << " val " << val 
              << " mi " << mi 
              << " mx " << mx
              << " const? " << ( yes ? "YES" : "NO" )
              ; 
    return yes ; 
}


template <typename T>
void NPY<T>::qdump(const char* msg)
{
    unsigned int nv = getNumValues(0);
    T* vv = getValues();

    LOG(info) << msg << " nv " << nv ; 
    for(unsigned i=0 ; i < nv ; i++ ) std::cout << *(vv + i) << " " ; 
    std::cout << std::endl ;       
}




template <typename T>
bool NPY<T>::equals(const NPY<T>* other, bool dump_) const 
{
    unsigned nv = getNumValues(0);
    unsigned no = other->getNumValues(0);

    if(dump_) 
        LOG(info)
            << " nv " << nv  
            << " no " << no
            ;  

    if( nv != no ) return false ; 
    T mxd = maxdiff(other, dump_) ;  
    T zero(0) ; 
    return mxd == zero ;  
} 

template <typename T>
T NPY<T>::maxdiff(const NPY<T>* other, bool dump_) const 
{
    unsigned int nv = getNumValues(0);
    unsigned int no = other->getNumValues(0);
    assert( no == nv);
    const T* v_ = getValuesConst();
    const T* o_ = other->getValuesConst();

    T mx(0); 
    for(unsigned int i=0 ; i < nv ; i++)
    {
        T v = *(v_+i) ; 
        T o = *(o_+i) ; 

        T df = std::fabs(v - o);

        if(dump_ && df > 1e-5)
             std::cout 
                 << "( " << std::setw(10) << i << "/" 
                 << std::setw(10) << nv << ")   "
                 << "v " << std::setw(10) << v 
                 << "o " << std::setw(10) << o 
                 << "d " << std::setw(10) << df
                 << "m " << std::setw(10) << mx
                 << std::endl ; 

        mx = std::max(mx, df );
    }  
    return mx ;  
}



template <typename T>
T* NPY<T>::getUnsetItem()
{
    if(!m_unset_item)
    {
        unsigned int nv = getNumValues(1); // item values  
        m_unset_item = new T[nv];
        while(nv--) m_unset_item[nv] = UNSET  ;         
    }
    return m_unset_item ; 
}

template <typename T>
bool NPY<T>::isUnsetItem(unsigned int i, unsigned int j)
{
    T* unset = getUnsetItem();
    T* item  = getValues(i, j);
    unsigned int nbytes = getNumBytes(1); // bytes in an item  
    return memcmp(unset, item, nbytes ) == 0 ;  // memcmp 0 for match
}








template <typename T>
NPY<T>* NPY<T>::debugload(const char* path)
{
    std::vector<int> shape ;
    std::vector<T> data ;
    std::string metadata = "{}";

    printf("NPY<T>::debugload [%s]\n", path);

    NPY* npy = NULL ;
    aoba::LoadArrayFromNumpy<T>(path, shape, data );
    npy = new NPY<T>(shape,data,metadata) ;

    return npy ;
}


       
template <typename T>
NPY<T>* NPY<T>::loadFromBuffer(const std::vector<unsigned char>& vsrc) // buffer must include NPY header 
{
    std::vector<T> data ; 
    std::vector<int> shape ;
    std::string metadata = "{}";
    bool quietly = false ; 

    const char* bytes = reinterpret_cast<const char*>(vsrc.data()); 
    unsigned size = vsrc.size(); 

    LOG(debug) << "NPY<T>::loadFromBuffer " ; 

    NPY* npy = NULL ;
    try 
    {
        LOG(verbose) <<  "NPY<T>::loadFromBuffer before aoba " ; 
         
        aoba::BufferLoadArrayFromNumpy<T>( bytes, size, shape, data );

        LOG(verbose) <<  "NPY<T>::loadFromBuffer after aoba " ; 

        npy = new NPY<T>(shape,data,metadata) ;  
        // data is copied but ctor, could do more efficiently via an adopt buffer approach ? 
    }
    catch(const std::runtime_error& /*error*/)
    {
        if(!quietly)
        {
        LOG(warning) << "NPY<T>::loadFromBuffer failed : check same scalar type in use "  ; 
        }
    }
    return npy ;
}



template <typename T>
NPY<T>* NPY<T>::load(const char* path_, bool quietly)
{
    std::string path = BFile::FormPath( path_ ); 

    if(GLOBAL_VERBOSE)
    {
        LOG(info) << "NPY<T>::load " << path ; 
    }

    std::vector<int> shape ;
    std::vector<T> data ;
    std::string metadata = "{}";

    LOG(debug) << "NPY<T>::load " << path ; 

    NPY* npy = NULL ;
    try 
    {
        LOG(LEVEL) <<  "[ aoba " ; 
        aoba::LoadArrayFromNumpy<T>(path.c_str(), shape, data );
        LOG(LEVEL) <<  "] aoba " ; 

        npy = new NPY<T>(shape,data,metadata) ;
    }
    catch(const std::runtime_error& /*error*/)
    {
        if(!quietly)
        {
        LOG(error) << "NPY<T>::load failed for path [" << path << "] use debugload with NPYLoadTest to investigate (problems are usually from dtype mismatches) "  ; 
        }
    }

    if( npy != NULL)
    {
        BMeta* meta = NPYBase::LoadMeta( path.c_str(), ".json" ) ; 
        if(meta != NULL)
        {
            npy->setMeta(meta); 
        }
        else
        {
            LOG(debug) << " no .json metadata loaded for " << path ; 
        }
    }

    return npy ;
}



template <typename T>
NPY<T>* NPY<T>::load(const char* dir, const char* name, bool quietly)
{
    std::string path_ = BFile::FormPath(dir, name);
    LOG(LEVEL) 
        << " dir " << dir 
        << " name " << name
        << " path " << path_ 
        ;
    return load(path_.c_str(), quietly);
}


template <typename T>
void NPY<T>::save(const char* dir, const char* name) const 
{
    std::string path_ = BFile::FormPath(dir, name);
    LOG(LEVEL) << "path:[" << path_ << "]" ; 
    save(path_.c_str());
}

template <typename T>
void NPY<T>::save(const char* dir, const char* reldir, const char* name) const 
{
    std::string path_ = BFile::FormPath(dir, reldir, name);
    save(path_.c_str());
}

template <typename T>
bool NPY<T>::exists(const char* dir, const char* name) const 
{
    std::string path_ = BFile::FormPath(dir, name);
    return exists(path_.c_str());
}




template <typename T>
NPY<T>* NPY<T>::load(const char* pfx, const char* tfmt, const char* source, const char* tag, const char* det, bool quietly)
{
    //  (ox,cerenkov,1,dayabay)  ->   (dayabay,cerenkov,1,ox)
    //
    //     arg order twiddling done here is transitional to ease the migration 
    //     once working in the close to old arg order, can untwiddling all the calls
    //
    std::string path_ = BOpticksEvent::path(pfx, det, source, tag, tfmt );
    return load(path_.c_str(),quietly);
}
template <typename T>
void NPY<T>::save(const char* pfx, const char* tfmt, const char* source, const char* tag, const char* det) const 
{
    std::string path_ = BOpticksEvent::path(pfx, det, source, tag, tfmt );
    save(path_.c_str());
}

template <typename T>
bool NPY<T>::exists(const char* pfx, const char* tfmt, const char* source, const char* tag, const char* det) const 
{
    std::string path_ = BOpticksEvent::path(pfx, det, source, tag, tfmt );
    return exists(path_.c_str());
}







template <typename T>
bool NPY<T>::exists(const char* path_) const
{
    fs::path _path(path_);
    return fs::exists(_path) && fs::is_regular_file(_path); 
}





template <typename T>
void NPY<T>::save(const char* raw) const 
{
    std::string native = BFile::FormPath(raw);   // potentially with prefixing/windozing 

    // TODO: replace below with BFile::preparePath do this ???

    fs::path _path(native);
    fs::path dir = _path.parent_path();

    if(dir.string().size() > 0 && !fs::exists(dir))
    {   
        LOG(LEVEL)
            << " raw [" << raw << "]" 
            << " native [" << native << "]" 
            << " _path [" << _path.string() << "]" 
            << " dir [" << dir.string() << "]" 
            << "creating directories " 
            ;
        if (fs::create_directories(dir))
        {   
            LOG(LEVEL)<< "NPYBase::save created directories [" << dir.string() << "]" ;
        }   
    }   


    NPYBase::saveMeta( native.c_str(), ".json" ) ; 


    unsigned int itemcount = getShape(0);    // dimension 0, corresponds to "length/itemcount"
    std::string itemshape = getItemShape(1); // shape of dimensions > 0, corresponds to "item"

    const T* values = getValuesConst();

    bool skip_saving_empty = false ; // <-- formerly caused flakey segfaults when passing NULL values to aoba 

    if(values == NULL && skip_saving_empty )
    {
         LOG(fatal) << "NPY values NULL, SKIP attempt to save  " 
                    << " skip_saving_empty " << skip_saving_empty
                    << " itemcount " << itemcount
                    << " itemshape " << itemshape
                    << " native " << native 
                    ; 
    }
    else
    {
        aoba::SaveArrayAsNumpy<T>(native.c_str(), itemcount, itemshape.c_str(), values );
        if(IsNPDump()) SSys::npdump( native.c_str(), "np.float32", "", "suppress=True") ;
    }

}





template <typename T>
NBufferSpec NPY<T>::saveToBuffer(std::vector<unsigned char>& vdst) const // including the header 
{
   // This enables saving NPY arrays into standards compliant gltf buffers
   // allowing rendering by GLTF supporting renderers.

    NBufferSpec spec = getBufferSpec() ;  

    bool fortran_order = false ; 

    unsigned int itemcount = getShape(0);    // dimension 0, corresponds to "length/itemcount"

    std::string itemshape = getItemShape(1); // shape of dimensions > 0, corresponds to "item"

    vdst.clear();

    vdst.resize(spec.bufferByteLength);

    char* buffer = reinterpret_cast<char*>(vdst.data() ) ; 

    std::size_t num_bytes = aoba::BufferSaveArrayAsNumpy<T>( buffer, fortran_order, itemcount, itemshape.c_str(), (T*)m_data.data() );  

    assert( num_bytes == spec.bufferByteLength ); 

    assert( spec.headerByteLength == 16*5 || spec.headerByteLength == 16*6 ) ; 
  
    return spec ; 
}


template <typename T>
std::size_t NPY<T>::getBufferSize(bool header_only, bool fortran_order) const 
{
    unsigned int itemcount = getShape(0);    // dimension 0, corresponds to "length/itemcount"
    std::string itemshape = getItemShape(1); // shape of dimensions > 0, corresponds to "item"
    return aoba::BufferSize<T>(itemcount, itemshape.c_str(), header_only, fortran_order );
}

template <typename T>
NBufferSpec NPY<T>::getBufferSpec() const 
{
    bool fortran_order = false ; 
    NBufferSpec spec ;  
    spec.bufferByteLength = getBufferSize(false, fortran_order );
    spec.headerByteLength = getBufferSize(true, fortran_order );   // header_only
    spec.uri = "" ; 
    spec.ptr = this ; 

    return spec ; 
}

/**
NPY<T>::make
-------------

Create an array with argspec shape but with NumItems replaced by ni.

**/

template <typename T>
NPY<T>* NPY<T>::make(unsigned int ni, const NPYSpec* argspec)
{
    NPYSpec* argspec2 = argspec->clone(); 
    argspec2->setNumItems(ni); 
    return NPY<T>::make( argspec2 ) ;  
}


template <typename T>
NPY<T>* NPY<T>::make(const NPYSpec* argspec)
{
    std::vector<int> shape ; 
    // i j k l m n
    for(unsigned d=0 ; d < NPYSpec::MAX_DIM ; d++)
    {
        unsigned nd = argspec->getDimension(d) ;
        if(d == 0 || nd > 0) shape.push_back(nd) ;  // only 1st dimension zero is admissable
    }

    NPY<T>* npy = make(shape);
    const NPYSpec* npyspec = npy->getShapeSpec(); 
    bool spec_match = npyspec->isEqualTo(argspec) ;

    if(!spec_match)
    {
       argspec->Summary("argspec"); 
       npyspec->Summary("npyspec"); 
    }
    assert( spec_match && "NPY<T>::make spec mismatch " );

    npy->setBufferSpec(argspec);  // also sets BufferName
    return npy ; 
}




template <typename T>
NPY<T>* NPY<T>::make(unsigned int ni)
{
    std::vector<int> shape ; 
    shape.push_back(ni);
    return make(shape);
}

template <typename T>
NPY<T>* NPY<T>::make(unsigned int ni, unsigned int nj)
{
    std::vector<int> shape ; 
    shape.push_back(ni);
    shape.push_back(nj);
    return make(shape);
}

template <typename T>
NPY<T>* NPY<T>::make(unsigned int ni, unsigned int nj, unsigned int nk)
{
    std::vector<int> shape ; 
    shape.push_back(ni);
    shape.push_back(nj);
    shape.push_back(nk);
    return make(shape);
}


template <typename T>
NPY<T>* NPY<T>::make(unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl)
{
    std::vector<int> shape ; 
    shape.push_back(ni);
    shape.push_back(nj);
    shape.push_back(nk);
    shape.push_back(nl);
    return make(shape);
}

template <typename T>
NPY<T>* NPY<T>::make(unsigned int ni, unsigned int nj, unsigned int nk, unsigned int nl, unsigned int nm)
{
    std::vector<int> shape ; 
    shape.push_back(ni);
    shape.push_back(nj);
    shape.push_back(nk);
    shape.push_back(nl);
    shape.push_back(nm);
    return make(shape);
}




template <typename T>
NPY<T>* NPY<T>::make(const std::vector<int>& shape)
{
    T* values = NULL ;
    std::string metadata = "{}";
    NPY<T>* npy = new NPY<T>(shape,values,metadata) ;
    return npy ; 
}


/**
NPY<T>::make_repeat
----------------------

n>0
    "outer" repeat the entire array 
n<0
    "inner" repeat each item of the array 

**/

template <typename T>
NPY<T>* NPY<T>::make_repeat(NPY<T>* src, int n)
{
    if(n == 0) n = 1 ; 

    unsigned int ni = src->getShape(0);
    assert( ni > 0);

    std::vector<int> dshape(src->getShapeVector());
    dshape[0] *= std::abs(n) ;      // bump up first dimension

    NPY<T>* dst = NPY<T>::make(dshape) ;
    dst->zero();

    unsigned int size = src->getNumBytes(1);  // item size in bytes (from dimension 1)  

    char* sbytes = (char*)src->getBytes();
    char* dbytes = (char*)dst->getBytes();

    assert(size == dst->getNumBytes(1)) ;

    if( n < 0 ) //  "inner" repeat each item of the array 
    {
        unsigned _n = -n ; 
        for(unsigned i=0 ; i < ni ; i++)
        for(unsigned r=0 ; r < _n ;  r++)
        memcpy( (void*)(dbytes + i*(_n)*size + r*size ),(void*)(sbytes + size*i), size ) ; 
    }
    else if( n > 0 )  //  "outer" repeat the entire array 
    {
        unsigned _n = n ; 
        for(unsigned r=0 ; r < _n ;  r++)
        for(unsigned i=0 ; i < ni ; i++)
        memcpy( (void*)(dbytes + i*(_n)*size + r*size ),(void*)(sbytes + size*i), size ) ; 
    }
    return dst ; 
}




template <typename T>
NPY<T>* NPY<T>::make(const std::vector<glm::vec4>& vals)
{
    NPY<T>* buf = NPY<T>::make(vals.size(), 4);
    buf->zero();
    for(unsigned i=0 ; i < vals.size() ; i++)  buf->setQuad(vals[i], i); 
    return buf ; 
}


/**
template <typename T>
NPY<T>* NPY<T>::make(const std::vector<glm::mat4>& mats)
{
    NPY<T>* buf = NPY<T>::make(mats.size(), 4, 4);
    buf->zero();
    buf->read(glm::value_ptr(mats)); 
    return buf ; 
}
**/




template <typename T>
NPY<T>* NPY<T>::make_from_str(const char* s, char delim)
{
    std::vector<T> v ; 
    unsigned n = BStr::Split<T>(v, s, delim);
    assert( v.size() == n );

    return make_from_vec(v) ; 
}

template <typename T>
NPY<T>* NPY<T>::make_from_vec(const std::vector<T>& vals)
{
    unsigned ni = vals.size() ;

    NPY<T>* buf = NPY<T>::make(ni);
    buf->zero();

    unsigned j(0); 
    unsigned k(0); 
    unsigned l(0); 
 
    for(unsigned i=0 ; i < ni ; i++)  buf->setValue(i,j,k,l, vals[i]); 

    return buf ; 
}







template <typename T>
void NPY<T>::setMsk(NPY<unsigned>* msk)
{
    m_msk = msk ; 
}

template <typename T>
NPY<unsigned>* NPY<T>::getMsk() const 
{
    return m_msk ; 
}

template <typename T>
int NPY<T>::getMskIndex(unsigned i) const 
{
    return m_msk ? m_msk->getValue(i, 0, 0) : -1 ; 
}

template <typename T>
bool NPY<T>::hasMsk() const 
{
    return m_msk != NULL ; 
}


/**
NPY::make_masked
------------------

memcpy the masked source items to the destination 
which is sized to fit 

Canonical usage is from NEmitPhotonsNPY::NEmitPhotonsNPY
for masking input photons for Masked running.

The array created retains the original mask array 
allowing origin (unselected) indices to be looked 
up from the selected array. Knowing origin indices
is essential for connecting from selections to originals 
and allows recreation of originals.

**/

template <typename T>
NPY<T>* NPY<T>::make_masked(NPY<T>* src, NPY<unsigned>* msk )
{
    unsigned ni = src->getShape(0);
    assert( ni > 0);

    unsigned nsel = msk->getShape(0); 
    assert( nsel > 0);

    unsigned msk_mi(0) ; 
    unsigned msk_mx(0) ; 
    msk->minmax(msk_mi, msk_mx);

    LOG(info) << "make_masked"
              << " src.ni " << ni
              << " msk.nsel " << nsel 
              << " msk.mi " << msk_mi
              << " msk.mx " << msk_mx
              ;
 
    assert( msk_mi < ni ); 
    assert( msk_mx < ni ); 

    std::vector<int> dshape(src->getShapeVector());
    dshape[0] = nsel ;          // adjust first dimension to selection count

    NPY<T>* dst = NPY<T>::make(dshape) ;
    dst->zero();

    unsigned nsel2 = _copy_masked( dst, src, msk) ; 
    assert(nsel == nsel2);

    dst->setMsk(msk); 

    return dst ; 
}

/**
NPY::_copy_masked
-------------------

Copy items from src to dst that are pointed to via indices
in the msk array. 

**/
 
template <typename T>
unsigned NPY<T>::_copy_masked(NPY<T>* dst, NPY<T>* src, NPY<unsigned>* msk )
{

    unsigned ni = src->getShape(0);
    unsigned nsel = msk->getShape(0);
    assert( ni > 0 && nsel > 0);  

    unsigned size = src->getNumBytes(1);  // item size in bytes (from dimension 1)  
    unsigned size2 = dst->getNumBytes(1) ;
    assert( size == size2 ); 

    char* sbytes = (char*)src->getBytes();
    char* dbytes = (char*)dst->getBytes();

    bool dump = true ; 
    if(dump) std::cout << "_copy_masked"
                       << " ni " << ni 
                       << " nsel " << nsel 
                       << " size " << size 
                       << " " 
                       << std::endl 
                        ; 

    unsigned s(0);
    for(unsigned i=0 ; i < nsel ; i++)
    {
        unsigned item_id = msk->getValue(i,0,0) ;
        assert( item_id < ni ); 
        memcpy( (void*)(dbytes + size*s),(void*)(sbytes + size*item_id), size ) ; 
        s += 1 ; 
    } 
    return s ; 
}



template<typename T>
void NPY<T>::addOffset( int j, int k, unsigned offset, bool preserve_zero, bool preserve_signbit )
{
    unsigned nd = getNumDimensions() ; 
    assert( nd == 3 );  
    unsigned ni = getShape(0); 
    unsigned nj = getShape(1); 
    unsigned nk = getShape(2); 

    unsigned jj = j < 0 ? nj + j : j ; 
    unsigned kk = k < 0 ? nk + k : k ;
    unsigned l = 0 ;  

    if(preserve_signbit)
    {
        for(unsigned i=0 ; i < ni ; i++)
        { 
            unsigned u = getUInt(i,jj,kk) ; 
            unsigned idx0 = u & NOTSIGNBIT ;
            unsigned bit0 = u & SIGNBIT ; 
            unsigned val1 = idx0 == 0 && preserve_zero ? u  : ( bit0 | (idx0 + offset) ) ;   // hmm looses signbit for idx0
            setUInt( i, jj, kk, l,  val1 );  
        }
    }   
    else
    { 
        for(unsigned i=0 ; i < ni ; i++)
        { 
            unsigned u = getUInt(i,jj,kk) ; 
            unsigned val0 = u ;
            unsigned val1 = val0 == 0 && preserve_zero ? val0 : val0 + offset ; 
            setUInt( i, jj, kk, l,  val1 );  
        }
    }
}



template <typename T>
unsigned NPY<T>::count_selection(NPY<T>* src, unsigned jj, unsigned kk, unsigned mask )
{
    unsigned ni = src->getShape(0);
    unsigned nsel(0) ;
    for(unsigned i=0 ; i < ni ; i++)
    {
        unsigned val = src->getUInt(i,jj,kk) ;
        if( val & mask ) nsel += 1 ; 
    } 
    return nsel ; 
}

/**
NPY<T>::make_selection
------------------------

CPU equivalent of thrust stream compaction
this is selecting items from src based on item values.
When the AND of the (jj,kk) element of the item and the mask 
is non-zero the item is selected.

**/

template <typename T>
NPY<T>* NPY<T>::make_selection(NPY<T>* src, unsigned jj, unsigned kk, unsigned mask )
{

    unsigned ni = src->getShape(0);
    assert( ni > 0);

    unsigned nsel = count_selection(src, jj, kk, mask ); 
    std::vector<int> dshape(src->getShapeVector());
    dshape[0] = nsel ;          // adjust first dimension to selection count

    NPY<T>* dst = NPY<T>::make(dshape) ;
    dst->zero();

    unsigned nsel2 = _copy_selection( dst, src, jj, kk, mask) ; 
    assert(nsel == nsel2);

    return dst ; 
}

template <typename T>
unsigned NPY<T>::copy_selection(NPY<T>* dst, NPY<T>* src, unsigned jj, unsigned kk, unsigned mask )
{
    unsigned fromdim = 1 ; 
    assert(dst->hasSameShape(src,fromdim));

    unsigned nsel = count_selection(src, jj, kk, mask ); 
    dst->setNumItems(nsel);
    dst->zero();

    unsigned s = _copy_selection( dst, src, jj, kk, mask) ; 
    assert( nsel == s );
    return nsel ; 
}

template <typename T>
unsigned NPY<T>::write_selection(NPY<T>* dst, unsigned jj, unsigned kk, unsigned mask )
{
    return copy_selection(dst, this, jj, kk, mask );
}


template <typename T>
unsigned NPY<T>::_copy_selection(NPY<T>* dst, NPY<T>* src, unsigned jj, unsigned kk, unsigned mask )
{
    unsigned ni = src->getShape(0);
    unsigned size = src->getNumBytes(1);  // item size in bytes (from dimension 1)  
    char* sbytes = (char*)src->getBytes();
    char* dbytes = (char*)dst->getBytes();
    assert(size == dst->getNumBytes(1)) ;

    bool dump = true ; 
    if(dump) std::cout << "_copy_selection"
                       << " ni " << ni 
                       << " mask " << mask 
                       << " size " << size 
                       << " " 
                       << std::endl 
                        ; 
    unsigned s(0);
    for(unsigned i=0 ; i < ni ; i++)
    {
        unsigned val = src->getUInt(i,jj,kk) ;
        //if(dump) std::cout << val << " " ; 
        if( (val & mask) != 0 ) 
        {
            //if(dump) std::cout << val << " " ; 
            memcpy( (void*)(dbytes + size*s),(void*)(sbytes + size*i), size ) ; 
            s += 1 ; 
        }
    } 
    //if(dump) std::cout << std::endl  ; 
    return s ; 
}









template <typename T>
NPY<T>* NPY<T>::make_like(const NPY<T>* src)
{
     NPY<T>* dst = NPY<T>::make(src->getShapeVector());
     dst->zero();
     return dst ; 
}


template <typename T>
NPY<T>* NPY<T>::make_inverted_transforms(NPY<T>* src, bool transpose)
{
     assert(src->hasItemShape(4,4));
     NPY<T>* dst = make_like(src);
     unsigned ni = src->getShape(0); 
     for(unsigned i=0 ; i < ni ; i++)
     {
         glm::mat4 t =  src->getMat4(i);  
         glm::mat4 v = nglmext::invert_tr( t );
         dst->setMat4(v, i, -1, transpose );
     }
     return dst ; 
}





template <typename T>
NPY<T>* NPY<T>::make_paired_transforms(NPY<T>* src, bool transpose)
{
     assert(src->hasItemShape(4,4));
     unsigned ni = src->getShape(0); 

     NPY<T>* dst = NPY<T>::make(ni, 2, 4, 4);
     dst->zero();

     bool match = true ; 
     std::vector<unsigned> mismatch ; 

     for(unsigned i=0 ; i < ni ; i++)
     {
         glm::mat4 t = src->getMat4(i);  
         glm::mat4 v = nglmext::invert_trs( t, match );
         if(!match) mismatch.push_back(i); 

         dst->setMat4(t, i, 0, transpose );
         dst->setMat4(v, i, 1, transpose );
     }

     if(mismatch.size() > 0)
     {
         LOG(error) << " invert_trs mis-matches found " ;  
         std::cout << " num_mismatch " << mismatch.size() << " num_items " << ni << "  mismatch indices : " ;
         for(unsigned i=0 ; i < mismatch.size() ; i++) std::cout << mismatch[i] << " " ; 
         std::cout << std::endl ; 
     }
     return dst ; 
}



template <typename T>
NPY<T>* NPY<T>::make_triple_transforms(NPY<T>* src)
{
     assert(src->hasItemShape(4,4));
     unsigned ni = src->getShape(0); 

     NPY<T>* dst = NPY<T>::make(ni, 3, 4, 4);
     dst->zero();

     bool match = true ; 
     std::vector<unsigned> mismatch ; 

     for(unsigned i=0 ; i < ni ; i++)
     {
         glm::mat4 t = src->getMat4(i);  
         glm::mat4 v = nglmext::invert_trs( t, match );
         glm::mat4 q = glm::transpose( v ) ;

         if(!match) mismatch.push_back(i); 

         dst->setMat4(t, i, 0 );
         dst->setMat4(v, i, 1 );
         dst->setMat4(q, i, 2 );
     }

     if(mismatch.size() > 0)
     {
         LOG(error) << " invert_trs mis-matches found " ;  
         std::cout << " num_mismatch " << mismatch.size() << " num_items " << ni << "  mismatch indices : " ;
         for(unsigned i=0 ; i < mismatch.size() ; i++) std::cout << mismatch[i] << " " ; 
         std::cout << std::endl ; 
     }

     return dst ; 
}



template <typename T>
NPY<T>* NPY<T>::make_identity_transforms(unsigned ni)
{
     NPY<T>* dst = NPY<T>::make(ni,4,4);
     dst->zero();
     glm::mat4 identity(1.0f);
     for(unsigned i=0 ; i < ni ; i++) dst->setMat4(identity, i, -1, false);
     return dst ; 
}

template <typename T>
NPY<T>* NPY<T>::make_identity_transforms(unsigned ni, unsigned nj)
{
     NPY<T>* dst = NPY<T>::make(ni,nj,4,4);
     dst->zero();
     glm::mat4 identity(1.0f);
     for(unsigned i=0 ; i < ni ; i++){
     for(unsigned j=0 ; j < nj ; j++){
        dst->setMat4(identity, i, j, false);
     }
     }
     return dst ; 
}

template <typename T>
NP* NPY<T>::spawn() const 
{
    return NPY<T>::copy_(this) ; 
}

template <typename T> 
NP* NPY<T>::copy_(const NPY<T>* src)   // static  
{
    const std::vector<int>& sh = src->getShapeVector();    

    NP* dst = NP::Make<T>(); 
    dst->set_shape(sh);   
   
    unsigned long long src_bytes = src->getNumBytes(0);  // total size in bytes 
    unsigned long long dst_bytes = dst->arr_bytes() ;  
    assert( src_bytes == dst_bytes ) ;  

    char* sbytes = (char*)src->getBytes();
    char* dbytes = (char*)dst->bytes();
    memcpy( (void*)dbytes, (void*)sbytes, src_bytes );

    // currently no metadata transfer between old and new array types  

    return dst ; 
}


template <typename T>
NPY<T>* NPY<T>::clone() const 
{
    return NPY<T>::copy(this) ;
}

template <typename T> 
NPY<T>* NPY<T>::copy(const NPY<T>* src) // static 
{
    NPY<T>* dst = NPY<T>::make_like(src);
   
    unsigned int size = src->getNumBytes(0);  // total size in bytes 
    char* sbytes = (char*)src->getBytes();
    char* dbytes = (char*)dst->getBytes();
    memcpy( (void*)dbytes, (void*)sbytes, size );

    NPYBase::transfer(dst, src);

    return dst ; 
}



template <typename T>
NPY<T>* NPY<T>::make_dbg_like(NPY<T>* src, int label_)
{
    NPY<T>* dst = NPY<T>::make(src->getShapeVector());
    dst->zero();

    assert(dst->hasSameShape(src));

    unsigned int ndim = dst->getDimensions();
    assert(ndim <= 5); 

    unsigned int ni = std::max(1u,dst->getShape(0)); 
    unsigned int nj = std::max(1u,dst->getShape(1)); 
    unsigned int nk = std::max(1u,dst->getShape(2)); 
    unsigned int nl = std::max(1u,dst->getShape(3)); 
    unsigned int nm = std::max(1u,dst->getShape(4));

    T* values = dst->getValues();

    for(unsigned int i=0 ; i < ni ; i++){
    for(unsigned int j=0 ; j < nj ; j++){
    for(unsigned int k=0 ; k < nk ; k++){
    for(unsigned int l=0 ; l < nl ; l++){
    for(unsigned int m=0 ; m < nm ; m++)
    {   
         unsigned int index = i*nj*nk*nl*nm + j*nk*nl*nm + k*nl*nm + l*nm + m  ;

         int label = 0 ; 

         if(      label_ == 0  ) label = index ;

         else if( label_ == 1  ) label = i ;  
         else if( label_ == 2  ) label = j ;  
         else if( label_ == 3  ) label = k ;  
         else if( label_ == 4  ) label = l ;  
         else if( label_ == 5  ) label = m ;  
         
         else if( label_ == -1 ) label = i ;  
         else if( label_ == -2 ) label = i*nj + j ;  
         else if( label_ == -3 ) label = i*nj*nk + j*nk + k  ;  
         else if( label_ == -4 ) label = i*nj*nk*nl + j*nk*nl + k*nl + l  ;  
         else if( label_ == -5 ) label = i*nj*nk*nl*nm + j*nk*nl*nm + k*nl*nm + l*nm + m  ;  

         *(values + index) = T(label) ;
    }   
    }   
    }   
    }   
    }   

    return dst ; 
}





template <typename T>
NPY<float>* NPY<T>::MakeFloat(const NPY<T>* src )
{
    const NPYSpec* argspec = src->getShapeSpec()->cloneAsFloat() ; 
    NPY<float>* dst = NPY<float>::make(argspec); 
    dst->zero(); 
    unsigned nv = src->getNumValues(0) ; 
    LOG(info) << " nv " << nv ; 
    for(unsigned idx=0 ; idx < nv ; idx++) 
    {
        T value = src->getValueFlat(idx); 
        dst->setValueFlat(idx, float(value));
    }
    return dst ; 
} 


template <typename T>
NPY<double>* NPY<T>::MakeDouble(const NPY<T>* src )
{
    const NPYSpec* argspec = src->getShapeSpec()->cloneAsDouble() ; 
    NPY<double>* dst = NPY<double>::make(argspec); 
    dst->zero(); 
    unsigned nv = src->getNumValues(0) ; 
    LOG(info) << " nv " << nv ; 
    for(unsigned idx=0 ; idx < nv ; idx++) 
    {
        T value = src->getValueFlat(idx); 
        dst->setValueFlat(idx, double(value));
    }
    return dst ; 
} 







/**
NPY<T>::make_modulo
---------------------

Modulo selection of src array.  

NB this is limited to 3d arrays. For a more general version
use make_modulo_selection. 

**/

template <typename T>
NPY<T>* NPY<T>::make_modulo(NPY<T>* src, unsigned int scaledown)
{

    assert(0 && "switch to more general make_modulo_selection"); 

    std::vector<T>& sdata = src->data();
    std::vector<T>  ddata ; 

    unsigned nd = src->getDimensions(); 
    assert( nd == 3 ); 

    unsigned ni = src->getShape(0) ;
    unsigned nj = src->getShape(1) ;
    unsigned nk = src->getShape(2) ;

    printf("make_modulo ni %d nj %d nk %d \n", ni, nj, nk );

    unsigned int dni(0);
    for(unsigned int i=0 ; i < ni ; i++)
    {
        if(i % scaledown == 0)
        {
            dni += 1 ; 
            for(unsigned int j=0 ; j < nj ; j++){
            for(unsigned int k=0 ; k < nk ; k++){
  
                unsigned int index = i*nj*nk + j*nk + k ;
                ddata.push_back(sdata[index]);

            }
            }
        }
    }

    std::vector<int> dshape ; 
    dshape.push_back(dni);
    dshape.push_back(nj);
    dshape.push_back(nk);

    assert(ddata.size() == dni*nj*nk );
    std::string dmetadata = "{}";

    NPY<T>* dst = new NPY<T>(dshape,ddata,dmetadata) ;
    return dst ; 
}



/**
NPY<T>::modulo_count
---------------------

Count items selected by modulo and index, where index < modulo.

**/

template <typename T>
unsigned NPY<T>::modulo_count(unsigned ni, unsigned modulo, unsigned index)  // static
{
    unsigned count(0); 
    for(unsigned i=0 ; i < ni ; i++ ) if( i % modulo == index ) count++ ; 
    return count ; 
}

/**
NPY<T>::make_modulo_selection
------------------------------

Construct a destination array from a source array by selecting 
items according to a modulo and an index within that modulo.
The index must be less than the modulo.

For example with modulo 2, using index 0 or 1 will pick alternating items. 
**/

template <typename T>
NPY<T>* NPY<T>::make_modulo_selection(const NPY<T>* src, unsigned modulo, unsigned index) // static
{
    assert( index < modulo ); 
    unsigned ni = src->getNumItems(); 
    unsigned dst_ni = modulo_count(ni, modulo, index); 

    std::vector<int> shape(src->getShapeVector()) ; 
    assert(shape[0] == int(ni)); 
    shape[0] = dst_ni ; 

    NPY<T>* dst = NPY<T>::make(shape);
    dst->zero(); 

    unsigned item_bytes = src->getNumBytes(1);  // from dimension 1

    char* src_bytes = (char*)src->getBytes();
    char* dst_bytes = (char*)dst->getBytes() ;    
    
    unsigned offset = 0 ; 
    for(unsigned i=0 ; i < ni ; i++)
    {  
        if( i % modulo == index )
        { 
            memcpy( (void*)(dst_bytes + offset),(void*)(src_bytes + item_bytes*i), item_bytes ) ; 
            offset += item_bytes ; 
        }
    }   
    return dst ;  
}


/**
NPY<T>::make_interleaved
--------------------------

Construct a destination array from multiple source arrays 
by selecting items from each in turn creating an interleaved array.
For example this can be used to reconstruct the original array from 
those created by make_modulo_selection.

**/

template <typename T>
NPY<T>* NPY<T>::make_interleaved( const std::vector<NPYBase*>& srcs )
{
    unsigned num_src = srcs.size(); 

    unsigned item_bytes = 0 ; 
    unsigned dst_ni = 0 ; 

    for(unsigned s=0 ; s < num_src ; s++)
    {
         NPYBase* src = srcs[s] ; 
         dst_ni += src->getNumItems();  
         if( item_bytes == 0 )
         {
             item_bytes = src->getNumBytes(1); 
         }
         else
         {
              assert( item_bytes == src->getNumBytes(1) && "all sources must have the same itemsize" );
         }
    }

    std::vector<int> shape(srcs[0]->getShapeVector()) ; 
    shape[0] = dst_ni ; 

    NPY<T>* dst = NPY<T>::make(shape);
    dst->zero(); 
    char* dst_bytes = (char*)dst->getBytes() ;    

    for(unsigned s=0 ; s < num_src ; s++)
    {
        NPYBase* src = srcs[s]; 
        char* src_bytes = (char*)src->getBasePtr(); 
        unsigned sni = src->getNumItems(); 
        for(unsigned si=0 ; si < sni ; si++)
        {
            unsigned di = num_src*si + s ;  ;      
            memcpy( (void*)(dst_bytes + item_bytes*di),(void*)(src_bytes + item_bytes*si), item_bytes ) ; 
        }
    }
    return dst ;  
}



template <typename T>
unsigned NPY<T>::compare( const NPY<T>* a, const NPY<T>* b, const std::vector<T>&  epsilons,  bool dump, unsigned dumplimit, char mode )
{
    unsigned mismatch_tot = 0 ; 
    for(unsigned i=0 ; i < epsilons.size() ; i++)
    {   
        double epsilon = epsilons[i] ; 
        unsigned mismatch = NPY<T>::compare( a, b, epsilon, dump, dumplimit, mode  );
        std::cout 
            << " epsilon " << std::setw(10) << std::scientific << epsilon 
            << " mismatch " << mismatch
            << std::endl 
            ;   
        mismatch_tot += mismatch ; 
    }   
    return mismatch_tot ; 
}


template <typename T>
bool NPY<T>::compare_value( const T a, const T b, const T epsilon,  char mode  )
{
     bool match(false) ; 
     if( mode == 'I' )
     {
         match = a == b ; 
     }
     else if ( mode == 'A' )
     {
         T d = a - b ;
         if(d < 0) d = -d ;  // std::abs has trouble with some types, but are here avoiding those with the mode
         match = d <= epsilon ;
     }
     else if ( mode == 'R' )
     {
         T r = b == 0. ? 1. : 1. - (a/b) ; 
         if( r < 0.) r = -r ; 
         match = r <= epsilon ; 
     }
     else
     {
         assert( 0 && "mode must be one of 'I' 'A' 'R' " );  
     }
     return match ; 
}


/**
NPY<T>::compare
-----------------

Itemwise comparison of array values. 

**/

template <typename T>
unsigned NPY<T>::compare( const NPY<T>* a, const NPY<T>* b, const T epsilon,  bool dump, unsigned dumplimit, char mode  )
{
    if(IsInteger())
    {
        bool expect = mode == 'I' ; 
        if(!expect)   
            LOG(fatal) << " for integer types the comparison mode must be 'I' and the epsilon is ignored " ; 
        assert(expect); 
    }   
    else if( IsReal() )
    {
        bool expect = mode == 'A' || mode == 'R' ; 
        if(!expect)   LOG(fatal) << " for real types the comparison mode must be 'A' or 'R' and the epsilon is used " ; 
        assert(expect); 
    }
    else
    {
        LOG(fatal) << " unexpected type " ;  
        assert(0); 
    }

    bool is_char = IsChar() ; 

    if(dump)
    {
        LOG(info) << " a " << a->getShapeString(); 
        LOG(info) << " b " << b->getShapeString(); 
    }

    assert( a->hasSameShape(b) ); 
    unsigned ni = a->getNumItems();      // first dimension 
    unsigned nv = a->getNumValues(1);    //  

    if(dump)
    {
        LOG(info) << " ni " << ni << " nv " << nv << " dumplimit " << dumplimit << " epsilon " << epsilon << " mode " << mode << " is_char " << is_char ; 
    }     

    unsigned mismatch_items(0); 

    for(unsigned i=0 ; i < ni ; i++)
    {
        const T* av = a->getValuesConst(i, 0); 
        const T* bv = b->getValuesConst(i, 0); 

        unsigned mismatch_values(0); 
        for(unsigned v=0 ; v < nv ;v++)
        {
            bool match = compare_value( av[v], bv[v], epsilon, mode );  
            if(!match)
            {
                mismatch_values++ ; 
                if(dump && mismatch_values < dumplimit ) 
                {
                    std::cout 
                        << " mismatch_values " << std::setw(4) << mismatch_values
                        << " i " << std::setw(4) << i 
                        << " v " << std::setw(4) << v
                        << " a " << std::setw(4) << ( is_char ? (int)av[v] : av[v] )
                        << " b " << std::setw(4) << ( is_char ? (int)bv[v] : bv[v] )
                        << " epsilon " << std::scientific << epsilon 
                        << std::endl 
                        ;
                }
            }
        }       
        if(mismatch_values > 0) 
        {
            std::cout 
                 << " i " << std::setw(4) << i 
                 << " mismatch_values " << std::setw(4) << mismatch_values
                 << std::endl 
                 ;
            mismatch_items++ ; 
        }
    }
    if( dump || mismatch_items > 0 )
    {
        LOG(info) << " mismatch_items " << mismatch_items ;
    }
    return mismatch_items ; 
}



/**
NPY<T>::compare_element_jk
------------------------------
**/
template <typename T>
unsigned NPY<T>::compare_element_jk(const NPY<T>* a, const NPY<T>* b, int j, int k, bool dump )  // static
{
    if(dump)
    {
        LOG(info) << " a " << a->getShapeString(); 
        LOG(info) << " b " << b->getShapeString(); 
        LOG(info) 
            << " (j,k):   " << "(" << j << "," << k << ") " 
            ;
    }

    unsigned nd_a = a->getNumDimensions(); 
    unsigned nd_b = b->getNumDimensions(); 
    assert( nd_a == nd_b ); 
    assert( nd_a == 3 ); 

    unsigned ni_a = a->getShape(0); 
    unsigned ni_b = b->getShape(0); 
    assert( ni_a == ni_b ); 
    unsigned ni = ni_a ; 

    unsigned mismatch = 0 ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
        unsigned av = a->getUInt(i,j,k);
        unsigned bv = b->getUInt(i,j,k);

        bool match = av == bv ; 
        if(!match) mismatch++ ; 

        if(dump)   
        {
            std::cout 
                << std::setw(10) << i 
                << " : " << std::setw(10) << av 
                << " : " << std::setw(10) << bv 
                << std::endl 
                ;
        }

    }
    return mismatch ; 
}







template <typename T>
NPY<T>* NPY<T>::make_slice(const char* slice_)
{
    NSlice* slice = slice_ ? new NSlice(slice_) : NULL ;
    return make_slice(slice);
}

template <typename T>
NPY<T>* NPY<T>::make_slice(NSlice* slice)
{
    unsigned int ni = getShape(0);
    if(!slice)
    {
        slice = new NSlice(0, ni, 1);
        LOG(warning) << "NPY::make_slice NULL slice, defaulting to full copy " << slice->description() ;
    }
    unsigned int count = slice->count();

    LOG(verbose) << "NPY::make_slice from " 
              << ni << " -> " << count 
              << " slice " << slice->description() ;

    assert(count <= ni);

    char* src = (char*)getBytes();
    unsigned int size = getNumBytes(1);  // from dimension 1  
    unsigned int numBytes = count*size ; 


    char* dest = new char[numBytes] ;    

    unsigned int offset = 0 ; 
    for(unsigned int i=slice->low ; i < slice->high ; i+=slice->step)
    {   
        memcpy( (void*)(dest + offset),(void*)(src + size*i), size ) ; 
        offset += size ; 
    }   

    const std::vector<int>& orig = getShapeVector();
    std::vector<int> shape(orig.size()) ; 
    std::copy(orig.begin(), orig.end(), shape.begin() );

    assert(shape[0] == int(ni)); 
    shape[0] = count ; 

    NPY<T>* npy = NPY<T>::make(shape);
    npy->setData( (T*)dest );        // reads into NPYs owned storage 
    delete[] dest ; 
    return npy ; 
}




template <typename T>
NPY<T>* NPY<T>::make_vec3(float* m2w_, unsigned int npo)
{
/*
   Usage example to create debug points in viscinity of a drawable

   npy = NPY<T>::make_vec3(dgeo->getModelToWorldPtr(),100); 
   vgst.add(new VecNPY("vpos",npy,0,0));

*/

    glm::mat4 m2w ;
    if(m2w_) m2w = glm::make_mat4(m2w_);

    std::vector<T> data;

    //std::vector<int>   shape = {int(npo), 1, 3} ;   this is a C++11 thing
    std::vector<int> shape ; 
    shape.push_back(npo);
    shape.push_back(1);
    shape.push_back(3);

    std::string metadata = "{}";

    float scale = 1.f/float(npo);

    for(unsigned int i=0 ; i < npo ; i++ )
    {
        glm::vec4 m(float(i)*scale, float(i)*scale, float(i)*scale, 1.f);
        glm::vec4 w = m2w * m ;

        data.push_back(w.x);
        data.push_back(w.y);
        data.push_back(w.z);
    } 
    NPY<T>* npy = new NPY<T>(shape,data,metadata) ;
    return npy ;
}






template <typename T>
unsigned int NPY<T>::getUSum(unsigned int j, unsigned int k) const 
{
    unsigned int ni = m_ni ;
    unsigned int nj = m_nj ;
    unsigned int nk = m_nk ;

    assert(m_dim == 3 && j < nj && k < nk);

    unsigned int usum = 0 ; 
    uif_t uif ; 
    for(unsigned i=0 ; i<ni ; i++ )
    {
        unsigned index = i*nj*nk + j*nk + k ;
        uif.f = m_data[index] ;
        usum += uif.u ;
    }
    return usum ; 
}




template <typename T>
std::set<int> NPY<T>::uniquei(unsigned int j, unsigned int k)
{
    unsigned int ni = m_ni ;
    unsigned int nj = m_nj ;
    unsigned int nk = m_nk ;
    assert(m_dim == 3 && j < nj && k < nk);

    std::set<int> uniq ; 
    uif_t uif ; 
    for(unsigned int i=0 ; i<ni ; i++ )
    {
        unsigned int index = i*nj*nk + j*nk + k ;
        uif.f = m_data[index] ;
        int ival = uif.i ;
        uniq.insert(ival);
    }
    return uniq ; 
}




template <typename T>
bool NPY<T>::second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b)
{
    return a.second > b.second ;
}

template <typename T>
std::vector<std::pair<int,int> > NPY<T>::count_uniquei_descending(unsigned int j, unsigned int k)
{
    std::map<int,int> uniqn = count_uniquei(j,k) ; 

    std::vector<std::pair<int,int> > pairs ; 
    for(std::map<int,int>::iterator it=uniqn.begin() ; it != uniqn.end() ; it++) pairs.push_back(*it);

    std::sort(pairs.begin(), pairs.end(), second_value_order );

    return pairs ; 
}



template <typename T>
std::map<int,int> NPY<T>::count_uniquei(unsigned int j, unsigned int k, int sj, int sk )
{
    unsigned int ni = m_ni ; 
    unsigned int nj = m_nj ;
    unsigned int nk = m_nk ;
    assert(m_dim == 3 && j < nj && k < nk);

    bool sign = sj > -1 && sk > -1 ;
 

    std::map<int, int> uniqn ; 
    uif_t uif ; 
    for(unsigned int i=0 ; i<ni ; i++ )
    {
        unsigned int index = i*nj*nk + j*nk + k ;
        uif.f = m_data[index] ;
        int ival = uif.i ;

        if(sign)
        {
            unsigned int sign_index = i*nj*nk + sj*nk + sk ;
            float sval = m_data[sign_index] ;
            if( sval < 0.f ) ival = -ival ; 
        }

        if(uniqn.count(ival) == 0)
        {
            uniqn[ival] = 1 ; 
        }
        else 
        {  
            uniqn[ival] += 1 ; 
        }
    }

    return uniqn ; 
}




template <typename T>
std::map<unsigned int,unsigned int> NPY<T>::count_unique_u(unsigned int j, unsigned int k )
{
    unsigned int ni = m_ni ;
    unsigned int nj = m_nj ;
    unsigned int nk = m_nk ;

    assert(m_dim == 3 && j < nj && k < nk);

    std::map<unsigned int, unsigned int> uniq ;
 
    uif_t uif ; 
    for(unsigned int i=0 ; i<ni ; i++ )
    {
        uif.f = m_data[i*nj*nk + j*nk + k] ;
        unsigned int uval = uif.u ;
        if(uniq.count(uval) == 0) uniq[uval] = 1 ; 
        else                      uniq[uval] += 1 ; 
    }
    return uniq ; 
}



template <typename T>
void NPY<T>::dump(const char* msg, unsigned int limit)
{
    LOG(info) << msg << " (" << getShapeString() << ") " ; 

    unsigned int ni = getShape(0);
    unsigned int nj(0) ;
    unsigned int nk(0) ; 


    if(m_dim == 3)
    {
        nj = getShape(1);
        nk = getShape(2);
    }
    else if(m_dim == 2)
    {
        nj = 1;
        nk = getShape(1);
    }
    else if(m_dim == 1)
    {
        nj = 1 ; 
        nk = 1 ; 
    }


    LOG(verbose) << "NPY<T>::dump " 
               << " dim " << m_dim 
               << " ni " << ni
               << " nj " << nj
               << " nk " << nk
               ;

    T* ptr = getValues();

    for(unsigned int i=0 ; i < std::min(ni, limit) ; i++)
    {   
        for(unsigned int j=0 ; j < nj ; j++)
        {
            for(unsigned int k=0 ; k < nk ; k++)
            {   
                unsigned int offset = i*nj*nk + j*nk + k ; 

                LOG(verbose) << "NPY<T>::dump " 
                           << " i " << i 
                           << " j " << j 
                           << " k " << k
                           << " offset " << offset 
                           ; 

                T* v = ptr + offset ; 
                if(k%nk == 0) std::cout << std::endl ; 

                if(k==0) std::cout << "(" <<std::setw(3) << i << ") " ;
                std::cout << " " << std::fixed << std::setprecision(3) << std::setw(10) << *v << " " ;   
            }   
        }
   }   


   LOG(verbose) << "NPY<T>::dump DONE " ; 

   std::cout << std::endl ; 
}





template <typename T>
NPY<T>* NPY<T>::scale(float factor)
{ 
   glm::mat4 m = glm::scale(glm::mat4(1.0f), glm::vec3(factor));
   return transform(m);
}

template <typename T>
bool NPY<T>::is_pshaped() const 
{
    const std::vector<int>& shape = getShapeVector(); 
    bool property_shaped = shape.size() == 2 && shape[1] == 2 && shape[0] > 1 ;
    return property_shaped ;
}

template <typename T>
void NPY<T>::pscale(T scale, unsigned column)
{
    assert( is_pshaped() );
    assert( column < 2 );
    T* vv = getValues();
    unsigned ni = getNumItems(); 
    for(unsigned i=0 ; i < ni ; i++) 
    {
        vv[2*i+column] = vv[2*i+column]*scale ;
    }
}

template<typename T> 
void NPY<T>::pdump(const char* msg) const
{
    bool property_shaped = is_pshaped();
    assert( property_shaped );

    unsigned ni = getNumItems() ;
    std::cout << msg << " ni " << ni << std::endl ;

    const T* vv = getValuesConst();

    for(unsigned i=0 ; i < ni ; i++)
    {
        std::cout
             << " i " << std::setw(3) << i
             << " px " << std::fixed << std::setw(10) << std::setprecision(5) << vv[2*i+0]
             << " py " << std::fixed << std::setw(10) << std::setprecision(5) << vv[2*i+1]
             << std::endl
             ;
    }
}







template <typename T>
NPY<T>* NPY<T>::transform(glm::mat4& mat)
{ 
    unsigned int ni = getShape(0);
    unsigned int nj(0) ;
    unsigned int nk(0) ; 

    if(m_dim == 3)
    {
        nj = getShape(1);
        nk = getShape(2);
    }
    else if(m_dim == 2)
    {
        nj = 1;
        nk = getShape(1);
    }
    else
        assert(0);

    assert(nk == 3 || nk == 4);

    std::vector<int> shape = getShapeVector();
    NPY<T>* tr = NPY<T>::make(shape);
    tr->zero();

    T* v0 = getValues();
    T* t0 = tr->getValues();

    for(unsigned int i=0 ; i < ni ; i++)
    {   
        for(unsigned int j=0 ; j < nj ; j++)
        {
             unsigned int offset = i*nj*nk + j*nk + 0 ; 
             T* v = v0 + offset ; 
             T* t = t0 + offset ; 
 
             glm::vec4 vec ;
             vec.x = *(v+0) ;  
             vec.y = *(v+1) ;  
             vec.z = *(v+2) ; 
             vec.w = nk == 4 ? *(v+3) : 1.0f ;  

             glm::vec4 tvec = mat * vec ;

             *(t+0) = tvec.x ; 
             *(t+1) = tvec.y ; 
             *(t+2) = tvec.z ; 

             if(nk == 4) *(t+3) = tvec.w ;  
        }
    }
    return tr ; 
}








template <typename T> 
std::vector<T>& NPY<T>::data() 
{
    return m_data ;
}

template <typename T> 
std::vector<T>& NPY<T>::vector() 
{
    return m_data ;
}





template <typename T> 
T* NPY<T>::getValues() 
{
    return m_data.data();
}


template <typename T> 
const T* NPY<T>::getValuesConst()  const 
{
    return m_data.data();
}




template <typename T> 
T* NPY<T>::begin()
{
    return m_data.data();
}

template <typename T> 
T* NPY<T>::end()
{
    return m_data.data() + getNumValues(0) ;
}






template <typename T> 
T* NPY<T>::getValues(unsigned i, unsigned j, unsigned k)
{
    unsigned idx = getValueIndex(i,j,k);
    return m_data.data() + idx ;
}


template <typename T> 
const T* NPY<T>::getValuesConst(unsigned i, unsigned j, unsigned k) const 
{
    unsigned idx = getValueIndex(i,j,k);
    return m_data.data() + idx ;
}






template <typename T> 
void* NPY<T>::getBytes() const 
{
    return hasData() ? (void*)getValuesConst() : NULL ;
}

template <typename T> 
void* NPY<T>::getPointer()
{
    return getBytes() ;
}

template <typename T> 
BBufSpec* NPY<T>::getBufSpec()
{
   if(m_bufspec == NULL)
   {
      int id = getBufferId();
      void* ptr = getBytes();
      unsigned int num_bytes = getNumBytes();
      int target = getBufferTarget() ; 
      m_bufspec = new BBufSpec(id, ptr, num_bytes, target);
   }
   return m_bufspec ; 
}

template <typename T> 
T NPY<T>::getValue( int i,  int j,  int k,  int l, int m) const 
{
    unsigned idx = getValueIndex(i,j,k,l,m);
    return getValueFlat(idx); 
}

template <typename T> 
T NPY<T>::getValueFlat(unsigned idx) const 
{
    const T* data_ = getValuesConst();
    return  *(data_ + idx);
}








template <typename T> 
 void NPY<T>::getU( short& value, unsigned short& uvalue, unsigned char& msb, unsigned char& lsb, unsigned int i, unsigned int j, unsigned int k, unsigned int l)
{
    // used for unpacking photon records

    assert(type == SHORT); // pragmatic template specialization, by death if you try to use the wrong one...

    unsigned int index = getValueIndex(i,j,k,l);

    value = m_data[index] ;

    hui_t hui ;
    hui.short_ = value ; 

    uvalue = hui.ushort_ ;

    msb = ( uvalue & 0xFF00 ) >> 8  ;  

    lsb = ( uvalue & 0xFF ) ;  
}

template <typename T> 
ucharfour  NPY<T>::getUChar4( unsigned int i, unsigned int j, unsigned int k, unsigned int l0, unsigned int l1 )
{
    assert(type == SHORT); // OOPS: pragmatic template specialization, by death if you try to use the wrong one... 

    unsigned int index_0 = getValueIndex(i,j,k,l0);
    unsigned int index_1 = getValueIndex(i,j,k,l1);

    hui_t hui_0, hui_1 ;

    hui_0.short_ = m_data[index_0];
    hui_1.short_ = m_data[index_1];

    ucharfour v ; 

    v.x = (hui_0.ushort_ & 0xFF ) ; 
    v.y = (hui_0.ushort_ & 0xFF00 ) >> 8 ; 
    v.z = (hui_1.ushort_ & 0xFF ) ; 
    v.w = (hui_1.ushort_ & 0xFF00 ) >> 8 ; 

    return v ;
}

template <typename T> 
charfour  NPY<T>::getChar4( unsigned int i, unsigned int j, unsigned int k, unsigned int l0, unsigned int l1 )
{
    assert(type == SHORT); // OOPS: pragmatic template specialization, by death if you try to use the wrong one... 

    unsigned int index_0 = getValueIndex(i,j,k,l0);
    unsigned int index_1 = getValueIndex(i,j,k,l1);

    hui_t hui_0, hui_1 ;

    hui_0.short_ = m_data[index_0];
    hui_1.short_ = m_data[index_1];

    charfour v ; 

    v.x = (hui_0.short_ & 0xFF ) ; 
    v.y = (hui_0.short_ & 0xFF00 ) >> 8 ; 
    v.z = (hui_1.short_ & 0xFF ) ; 
    v.w = (hui_1.short_ & 0xFF00 ) >> 8 ; 

    // hmm signbit complications ?
    return v ;
}



template <typename T> 
void NPY<T>::setValue(int i, int j, int k, int l, T value)
{
    bool in_range = unsigned(i) < m_ni ; 
    if(!in_range) LOG(fatal) << " i " << i <<  " m_ni " << m_ni ;     
    assert( in_range ); 
    unsigned int idx = getValueIndex(i,j,k,l);
    setValueFlat(idx, value); 
}


template <typename T> 
void NPY<T>::setAllValue(int j, int k, int l, T value)
{
    for(int i=0 ; unsigned(i) < m_ni ; i++) setValue(i, j, k, l, value); 
}



template <typename T> 
void NPY<T>::setValueFlat(unsigned idx, T value)
{
    T* dat = getValues();
    assert(dat && "must zero() the buffer before can setValue");
    *(dat + idx) = value ;
}






#if defined(_MSC_VER)
// conversion from 'glm::uint' to 'short'
#pragma warning( disable : 4244 )
#endif


// hmm assumes float 
template <typename T> 
void NPY<T>::setPart(const npart& p, unsigned int i)
{
    setQuad( p.q0.f , i, 0, 0 );
    setQuad( p.q1.f , i, 1, 0 );
    setQuad( p.q2.f , i, 2, 0 );
    setQuad( p.q3.f , i, 3, 0 );
}



/**
NPY<T>::setMat4
------------------

CAUTION: branch handling for different array shapes controlled by j_

For the normal case of an array of shape (ni,4,4) with a single 
matrix in each item slot the j_ of -1 is appropriate.

For the special case of an array of shape (ni,nj,4,4) where there 
is more than one matrix for each item slot the j_ value must 
specify which of the nj to fill. 

**/

template <typename T> 
void NPY<T>::setMat4(const glm::mat4& mat, int i, int j_, bool transpose)
{
    T* dat = getValues();

    bool expect = j_ == -1 ? hasItemShape(4,4) : hasItemShape(-1,4,4) ; 
    if(!expect) LOG(fatal) << "unexpected itemShape, shape: " << getShapeString() << " j_ " << j_  ; 
    assert(expect);

    if( j_ == -1)
    { 
        for(unsigned j=0 ; j < 4 ; j++)
        {
            for(unsigned k=0 ; k < 4 ; k++) 
               *(dat + getValueIndex(i,j,k,0)) = transpose ? mat[k][j] : mat[j][k] ;
        }
   }
   else
   {
        unsigned j = j_ ; 
        for(unsigned k=0 ; k < 4 ; k++)
        {
            for(unsigned l=0 ; l < 4 ; l++) 
               *(dat + getValueIndex(i,j,k,l)) = transpose ? mat[l][k] : mat[k][l] ;
        }
   }
}


template <typename T> 
glm::mat4 NPY<T>::getMat4(int i, int j_) const 
{
    bool expect = j_ == -1 ? hasItemShape(4,4) : hasItemShape(-1,4,4) ; 
    if(!expect) LOG(fatal) << "unexpected itemShape, shape: " << getShapeString() << " j_ " << j_  ; 
    assert(expect);

    int j = j_ == -1 ? 0 : j_ ; 

    const T* vals = getValuesConst(i, j);
    return glm::make_mat4(vals);
}


template <typename T> 
glm::tmat4x4<T> NPY<T>::getMat4_(int i, int j_) const 
{
    bool expect = j_ == -1 ? hasItemShape(4,4) : hasItemShape(-1,4,4) ; 
    if(!expect) LOG(fatal) << "unexpected itemShape, shape: " << getShapeString() << " j_ " << j_  ; 
    assert(expect);

    int j = j_ == -1 ? 0 : j_ ; 

    const T* vals = getValuesConst(i, j);
    return glm::make_mat4x4(vals);
}





template <typename T> 
glm::mat4* NPY<T>::getMat4Ptr(int i, int j_) const 
{
    glm::mat4 m = getMat4(i, j_) ; 
    return new glm::mat4(m) ; 
}


template <typename T> 
nmat4pair* NPY<T>::getMat4PairPtr(int i) const 
{
    // return Ptr as including NGLMExt into NPY header
    // causes thrustrap- issues

    assert(hasShape(-1,2,4,4));

    glm::mat4 t = getMat4(i, 0);   
    glm::mat4 v = getMat4(i, 1);

    return new nmat4pair(t, v) ; 
}


template <typename T> 
nmat4triple* NPY<T>::getMat4TriplePtr(int i) const 
{
    assert(hasShape(-1,3,4,4));

    glm::mat4 t = getMat4(i, 0);   
    glm::mat4 v = getMat4(i, 1);
    glm::mat4 q = getMat4(i, 2);

    return new nmat4triple(t, v, q) ; 
}


/*

template <typename T> 
nmat4triple_<T>* NPY<T>::getMat4Triple_Ptr(int i) const 
{
    assert(hasShape(-1,3,4,4));

    glm::tmat4x4<T> t = getMat4_(i, 0);   
    glm::tmat4x4<T> v = getMat4_(i, 1);
    glm::tmat4x4<T> q = getMat4_(i, 2);

    return new nmat4triple_<T>(t, v, q) ; 
}

*/





template <typename T> 
void NPY<T>::setMat4Pair(const nmat4pair* pair, unsigned i )
{
    assert(hasShape(-1,2,4,4));
    assert(pair);

    setMat4(pair->t, i, 0 );
    setMat4(pair->v, i, 1 );
}


template <typename T> 
void NPY<T>::setMat4Triple(const nmat4triple* triple, unsigned i )
{
    assert(hasShape(-1,3,4,4));
    assert(triple);

    setMat4(triple->t, i, 0 );
    setMat4(triple->v, i, 1 );
    setMat4(triple->q, i, 2 );
}


template <typename T> 
void NPY<T>::setString(const char* s, unsigned i, unsigned j, unsigned k )
{
    unsigned nl = getShape(-1); 
    unsigned sl = strlen(s); 

    for(unsigned l=0 ; l < nl ; l++) 
    { 
        char c = l < sl ? s[l] : '\0' ; 
        setValue(i,j,k,l, c  ); 
    }
}

template <typename T> 
const char* NPY<T>::getString(unsigned i, unsigned j, unsigned k )
{
    unsigned nl = getShape(-1); 
    char* s = new char[nl+1] ; 
    for(unsigned l=0 ; l < nl ; l++) 
    { 
        char c = getValue(i,j,k,l);
        s[l] = c ; 
    }
    s[nl] = '\0' ; 
    return s ; 
}


/**
setQuad : same type quad setters
---------------------------------

**/

template <typename T> 
void NPY<T>::setQuad(unsigned i, unsigned j, T x, T y, T z, T w )
{
    glm::tvec4<T> vec(x,y,z,w); 
    setQuad_(vec, i, j);
}
template <typename T> 
 void NPY<T>::setQuad(unsigned i, unsigned j, unsigned k, T x, T y, T z, T w )
{
    glm::tvec4<T> vec(x,y,z,w); 
    setQuad_(vec, i, j, k);
}

template <typename T> 
void NPY<T>::setQuad(const nvec4& f, unsigned int i, unsigned int j, unsigned int k )
{
    glm::vec4 vec(f.x,f.y,f.z,f.w); 
    for(unsigned int l=0 ; l < 4 ; l++) setValue(i,j,k,l, vec[l]); 
}
template <typename T> 
void NPY<T>::setQuad(const glm::vec4& vec, unsigned int i, unsigned int j, unsigned int k )
{
    for(unsigned int l=0 ; l < 4 ; l++) setValue(i,j,k,l, vec[l]); 
}
template <typename T> 
void NPY<T>::setQuad(const glm::ivec4& vec, unsigned int i, unsigned int j, unsigned int k )
{
    for(unsigned int l=0 ; l < 4 ; l++) setValue(i,j,k,l,vec[l]); 
}
template <typename T> 
 void NPY<T>::setQuad(const glm::uvec4& vec, unsigned int i, unsigned int j, unsigned int k )
{
    for(unsigned int l=0 ; l < 4 ; l++) setValue(i,j,k,l,vec[l]); 
}



/**
NPY::setQuad_ getQuad_
------------------------

Use of template vector type ensures avoids any type shifting, so the 
method appropriate for the array type is used.  

Using vector type other than of the array does compile and run and gives
a simply converted version of what is in the array. This is much better
than getting invisible lsb truncation for some values which happens when 
using getQuadF (formerly named getQuad) with a mismatched vector types::

    glm::uvec4 u = af->getQuadF() ;   // SILENT TRUNCATION BUG, DO NOT DO THIS

**/


#ifndef __CUDACC__
template <typename T> 
void NPY<T>::setQuad_(const glm::tvec4<T>& vec, unsigned int i, unsigned int j, unsigned int k) 
{
    for(unsigned int l=0 ; l < 4 ; l++) setValue(i,j,k,l, vec[l]); 
}
#endif

#ifndef __CUDACC__
template <typename T> 
glm::tvec4<T> NPY<T>::getQuad_(unsigned int i, unsigned int j, unsigned int k) const 
{
    glm::tvec4<T> vec ; 
    for(unsigned int l=0 ; l < 4 ; l++) vec[l] = getValue(i,j,k,l); 
    return vec ; 
}
#endif












/**
NPY::setQuadI setQuadU
-------------------------

Union type shifting setters.

**/


template <typename T> 
void NPY<T>::setQuadI(const glm::ivec4& vec,  int i,  int j,  int k )
{
    for(unsigned int l=0 ; l < 4 ; l++) setInt(i,j,k,l,vec[l]); 
}

template <typename T> 
void NPY<T>::setQuadI(const nivec4& vec,  int i,  int j,  int k )
{
    setInt(i,j,k,0,vec.x); 
    setInt(i,j,k,1,vec.y); 
    setInt(i,j,k,2,vec.z); 
    setInt(i,j,k,3,vec.w); 
}

template <typename T> 
void NPY<T>::setQuadU(const glm::uvec4& vec,  int i,  int j,  int k )
{
    for(unsigned int l=0 ; l < 4 ; l++) setUInt(i,j,k,l,vec[l]); 
}

template <typename T> 
 void NPY<T>::setQuadU(const nuvec4& vec,  int i,  int j,  int k )
{
    setUInt(i,j,k,0,vec.x); 
    setUInt(i,j,k,1,vec.y); 
    setUInt(i,j,k,2,vec.z); 
    setUInt(i,j,k,3,vec.w); 
}





/**
NPY<T>::getQuadF
------------------

Former name getQuad changed to getQuadF in attempt to 
avoid future bugs, like the one described below.

The below seems to work fine::

   NPT<unsigned>* a = NPY<unsigned>::load(path); 
   glm::uvec4 id = a->getQuadF(3199);   // SILENT LSB TRUNCATION BUG LURKS 

BUT silent truncation occurs when the value of the unsigned contents 
exceeds (0x1 << 24) = 0x1000000 at which point least significant bits 
start getting lost, because of a silent constricting conversion unsigned->float->unsigned.   

See notes/issues/triplet-id-loosing-offset-index-in-NPY.rst and tests/numpyTest.cc and::

    In [14]: "{0:x} {0}".format(0x1 << 24)
    Out[14]: '1000000 16777216'

    In [10]: np.uint32(np.float32(np.uint32(0+(0x1 << 24)))) == 0+(0x1 << 24)
    Out[10]: True

    In [11]: np.uint32(np.float32(np.uint32(1+(0x1 << 24)))) == 1+(0x1 << 24)
    Out[11]: False

For getting unsigned quads should be doing, eg::

    glm::uvec4 id = a->getQuadU(3199);   
    glm::uvec4 id = a->getQuad_(3199);   
 

Reccommendation is to not use this, instead use templated variant which 
will use the appropriate method for the array typo::

    glm::vec4  fv = af->getQuad_(i); 
    glm::ivec4 iv = ai->getQuad_(i); 
    glm::uvec4 uv = au->getQuad_(i); 


 
**/

template <typename T> 
glm::vec4 NPY<T>::getQuadF( int i,  int j,  int k) const 
{
    glm::vec4 vec ; 
    for(unsigned int l=0 ; l < 4 ; l++) vec[l] = getValue(i,j,k,l); 
    return vec ; 
}







template <typename T> 
nvec4 NPY<T>::getVQuad(unsigned int i, unsigned int j, unsigned int k) const 
{
    nvec4 vec ; 
    vec.x = getValue(i,j,k,0);
    vec.y = getValue(i,j,k,1);
    vec.z = getValue(i,j,k,2);
    vec.w = getValue(i,j,k,3);
    return vec ; 
}

/**
NPY::getQuadI
---------------

Type shifting getter, allowing to pull ints out of a float array.

**/

template <typename T> 
glm::ivec4 NPY<T>::getQuadI( int i,  int j,  int k) const 
{
    glm::ivec4 vec ; 
    for(unsigned l=0 ; l < 4 ; l++) vec[l] = getInt(i,j,k,l);   // changed from getValue Aug18 2018 
    return vec ; 
}

template <typename T> 
glm::uvec4 NPY<T>::getQuadU( int i,  int j,  int k) const 
{
    glm::uvec4 vec ; 
    for(unsigned int l=0 ; l < 4 ; l++) vec[l] = getUInt(i,j,k,l); 
    return vec ; 
}





// type shifting get/set using union trick


template <typename T> 
float NPY<T>::getFloat( int i,  int j,  int k,  int l) const
{
    uif_t uif ; 
    uif.u = 0 ;

    T t = getValue(i,j,k,l);
    switch(type)
    {
        case FLOAT:uif.f = t ; break ; 
        case DOUBLE:uif.f = t ; break ; 
        case SHORT:uif.i = t ; break ; 
        default: assert(0);  break ;
    }
    return uif.f ;
}

template <typename T> 
 void NPY<T>::setFloat( int i,  int j,  int k,  int l, float  value)
{
    uif_t uif ; 
    uif.f = value ;

    T t(0) ;
    switch(type)
    {
        case FLOAT:t = uif.f ; break ; 
        case DOUBLE:t = uif.f ; break ; 
        case SHORT:t = uif.i ; break ; 
        default: assert(0);  break ;
    }
    setValue(i,j,k,l,t); 
}



template <typename T> 
unsigned NPY<T>::getUInt( int i,  int j,  int k,  int l) const 
{
    uif_t uif ; 
    uif.u = 0 ;

    T t = getValue(i,j,k,l);
    switch(type)
    {
        case FLOAT:uif.f = t ; break ; 
        case DOUBLE:uif.f = t ; break ; 
        case SHORT:uif.i = t ; break ; 
        case UINT:uif.u = t ; break ; 
        case INT:uif.i = t ; break ; 
        default: assert(0);  break ;
    }
    return uif.u ;
}


template <typename T> 
void NPY<T>::setUSign(int i, int j, int k, int l, bool value)
{
    unsigned u = getUInt(i, j, k, l ); 
    if( value ) u |= SIGNBIT ; 
    setUInt(i, j, k, l, u ); 
}

template <typename T> 
bool NPY<T>::getUSign(int i, int j, int k, int l) const 
{
    unsigned u = getUInt(i, j, k, l ); 
    bool value = (u & SIGNBIT) != 0 ; 
    return value ;
}


template <typename T> 
void NPY<T>::setUInt(int i, int j, int k, int l, unsigned value)
{
    uif_t uif ; 
    uif.u = value ;

    T t(0) ;
    switch(type)
    {
        case FLOAT:t = uif.f ; break ; 
        case DOUBLE:t = uif.f ; break ; 
        case SHORT:t = uif.i ; break ; 
        case UINT:t = uif.u ; break ; 
        case INT:t = uif.i ; break ; 
        default: assert(0);  break ;
    }
    setValue(i,j,k,l, t); 
}











template <typename T> 
void NPY<T>::bitwiseOrUInt(unsigned int i, unsigned int j, unsigned int k, unsigned int l, unsigned int value)
{
    unsigned current = getUInt(i,j,k,l); 
    unsigned new_value = current | value ; 

    uif_t uif ; 
    uif.u = new_value ;

    T t(0) ;
    switch(type)
    {
        case FLOAT:t = uif.f ; break ; 
        case DOUBLE:t = uif.f ; break ; 
        case SHORT:t = uif.i ; break ; 
        case UINT:t = uif.u ; break ; 
        case INT:t = uif.i ; break ; 
        default: assert(0);  break ;
    }
    setValue(i,j,k,l, t); 
}




template <typename T> 
 int NPY<T>::getInt( int i,  int j,  int k,  int l) const 
{
    uif_t uif ;             // how does union handle different sizes ? 
    uif.u = 0 ; 
    T t = getValue(i,j,k,l);
    switch(type)
    {   
        case FLOAT: uif.f = t ; break;
        case DOUBLE: uif.f = t ; break;
        case SHORT: uif.i = t ; break;
        case UINT: uif.u = t ; break;
        case INT: uif.i = t ; break;
        default: assert(0);   break;
    }
    return uif.i ;
}


/**
NPY<T>::setInt
----------------

This is a type shifting setter that allows an integer value 
to be set into another typed array using union trickery. 
Viewing the array as float gives NaN values or very small/large values.
**/

template <typename T> 
void NPY<T>::setInt( int i,  int j,  int k,  int l, int value)
{
    uif_t uif ; 
    uif.i = value ;

    T t(0) ;
    switch(type)
    {
        case FLOAT:t = uif.f ; break ; 
        case DOUBLE:t = uif.f ; break ; 
        case SHORT:t = uif.i ; break ; 
        case UINT:t = uif.u ; break ; 
        case INT:t = uif.i ; break ; 
        default: assert(0);  break ;
    }
    setValue(i,j,k,l, t); 
}




/*
   size_t dim = shape.size() ;
    std::stringstream ss ;

    int itemcount = shape[0] ;
    for(size_t i=1 ; i < dim ; ++i)
    {   
        ss << shape[i] ;
        if( i < dim - 1 ) ss << ", " ;  // need the space for buffer memcmp matching
    }   
    std::string itemshape = ss.str();

    bool fortran_order = false ;

    // pre-calculate total buffer size including the padded header
    size_t nbytes = aoba::BufferSize<float>(shape[0], itemshape.c_str(), fortran_order  );  

    // allocate frame to hold 
    boost::asio::zmq::frame npy_frame(nbytes);

    size_t wbytes = aoba::BufferSaveArrayAsNumpy<float>( (char*)npy_frame.data(), fortran_order, itemcount, itemshape.c_str(), data.data() );
    assert( wbytes == nbytes );

*/







template <typename T> 
void NPY<T>::copyTo(std::vector<T>& dst )
{
    std::copy(m_data.begin(), m_data.end(), std::back_inserter(dst));
}

template <typename T> 
void NPY<T>::copyTo(std::vector<glm::vec3>& dst )
{
    assert( hasShape(-1,3) );
    for(unsigned i=0 ; i < getShape(0) ; i++)
    {
        glm::vec3 vec  ; 
        vec.x = getValue(i,0,0);
        vec.y = getValue(i,1,0);
        vec.z = getValue(i,2,0);
        dst.push_back(vec); 
    }
}

template <typename T> 
void NPY<T>::copyTo(std::vector<glm::vec4>& dst )
{
    assert( hasShape(-1,4) );
    for(unsigned i=0 ; i < getShape(0) ; i++)
    {
        glm::vec4 vec = getQuadF(i) ; 
        dst.push_back(vec); 
    }
}

template <typename T> 
void NPY<T>::copyTo(std::vector<glm::ivec4>& dst )
{
    assert( hasShape(-1,4) );
    for(unsigned i=0 ; i < getShape(0) ; i++)
    {
        glm::ivec4 vec = getQuadI(i);
        dst.push_back(vec); 
    }
}


/**
NPY::interp
-------------

From NP::interp, see also ~/np/tests/NPInterpTest.cc

**/

template <typename T> 
T NPY<T>::interp(T x) const 
{
    const std::vector<int>& shape = getShapeVector() ; 
    assert( shape.size() == 2 && shape[1] == 2 && shape[0] > 1);  
    unsigned ni = shape[0] ; 

    const T* vv = getValuesConst(); 

    int lo = 0 ;
    int hi = ni-1 ;

    /*   
    std::cout 
         << " NPY::interp "
         << " x " << x 
         << " ni " << ni 
         << " lo " << lo
         << " hi " << hi
         << " vx_lo " << vv[2*lo+0] 
         << " vy_lo " <<  vv[2*lo+1] 
         << " vx_hi " << vv[2*hi+0] 
         << " vy_hi " <<  vv[2*hi+1] 
         << std::endl
         ; 

    */

    if( x <= vv[2*lo+0] ) return vv[2*lo+1] ; 
    if( x >= vv[2*hi+0] ) return vv[2*hi+1] ; 

    while (lo < hi-1)
    {    
        int mi = (lo+hi)/2;
        if (x < vv[2*mi+0]) hi = mi ; 
        else lo = mi;
    }    

    T dy = vv[2*hi+1] - vv[2*lo+1] ;
    T dx = vv[2*hi+0] - vv[2*lo+0] ;
    T y = vv[2*lo+1] + dy*(x-vv[2*lo+0])/dx ;
    return y ;
}





template <typename T> bool NPY<T>::IsReal() // static 
{
    return IsRealType(type); 
}

template <typename T> bool NPY<T>::IsInteger() // static 
{
    return IsIntegerType(type); 
}

template <typename T> bool NPY<T>::IsUnsigned() // static 
{
    return IsUnsignedType(type); 
}

template <typename T> bool NPY<T>::IsChar() // static 
{
    return IsCharType(type); 
}




// template specializations : allow branching on type
template<>
NPYBase::Type_t NPY<float>::type = FLOAT ;
template<>
NPYBase::Type_t NPY<double>::type = DOUBLE ;


template<>
NPYBase::Type_t NPY<int>::type = INT ;

template<>
NPYBase::Type_t NPY<short>::type = SHORT ;

template<>
NPYBase::Type_t NPY<char>::type = CHAR ;



template<>
NPYBase::Type_t NPY<unsigned char>::type = UCHAR ;
template<>
NPYBase::Type_t NPY<unsigned int>::type = UINT ;
template<>
NPYBase::Type_t NPY<unsigned long long>::type = ULONGLONG ;




template<>
//short NPY<short>::UNSET = SHRT_MIN ;
short NPY<short>::UNSET = 0 ;

template<>
//int NPY<int>::UNSET = INT_MIN ;
int NPY<int>::UNSET = 0 ;

template<>
//float  NPY<float>::UNSET = FLT_MAX ;
float  NPY<float>::UNSET = 0 ;

template<>
//double NPY<double>::UNSET = DBL_MAX ;
double NPY<double>::UNSET = 0 ;

template<>
char NPY<char>::UNSET = 0 ;

template<>
unsigned char NPY<unsigned char>::UNSET = 0 ;

template<>
unsigned int NPY<unsigned int>::UNSET = 0 ;

template<>
unsigned long long NPY<unsigned long long>::UNSET = 0 ;





/*
* :google:`move templated class implementation out of header`
* http://www.drdobbs.com/moving-templates-out-of-header-files/184403420

A compiler warning "declaration does not declare anything" was avoided
by putting the explicit template instantiation at the tail rather than the 
head of the implementation.
*/

template class NPY<float>;
template class NPY<double>;
template class NPY<short>;
template class NPY<int>;
template class NPY<char>;
template class NPY<unsigned char>;
template class NPY<unsigned int>;
template class NPY<unsigned long long>;


