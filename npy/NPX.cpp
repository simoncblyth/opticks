#include "PLOG.hh"
#include <iostream>
#include "NPX.hpp"

template <typename T>
const plog::Severity NPX<T>::LEVEL = PLOG::EnvLevel("NPX", "DEBUG") ; 

template <typename T>
NPX<T>* NPX<T>::make(unsigned int ni, unsigned int nj, unsigned int nk)
{
    std::vector<int> shape ; 
    shape.push_back(ni);
    shape.push_back(nj);
    shape.push_back(nk);
    return make(shape);
}

template <typename T>
NPX<T>* NPX<T>::make(const std::vector<int>& shape)
{
    T* values = NULL ;
    std::string metadata = "{}";
    NPX<T>* npy = new NPX<T>(shape,values,metadata) ;
    return npy ; 
}


template <typename T>
NPX<T>::NPX(const std::vector<int>& shape, const T* data_, std::string& metadata) 
    :
    NPYBase(shape, sizeof(T), FLOAT, metadata, data_ != NULL),
    m_data(new std::vector<T>())
{
    if(data_) 
    {
        setData(data_);
    }
}

template <typename T>
void NPX<T>::setData(const T* data_)
{
    LOG(LEVEL); 
    assert(data_);
    allocate();
    read(data_);
}

template <typename T>
T* NPX<T>::allocate()
{
    unsigned num_vals = getNumValues(0) ;
    LOG(LEVEL) << " num_vals " << num_vals ; 
    assert(m_data->size() == 0);  
    setHasData(true);
    m_data->resize(num_vals);

    void* old_base_ptr = getBasePtr(); 
    void* new_base_ptr = m_data->data() ;

    if( new_base_ptr != old_base_ptr )
    {
        std::cout << "base_ptr moved " << std::endl ; 
    }
    else
    {
        std::cout << "base_ptr same " << std::endl ; 
    }

    setBasePtr(new_base_ptr);
    return (T*)new_base_ptr ;
}

template <typename T>
void NPX<T>::read(const void* src)
{
    if(m_data->size() == 0)
    {
        allocate();
    }
    unsigned num_bytes = getNumBytes(0) ;
    LOG(LEVEL) << "num_bytes " << num_bytes ; 
    memcpy(m_data->data(), src, num_bytes );
}

template <typename T>
void NPX<T>::deallocate()
{
    //m_data->clear();
    //m_data->shrink_to_fit(); 

    LOG(LEVEL) << "deleting m_data" ; 
    delete m_data ; 
    m_data = NULL ; 
    m_data = new std::vector<T>(); 

    setHasData(false);
    setBasePtr(NULL); 
    setNumItems( 0 );
}

template <typename T>
void NPX<T>::reset()
{
    LOG(LEVEL) ; 
    deallocate();
}


template <typename T>
void NPX<T>::add(const T* values, unsigned int nvals)  
{
    //LOG(LEVEL) << "nvals " << nvals ; 
    unsigned int orig = getNumItems();

    int reservation = getReservation(); 
    if(orig == 0 && hasReservation())
    {    
        LOG(info) << "adding on empty : setting reservation " << reservation ; 
        reserve(getReservation()); 
    }    

    unsigned int itemsize = getNumValues(1) ;
    assert( nvals % itemsize == 0 && "values adding is restricted to integral multiples of the item size");
    unsigned int extra = nvals/itemsize ; 
    memcpy( grow(extra), values, nvals*sizeof(T) );
    setNumItems( orig + extra );
}

template <typename T>
T* NPX<T>::grow(unsigned nitems)
{
    //LOG(LEVEL) ; 
    setHasData(true);

    unsigned origvals = m_data->size() ; 
    unsigned itemvals = getNumValues(1); 
    unsigned growvals = nitems*itemvals ; 

    unsigned new_size = origvals + growvals ; 
    m_data->resize(new_size);  // <--- CAUTION this can cause a change to the base ptr, as might need to be relocated to be contiguous

    void* old_base_ptr  = getBasePtr();   
    void* new_base_ptr = (void*)m_data->data() ; 
    setBasePtr(new_base_ptr);

    if(old_base_ptr != new_base_ptr) 
    {
        std::cout 
            << "NPX<T>::grow base_ptr shift " 
            << getShapeString()
            << " m_data->size() " << m_data->size() 
            << " m_data->capacity() " << m_data->capacity() 
            << std::endl 
            ; 
    }

    return m_data->data() + origvals ;
}


/**

reserve
-----------

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
void NPX<T>::reserve(unsigned items)
{
    unsigned itemvals = getNumValues(1); 
    unsigned vals = items*itemvals ; 
    unsigned cap0 = m_data->capacity(); 

    m_data->reserve(vals); 

    unsigned cap1 = m_data->capacity(); 

    std::cout 
        << " reserve " 
        << " items " << items
        << " itemvals " << itemvals
        << " vals " << vals
        << " cap0 " << cap0
        << " cap1 " << cap1
        << std::endl 
        ; 
    
}



template <typename T>
void NPX<T>::zero()
{
}

template <typename T>
void NPX<T>::save(const char* path) const
{
}

template <typename T>
void NPX<T>::save(const char* dir, const char* name) const
{
}

template <typename T>
void NPX<T>::save(const char* pfx, const char* tfmt, const char* targ, const char* tag, const char* det ) const 
{
}

template <typename T>
void* NPX<T>::getBytes() const
{
    return NULL ; 
}

template <typename T>
void NPX<T>::setQuad(const glm::vec4& vec, unsigned int i, unsigned int j, unsigned int k)
{
}

template <typename T>
void NPX<T>::setQuad(const glm::ivec4& vec, unsigned int i, unsigned int j, unsigned int k) 
{
}

template <typename T>
glm::vec4  NPX<T>::getQuadF( int i,  int j,  int k ) const 
{
    glm::vec4 v(0.f,0.f,0.f,0.f); 
    return v ; 
}

template <typename T>
glm::ivec4  NPX<T>::getQuadI( int i,  int j,  int k ) const 
{
    glm::ivec4 iv(0,0,0,0); 
    return iv ; 
}

template struct NPX<float>;

