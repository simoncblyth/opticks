#include <iomanip>
#include <algorithm>
#include <iterator>
#include <limits>  

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

#include "BFile.hh"
#include "BStr.hh"
#include "BBufSpec.hh"

#include "NSlice.hpp"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NPYSpec.hpp"
#include "NPY.hpp"

#include "PLOG.hh"





// ctor takes ownership of a copy of the inputs 

template <typename T>
NPY<T>::NPY(const std::vector<int>& shape, std::vector<T>& data_, std::string& metadata) 
         :
         NPYBase(shape, sizeof(T), type, metadata, data_.size() > 0),
         m_data(data_),      // copies the vector
         m_unset_item(NULL),
         m_bufspec(NULL),
         m_msk(NULL)
{
} 

template <typename T>
NPY<T>::NPY(const std::vector<int>& shape, T* data_, std::string& metadata) 
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
void NPY<T>::setData(T* data_)
{
    assert(data_);
    allocate();
    read(data_);
}

template <typename T>
void NPY<T>::fill( T value)
{
    allocate();
    std::fill(m_data.begin(), m_data.end(), value);
}

template <typename T>
T* NPY<T>::zero()
{
    T* data_ = allocate();
    memset( data_, 0, getNumBytes(0) );
    return data_ ; 
}

template <typename T>
T* NPY<T>::allocate()
{
    //assert(m_data.size() == 0);  tripped when indexing a loaded event

    setHasData(true);
    m_data.resize(getNumValues(0));
    return m_data.data();
    //
    // *reserve* vs *resize*
    //     *reserve* just allocates without changing size, does that matter ? 
    //     most NPY usage just treats m_data as a buffer so not greatly however
    //     there is some use of m_data.size() so using resize
}

template <typename T>
void NPY<T>::deallocate()
{
    setHasData(false);
    m_data.clear();
    setNumItems( 0 );
}

template <typename T>
void NPY<T>::reset()
{
    deallocate();
}

template <typename T>
void NPY<T>::read(void* src)
{
    if(m_data.size() == 0)
    {
        unsigned int nv0 = getNumValues(0) ; 
        LOG(debug) << "NPY<T>::read allocating space now (deferred from earlier) for NumValues(0) " << nv0 ; 
        allocate();
    }
    memcpy(m_data.data(), src, getNumBytes(0) );
}

template <typename T> 
void NPY<T>::write(void* dst )
{
    memcpy( dst, m_data.data(), getNumBytes(0) ); 
}




template <typename T>
T* NPY<T>::grow(unsigned int nitems)
{
    setHasData(true);

    unsigned int origvals = m_data.size() ; 
    unsigned int itemvals = getNumValues(1); 
    unsigned int growvals = nitems*itemvals ; 

    LOG(debug) << "NPY<T>::grow with space for"
              << " nitems " << nitems
              << " itemvals " << itemvals
              << " origvals " << origvals   
              << " growvals " << growvals
              ;

    m_data.resize(origvals + growvals);
    return m_data.data() + origvals ;
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



template <typename T>
void NPY<T>::addItem(NPY<T>* other, unsigned item)      // add another buffer to this one, they must have same itemsize (ie size after 1st dimension)
{
    assert( item < other->getNumItems() );
    unsigned orig = getNumItems();
    unsigned extra = 1 ; 

    bool same= hasSameItemSize(this, other);
    assert(same);

    unsigned itemNumBytes = getNumBytes(1) ; 
    char* itemBytes = (char*)other->getBytes() + itemNumBytes*item; 

    memcpy( grow(extra), itemBytes, itemNumBytes );

    setNumItems( orig + extra );
}
 


template <typename T>
void NPY<T>::add(NPY<T>* other)      // add another buffer to this one, they must have same itemsize (ie size after 1st dimension)
{
    unsigned orig = getNumItems();
    unsigned extra = other->getNumItems() ;

    bool same= hasSameItemSize(this, other);
    assert(same);

    memcpy( grow(extra), other->getBytes(), other->getNumBytes(0) );

    setNumItems( orig + extra );
}

template <typename T>
void NPY<T>::add(const T* values, unsigned int nvals)  
{
    unsigned int orig = getNumItems();
    unsigned int itemsize = getNumValues(1) ;
    assert( nvals % itemsize == 0 && "values adding is restricted to integral multiples of the item size");
    unsigned int extra = nvals/itemsize ; 
    memcpy( grow(extra), values, nvals*sizeof(T) );
    setNumItems( orig + extra );
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
void NPY<T>::minmax(T& mi_, T& mx_)
{
    unsigned int nv = getNumValues(0);
    T* vv = getValues();

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
void NPY<T>::minmax_strided(T& mi_, T& mx_, unsigned stride, unsigned offset)
{
    unsigned int nv = getNumValues(0);
    assert( nv % stride == 0);
    assert( offset < stride);

    T* vv = getValues();

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
void NPY<T>::minmax3(ntvec3<T>& mi_, ntvec3<T>& mx_)
{
    minmax_strided( mi_.x , mx_.x,  3, 0 );
    minmax_strided( mi_.y , mx_.y,  3, 1 );
    minmax_strided( mi_.z , mx_.z,  3, 2 );
}

template <typename T>
void NPY<T>::minmax4(ntvec4<T>& mi_, ntvec4<T>& mx_)
{
    minmax_strided( mi_.x , mx_.x,  4, 0 );
    minmax_strided( mi_.y , mx_.y,  4, 1 );
    minmax_strided( mi_.z , mx_.z,  4, 2 );
    minmax_strided( mi_.w , mx_.w,  4, 3 );
}


template <typename T>
void NPY<T>::minmax(std::vector<T>& min_,  std::vector<T>& max_)
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
ntrange3<T> NPY<T>::minmax3()
{
    ntrange3<T> r ; 
    minmax3( r.min , r.max );
    return r ; 
}

template <typename T>
ntrange4<T> NPY<T>::minmax4()
{
    ntrange4<T> r ; 
    minmax4( r.min , r.max );
    return r ; 
}




template <typename T>
bool NPY<T>::isConstant(T val)
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
T NPY<T>::maxdiff(NPY<T>* other, bool dump_)
{
    unsigned int nv = getNumValues(0);
    unsigned int no = getNumValues(0);
    assert( no == nv);
    T* v_ = getValues();
    T* o_ = other->getValues();

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
        LOG(trace) <<  "NPY<T>::loadFromBuffer before aoba " ; 
         
        aoba::BufferLoadArrayFromNumpy<T>( bytes, size, shape, data );

        LOG(trace) <<  "NPY<T>::loadFromBuffer after aoba " ; 

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

        LOG(trace) <<  "NPY<T>::load before aoba " ; 
         
        aoba::LoadArrayFromNumpy<T>(path.c_str(), shape, data );

        LOG(trace) <<  "NPY<T>::load after aoba " ; 

        npy = new NPY<T>(shape,data,metadata) ;
    }
    catch(const std::runtime_error& /*error*/)
    {
        if(!quietly)
        {
        LOG(warning) << "NPY<T>::load failed for path [" << path << "] use debugload to see why"  ; 
        }
    }

    return npy ;
}



template <typename T>
NPY<T>* NPY<T>::load(const char* dir, const char* name, bool quietly)
{
    std::string path = NPYBase::path(dir, name);
    return load(path.c_str(), quietly);
}


template <typename T>
void NPY<T>::save(const char* dir, const char* name)
{
    std::string path_ = NPYBase::path(dir, name);
    save(path_.c_str());
}

template <typename T>
void NPY<T>::save(const char* dir, const char* reldir, const char* name)
{
    std::string path_ = NPYBase::path(dir, reldir, name);
    save(path_.c_str());
}



template <typename T>
bool NPY<T>::exists(const char* dir, const char* name)
{
    std::string path_ = NPYBase::path(dir, name);
    return exists(path_.c_str());
}



template <typename T>
NPY<T>* NPY<T>::load(const char* tfmt, const char* source, const char* tag, const char* det, bool quietly)
{
    //  (ox,cerenkov,1,dayabay)  ->   (dayabay,cerenkov,1,ox)
    //
    //     arg order twiddling done here is transitional to ease the migration 
    //     once working in the close to old arg order, can untwiddling all the calls
    //
    std::string path = NPYBase::path(det, source, tag, tfmt );
    return load(path.c_str(),quietly);
}
template <typename T>
void NPY<T>::save(const char* tfmt, const char* source, const char* tag, const char* det)
{
    std::string path_ = NPYBase::path(det, source, tag, tfmt );
    save(path_.c_str());
}



template <typename T>
bool NPY<T>::exists(const char* tfmt, const char* source, const char* tag, const char* det)
{
    std::string path_ = NPYBase::path(det, source, tag, tfmt );
    return exists(path_.c_str());
}



template <typename T>
bool NPY<T>::exists(const char* path_)
{
    fs::path _path(path_);
    return fs::exists(_path) && fs::is_regular_file(_path); 
}





template <typename T>
void NPY<T>::save(const char* raw)
{
    std::string native = BFile::FormPath(raw);   // potentially with prefixing/windozing 

    if(m_verbose || GLOBAL_VERBOSE) 
    {
        LOG(info) << "NPY::save raw    np.load(\"" << raw << "\") " ; 
        LOG(info) << "NPY::save native np.load(\"" << native << "\") " ; 
    }

    fs::path _path(native);
    fs::path dir = _path.parent_path();

    if(!fs::exists(dir))
    {   
        LOG(info)<< "NPYBase::save creating directories [" << dir.string() << "]" << raw ;
        if (fs::create_directories(dir))
        {   
            LOG(info)<< "NPYBase::save created directories [" << dir.string() << "]" ;
        }   
    }   
    else
    {
        if(m_verbose || GLOBAL_VERBOSE) 
        {
            LOG(info) << "NPY::save dir exists \"" << _path.string() << "\" " ; 
        }
    }



    unsigned int itemcount = getShape(0);    // dimension 0, corresponds to "length/itemcount"
    std::string itemshape = getItemShape(1); // shape of dimensions > 0, corresponds to "item"

    T* values = getValues();

    bool allow_save_empty = false ; // <-- causes flakey segfaults inside aoba when true 

    if(values == NULL && !allow_save_empty )
    {
         LOG(fatal) << "NPY values NULL, SKIP attempt to save  " 
                    << " itemcount " << itemcount
                    << " itemshape " << itemshape
                    << " native " << native 
                    ; 
    }
    else
    {
        aoba::SaveArrayAsNumpy<T>(native.c_str(), itemcount, itemshape.c_str(), values );
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



template <typename T>
NPY<T>* NPY<T>::make(NPYSpec* argspec)
{
    std::vector<int> shape ; 
    for(unsigned int x=0 ; x < 4 ; x++)
    {
        unsigned int nx = argspec->getDimension(x) ;
        if(x == 0 || nx > 0) shape.push_back(nx) ;
        // only 1st dimension zero is admissable
    }

    NPY<T>* npy = make(shape);
    NPYSpec* npyspec = npy->getShapeSpec(); 

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


template <typename T>
NPY<T>* NPY<T>::make_repeat(NPY<T>* src, unsigned int n)
{
    unsigned int ni = src->getShape(0);
    assert( ni > 0);

    std::vector<int> dshape(src->getShapeVector());
    dshape[0] *= n ;          // bump up first dimension

    NPY<T>* dst = NPY<T>::make(dshape) ;
    dst->zero();


    unsigned int size = src->getNumBytes(1);  // item size in bytes (from dimension 1)  

    char* sbytes = (char*)src->getBytes();
    char* dbytes = (char*)dst->getBytes();

    assert(size == dst->getNumBytes(1)) ;
   
    for(unsigned int i=0 ; i < ni ; i++){
    for(unsigned int r=0 ; r < n ;  r++){

        memcpy( (void*)(dbytes + i*n*size + r*size ),(void*)(sbytes + size*i), size ) ; 

    }
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
 
template <typename T>
unsigned NPY<T>::_copy_masked(NPY<T>* dst, NPY<T>* src, NPY<unsigned>* msk )
{
    // copy items from src to dst that are pointed to by msk 

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

template <typename T>
NPY<T>* NPY<T>::make_selection(NPY<T>* src, unsigned jj, unsigned kk, unsigned mask )
{
    // CPU equivalent of thrust stream compaction
    // this is selecting items from src based the (jj,kk) element of
    // the item ANDing with the mask 

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
NPY<T>* NPY<T>::make_like(NPY<T>* src)
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

     for(unsigned i=0 ; i < ni ; i++)
     {
         glm::mat4 t = src->getMat4(i);  
         glm::mat4 v = nglmext::invert_trs( t );

         dst->setMat4(t, i, 0, transpose );
         dst->setMat4(v, i, 1, transpose );
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

     for(unsigned i=0 ; i < ni ; i++)
     {
         glm::mat4 t = src->getMat4(i);  
         glm::mat4 v = nglmext::invert_trs( t );
         glm::mat4 q = glm::transpose( v ) ;

         dst->setMat4(t, i, 0 );
         dst->setMat4(v, i, 1 );
         dst->setMat4(q, i, 2 );
     }
     return dst ; 
}



template <typename T>
NPY<T>* NPY<T>::make_identity_transforms(unsigned n)
{
     NPY<T>* dst = NPY<T>::make(n,4,4);
     dst->zero();
     glm::mat4 identity(1.0f);
     for(unsigned i=0 ; i < n ; i++) dst->setMat4(identity, i, -1, false);
     return dst ; 
}






template <typename T>
NPY<T>* NPY<T>::clone()
{
    return NPY<T>::copy(this) ;
}

template <typename T>
NPY<T>* NPY<T>::copy(NPY<T>* src)
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
NPY<T>* NPY<T>::make_modulo(NPY<T>* src, unsigned int scaledown)
{
    std::vector<T>& sdata = src->data();
    std::vector<T>  ddata ; 

    unsigned int ni = src->getShape(0) ;
    unsigned int nj = src->getShape(1) ;
    unsigned int nk = src->getShape(2) ;

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

    LOG(trace) << "NPY::make_slice from " 
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
unsigned int NPY<T>::getUSum(unsigned int j, unsigned int k)
{
    unsigned int ni = m_ni ;
    unsigned int nj = m_nj ;
    unsigned int nk = m_nk ;

    assert(m_dim == 3 && j < nj && k < nk);

    unsigned int usum = 0 ; 
    uif_t uif ; 
    for(unsigned int i=0 ; i<ni ; i++ )
    {
        unsigned int index = i*nj*nk + j*nk + k ;
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


    LOG(trace) << "NPY<T>::dump " 
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

                LOG(trace) << "NPY<T>::dump " 
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


   LOG(trace) << "NPY<T>::dump DONE " ; 

   std::cout << std::endl ; 
}





template <typename T>
NPY<T>* NPY<T>::scale(float factor)
{ 
   glm::mat4 m = glm::scale(glm::mat4(1.0f), glm::vec3(factor));
   return transform(m);
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






//template <typename T> 
// unsigned int NPY<T>::getNumValues()
//{
//    return m_data.size();
//}


template <typename T> 
T* NPY<T>::getValues(unsigned int i, unsigned int j)
{
    unsigned int idx = getValueIndex(i,j,0);
    return m_data.data() + idx ;
}


template <typename T> 
const T* NPY<T>::getValuesConst(unsigned int i, unsigned int j) const 
{
    unsigned int idx = getValueIndex(i,j,0);
    return m_data.data() + idx ;
}






template <typename T> 
void* NPY<T>::getBytes()
{
    return hasData() ? (void*)getValues() : NULL ;
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
 T NPY<T>::getValue(unsigned int i, unsigned int j, unsigned int k, unsigned int l)
{
    unsigned int idx = getValueIndex(i,j,k,l);
    T* data_ = getValues();
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




/*
template <typename T> 
 void NPY<T>::setValue(unsigned int i, unsigned int j, unsigned int k, T value)
{
    unsigned int idx = getValueIndex(i,j,k);
    T* dat = getValues();
    assert(dat && "must zero() the buffer before can setValue");
    *(dat + idx) = value ;
}
*/


template <typename T> 
 void NPY<T>::setValue(unsigned int i, unsigned int j, unsigned int k, unsigned int l, T value)
{
    unsigned int idx = getValueIndex(i,j,k,l);
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


 

template <typename T> 
void NPY<T>::setMat4(const glm::mat4& mat, int i, int j_, bool transpose)
{
    T* dat = getValues();

    // have to use j_ == -1, to indicate no such index
    // as the usual approach of using default 0 wont work here  
    // as depend on the 4,4 shape of the rest of the indices 

    if( j_ == -1)
    { 
        assert(hasItemShape(4,4));
        for(unsigned j=0 ; j < 4 ; j++)
        {
            for(unsigned k=0 ; k < 4 ; k++) 
               *(dat + getValueIndex(i,j,k,0)) = transpose ? mat[k][j] : mat[j][k] ;
        }
   }
   else
   {
        assert(hasItemShape(-1,4,4));
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
    if(j_ == -1) assert(hasItemShape(4,4));
    else         assert(hasItemShape(-1,4,4));

    int j = j_ == -1 ? 0 : j_ ; 

    const T* vals = getValuesConst(i, j);
    return glm::make_mat4(vals);
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








// same type quad setters
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


template <typename T> 
 void NPY<T>::setQuad(unsigned int i, unsigned int j, float x, float y, float z, float w )
{
    glm::vec4 vec(x,y,z,w); 
    setQuad(vec, i, j);
}
template <typename T> 
 void NPY<T>::setQuad(unsigned int i, unsigned int j, unsigned int k, float x, float y, float z, float w )
{
    glm::vec4 vec(x,y,z,w); 
    setQuad(vec, i, j, k);
}


// type shifting quad setters
template <typename T> 
 void NPY<T>::setQuadI(const glm::ivec4& vec, unsigned int i, unsigned int j, unsigned int k )
{
    for(unsigned int l=0 ; l < 4 ; l++) setInt(i,j,k,l,vec[l]); 
}
template <typename T> 
 void NPY<T>::setQuadU(const glm::uvec4& vec, unsigned int i, unsigned int j, unsigned int k )
{
    for(unsigned int l=0 ; l < 4 ; l++) setUInt(i,j,k,l,vec[l]); 
}

template <typename T> 
 void NPY<T>::setQuadU(const nuvec4& vec, unsigned int i, unsigned int j, unsigned int k )
{
    setUInt(i,j,k,0,vec.x); 
    setUInt(i,j,k,1,vec.y); 
    setUInt(i,j,k,2,vec.z); 
    setUInt(i,j,k,3,vec.w); 
}







template <typename T> 
 glm::vec4 NPY<T>::getQuad(unsigned int i, unsigned int j, unsigned int k)
{
    glm::vec4 vec ; 
    for(unsigned int l=0 ; l < 4 ; l++) vec[l] = getValue(i,j,k,l); 
    return vec ; 
}


template <typename T> 
nvec4 NPY<T>::getVQuad(unsigned int i, unsigned int j, unsigned int k)
{
    nvec4 vec ; 
    vec.x = getValue(i,j,k,0);
    vec.y = getValue(i,j,k,1);
    vec.z = getValue(i,j,k,2);
    vec.w = getValue(i,j,k,3);
    return vec ; 
}


template <typename T> 
 glm::ivec4 NPY<T>::getQuadI(unsigned int i, unsigned int j, unsigned int k)
{
    glm::ivec4 vec ; 
    for(unsigned int l=0 ; l < 4 ; l++) vec[l] = getValue(i,j,k,l); 
    return vec ; 
}

template <typename T> 
 glm::uvec4 NPY<T>::getQuadU(unsigned int i, unsigned int j, unsigned int k)
{
    glm::uvec4 vec ; 
    for(unsigned int l=0 ; l < 4 ; l++) vec[l] = getUInt(i,j,k,l); 
    return vec ; 
}





// type shifting get/set using union trick


template <typename T> 
 float NPY<T>::getFloat(unsigned int i, unsigned int j, unsigned int k, unsigned int l)
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
 void NPY<T>::setFloat(unsigned int i, unsigned int j, unsigned int k, unsigned int l, float  value)
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
 unsigned int NPY<T>::getUInt(unsigned int i, unsigned int j, unsigned int k, unsigned int l)
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
 void NPY<T>::setUInt(unsigned int i, unsigned int j, unsigned int k, unsigned int l, unsigned int value)
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
 int NPY<T>::getInt(unsigned int i, unsigned int j, unsigned int k, unsigned int l)
{
    uif_t uif ;             // how does union handle different sizes ? 
    uif.u = 0 ; 
    T t = getValue(i,j,k,l);
    switch(type)
    {   
        case FLOAT: uif.f = t ; break;
        case DOUBLE: uif.f = t ; break;
        case SHORT: uif.i = t ; break;
        default: assert(0);   break;
    }
    return uif.i ;
}

template <typename T> 
 void NPY<T>::setInt(unsigned int i, unsigned int j, unsigned int k, unsigned int l, int value)
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
        glm::vec4 vec = getQuad(i) ; 
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














// template specializations : allow branching on type
template<>
NPYBase::Type_t NPY<float>::type = FLOAT ;
template<>
NPYBase::Type_t NPY<double>::type = DOUBLE ;


template<>
NPYBase::Type_t NPY<int>::type = INT ;

template<>
NPYBase::Type_t NPY<short>::type = SHORT ;
//template<>
//NPYBase::Type_t NPY<char>::type = CHAR ;



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

//template<>
//char NPY<char>::UNSET = 0 ;

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
//template class NPY<char>;
template class NPY<unsigned char>;
template class NPY<unsigned int>;
template class NPY<unsigned long long>;


