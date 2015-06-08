#include "NPY.hpp"
#include <sstream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;



// ctor takes ownership of a copy of the inputs 

template <typename T>
NPY<T>::NPY(std::vector<int>& shape, std::vector<T>& data, std::string& metadata) 
         :
         m_buffer_id(-1),
         m_shape(shape),
         m_data(data),
         m_metadata(metadata)
{
    m_len0 = getShape(0);
    m_len1 = getShape(1);
    m_len2 = getShape(2);
    m_dim  = m_shape.size();
} 


template <typename T>
NPY<T>::NPY(std::vector<int>& shape, T* data, std::string& metadata) 
         :
         m_buffer_id(-1),
         m_shape(shape),
         m_data(),
         m_metadata(metadata)
{
    m_data.reserve(getNumValues(0));
    read(data);
}


template <typename T>
void NPY<T>::read(void* ptr)
{
    memcpy(m_data.data(), ptr, getNumBytes(0) );
}


template <typename T>
void NPY<T>::Summary(const char* msg)
{
    std::string desc = description(msg);
    std::cout << desc << std::endl ; 
}   

template <typename T>
std::string NPY<T>::description(const char* msg)
{
    std::stringstream ss ; 

    ss << msg << " (" ;

    for(size_t i=0 ; i < m_shape.size() ; i++)
    {
        ss << m_shape[i]  ;
        if( i < m_shape.size() - 1) ss << "," ;
    }
    ss << ") " ;
    ss << " len0 " << m_len0 ;
    ss << " len1 " << m_len1 ;
    ss << " len2 " << m_len2 ;
    ss << " nfloat " << m_data.size() << " " ;

    ss << " getNumBytes(0) " << getNumBytes(0) ;
    ss << " getNumBytes(1) " << getNumBytes(1) ;
    ss << " getNumValues(0) " << getNumValues(0) ;
    ss << " getNumValues(1) " << getNumValues(1) ;

    ss << m_metadata  ;

    return ss.str();
}


template <typename T>
std::string NPY<T>::path(const char* typ, const char* tag)
{
    char* TYP = strdup(typ);
    char* p = TYP ;
    while(*p)
    {
       if( *p >= 'a' && *p <= 'z') *p += 'A' - 'a' ;
       p++ ; 
    } 


    // NB envvars and valid typ are defined in env/export-
    char envvar[64];
    snprintf(envvar, 64, "DAE_%s_PATH_TEMPLATE", TYP ); 
    free(TYP); 

    char* tmpl = getenv(envvar) ;
    if(!tmpl)
    {
         LOG(fatal)<< "NPY<T>::path missing envvar for "
                   << " typ " << typ 
                   << " envvar " << envvar  
                   << " define new typs with env-;export-;export-vi " ; 
         assert(0);
         return "missing-template-envvar" ; 
    }

    char path_[256];
    snprintf(path_, 256, tmpl, tag );

    return path_ ;   
}


template <typename T>
void NPY<T>::save(const char* tfmt, const char* targ, const char* tag )
{
    char typ[64];
    snprintf(typ, 64, tfmt, targ ); 
    save(typ, tag);
}

template <typename T>
void NPY<T>::save(const char* typ, const char* tag)
{
    std::string path = NPY<T>::path(typ, tag);
    save(path.c_str());
}

template <typename T>
void NPY<T>::save(const char* path_)
{
    fs::path path(path_);
    fs::path dir = path.parent_path();

    if(!fs::exists(dir))
    {   
        LOG(info)<< "NPY<T>::save creating directory [" << dir.string() << "]" << path_ ;
        if (fs::create_directory(dir))
        {   
            LOG(info)<< "NPY<T>::save created directory [" << dir.string() << "]" ;
        }   
    }   

    unsigned int itemcount = getShape(0);    // dimension 0, corresponds to "length/itemcount"
    std::string itemshape = getItemShape(1); // shape of dimensions > 0, corresponds to "item"
    aoba::SaveArrayAsNumpy<T>(path_, itemcount, itemshape.c_str(), getValues()  );
}

template <typename T>
std::string NPY<T>::getItemShape(unsigned int ifr)
{
    std::stringstream ss ; 
    for(size_t i=ifr ; i < m_shape.size() ; i++)
    {
        ss << m_shape[i]  ;
        if( i < m_shape.size() - 1) ss << "," ;
    }
    return ss.str(); 
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
void NPY<T>::debugdump()
{
   /*
    T* data = getValues() ; 
    for(unsigned int i=0 ; i < 16 ; i++)
    {
         if(i % 4 == 0) printf("\n");
         switch(type)
         {
             case 'f':printf(" %10.4f ", data[i]);break;
             case 's':printf(" %10.4f ", data[i]);break;
         }
    }
    printf("\n");
   */
}



template <typename T>
NPY<T>* NPY<T>::load(const char* path)
{
   /*
    Currently need to save as np.float32 for this to manage to load, do so with::

    In [3]: a.dtype
    Out[3]: dtype('float64')

    In [4]: b = np.array(a, dtype=np.float32)

    np.save("/tmp/slowcomponent.npy", b ) 
   */

    std::vector<int> shape ;
    std::vector<T> data ;
    std::string metadata = "{}";

    LOG(debug) << "NPY<T>::load " << path ; 

    NPY* npy = NULL ;
    try 
    {
        aoba::LoadArrayFromNumpy<T>(path, shape, data );
        npy = new NPY<T>(shape,data,metadata) ;
    } 
    catch(const std::runtime_error& error)
    {
        std::cout << "NPY<T>::load failed for path [" << path << "]" <<  std::endl ; 
    }

    return npy ;
}


template <typename T>
NPY<T>* NPY<T>::load(const char* typ, const char* tag)
{
    std::string path = NPY<T>::path(typ, tag);
    return load(path.c_str());
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

    for(int i=0 ; i < npo ; i++ )
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
NPY<T>* NPY<T>::make_vec4(unsigned int ni, unsigned int nj, T value)
{
    std::string metadata = "{}";
    std::vector<T> data;
    std::vector<int> shape ; 

    unsigned int nk = 4 ;
 
    shape.push_back(ni);
    shape.push_back(nj);
    shape.push_back(nk);

    for(int i=0 ; i < ni ; i++ ){
    for(int j=0 ; j < nj ; j++ ){
    for(int k=0 ; k < nk ; k++ )
    { 
        data.push_back(value);
    }
    } 
    }

    NPY<T>* npy = new NPY<T>(shape,data,metadata) ;
    return npy ;
}





template <typename T>
unsigned int NPY<T>::getUSum(unsigned int j, unsigned int k)
{
    unsigned int ni = m_len0 ;
    unsigned int nj = m_len1 ;
    unsigned int nk = m_len2 ;

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
    unsigned int ni = m_len0 ;
    unsigned int nj = m_len1 ;
    unsigned int nk = m_len2 ;
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
std::map<int,int> NPY<T>::count_uniquei(unsigned int j, unsigned int k, int sj, int sk )
{
    unsigned int ni = m_len0 ;
    unsigned int nj = m_len1 ;
    unsigned int nk = m_len2 ;
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



// template specializations 
template<>
const char NPY<float>::type = FLOAT ;
template<>
const char NPY<short>::type = SHORT ;
template<>
const char NPY<double>::type = DOUBLE ;



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



/*
// http://stackoverflow.com/questions/6448653/c-how-to-use-type-in-template-function-to-branch

template<typename T>
struct switch_value {};

template<>
struct switch_value<int>
{
    enum { value = 1 };
};

template<>
struct switch_value<char>
{
    enum { value = 2 };
};

*/

