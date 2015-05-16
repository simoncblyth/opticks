#include "NPY.hpp"
#include <sstream>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


// ctor takes ownership of a copy of the inputs 
NPY::NPY(std::vector<int>& shape, std::vector<float>& data, std::string& metadata) 
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


NPY::NPY(std::vector<int>& shape, float* data, std::string& metadata) 
         :
         m_buffer_id(-1),
         m_shape(shape),
         m_data(),
         m_metadata(metadata)
{
    m_data.reserve(getNumValues(0));
    read(data);
}

// not expected to work : but needed to get to compile
//  solution is to turn this into templated class
NPY::NPY(std::vector<int>& shape, double* data, std::string& metadata) 
         :
         m_buffer_id(-1),
         m_shape(shape),
         m_data(),
         m_metadata(metadata)
{
    m_data.reserve(getNumValues(0));
    read(data);
}


void NPY::read(void* ptr)
{
    memcpy(m_data.data(), ptr, getNumBytes(0) );
}



void NPY::Summary(const char* msg)
{
    std::string desc = description(msg);
    std::cout << desc << std::endl ; 
}   


std::string NPY::description(const char* msg)
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



std::string NPY::path(const char* typ, const char* tag)
{
    char* TYP = strdup(typ);
    char* p = TYP ;
    while(*p)
    {
       if( *p >= 'a' && *p <= 'z') *p += 'A' - 'a' ;
       p++ ; 
    } 

    char envvar[64];
    snprintf(envvar, 64, "DAE_%s_PATH_TEMPLATE", TYP ); 
    free(TYP); 

    char* tmpl = getenv(envvar) ;
    if(!tmpl) return "missing-template-envvar" ; 
    
    char path_[256];
    snprintf(path_, 256, tmpl, tag );

    return path_ ;   
}


void NPY::save(const char* path)
{
    unsigned int itemcount = getShape(0);    // dimension 0, corresponds to "length/itemcount"
    std::string itemshape = getItemShape(1); // shape of dimensions > 0, corresponds to "item"
    aoba::SaveArrayAsNumpy<float>(path, itemcount, itemshape.c_str(), getFloats()  );
}


std::string NPY::getItemShape(unsigned int ifr)
{
    std::stringstream ss ; 
    for(size_t i=ifr ; i < m_shape.size() ; i++)
    {
        ss << m_shape[i]  ;
        if( i < m_shape.size() - 1) ss << "," ;
    }
    return ss.str(); 
}


NPY* NPY::debugload(const char* path)
{
    std::vector<int> shape ;
    std::vector<float> data ;
    std::string metadata = "{}";

    printf("NPY::debugload [%s]\n", path);

    NPY* npy = NULL ;
    aoba::LoadArrayFromNumpy<float>(path, shape, data );
    npy = new NPY(shape,data,metadata) ;

    return npy ;
}


void NPY::debugdump()
{
    float* data = getFloats() ; 
    for(unsigned int i=0 ; i < 16 ; i++)
    {
         if(i % 4 == 0) printf("\n");
         printf(" %10.4f ", data[i]);
    }
    printf("\n");
}




NPY* NPY::load(const char* path)
{
   /*

    Currently need to save as np.float32 for this to manage to load, do so with::

    In [3]: a.dtype
    Out[3]: dtype('float64')

    In [4]: b = np.array(a, dtype=np.float32)

    np.save("/tmp/slowcomponent.npy", b ) 

   */


    std::vector<int> shape ;
    std::vector<float> data ;
    std::string metadata = "{}";

    LOG(debug) << "NPY::load " << path ; 

    NPY* npy = NULL ;
    try 
    {
        aoba::LoadArrayFromNumpy<float>(path, shape, data );
        npy = new NPY(shape,data,metadata) ;
    } 
    catch(const std::runtime_error& error)
    {
        std::cout << "NPY::load failed for path [" << path << "]" <<  std::endl ; 
    }

    return npy ;
}


NPY* NPY::load(const char* typ, const char* tag)
{
    std::string path = NPY::path(typ, tag);
    return load(path.c_str());
}


NPY* NPY::make_vec3(float* m2w_, unsigned int npo)
{
/*
   Usage example to create debug points in viscinity of a drawable

   npy = NPY::make_vec3(dgeo->getModelToWorldPtr(),100); 
   vgst.add(new VecNPY("vpos",npy,0,0));

*/

    glm::mat4 m2w ;
    if(m2w_) m2w = glm::make_mat4(m2w_);

    std::vector<float> data;

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
    NPY* npy = new NPY(shape,data,metadata) ;
    return npy ;
}



NPY* NPY::make_float4(unsigned int ni, unsigned int nj, float value)
{
    std::string metadata = "{}";
    std::vector<float> data;
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

    NPY* npy = new NPY(shape,data,metadata) ;
    return npy ;
}






unsigned int NPY::getUSum(unsigned int j, unsigned int k)
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




