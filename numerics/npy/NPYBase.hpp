#pragma once

#include <vector>
#include <string>
#include "assert.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


class NPYBase {
   public:
       typedef enum { FLOAT, SHORT, DOUBLE, INT, UINT, CHAR, UCHAR, ULONGLONG} Type_t ;
       static const char* DEFAULT_DIR_TEMPLATE  ; 
   public:
        NPYBase(std::vector<int>& shape, unsigned char sizeoftype, Type_t type, std::string& metadata, bool has_data);
        void setHasData(bool has_data=true);
        bool hasData();
   public:
       // shape related
       std::vector<int>& getShapeVector();
       std::string  getItemShape(unsigned int ifr=1);
       std::string  getDigestString();
       static std::string  getDigestString(void* bytes, unsigned int nbytes);
       bool isEqualTo(void* bytes, unsigned int nbytes);
       bool isEqualTo(NPYBase* other);

       //unsigned int getLength();
       unsigned int getNumItems(int ifr=0, int ito=1);  // default ifr/ito=0/1 is size of 1st dimension
       unsigned int getNumElements();   // size of last dimension
       unsigned int getDimensions();
       std::string  getShapeString(unsigned int ifr=0);
       unsigned int getShape(unsigned int dim);
       unsigned int getValueIndex(unsigned int i, unsigned int j, unsigned int k, unsigned int l=0);
       unsigned int getNumValues(unsigned int from_dim=0);
   public:
       // depending on sizeoftype
       Type_t        getType();
       unsigned char getSizeOfType();
       unsigned int  getNumBytes(unsigned int from_dim=0);
       unsigned int  getByteIndex(unsigned int i, unsigned int j, unsigned int k, unsigned int l=0);
   public:
       void reshape(int ni, unsigned int nj=0, unsigned int nk=0, unsigned int nl=0);
   private:
       void init();
       void updateDimensions();
   public:
       // OpenGL related
       void         setBufferId(int buffer_id);
       int          getBufferId();  // either -1 if not uploaded, or the OpenGL buffer Id

       void         setBufferTarget(int buffer_target);
       int          getBufferTarget();  // -1 if unset

       void         setAux(void* aux);
       void*        getAux();
       void         setDynamic(bool dynamic=true);
       bool         isDynamic();    // used by oglrap-/Rdr::upload
   public:
       // NumPy persistency
       static std::string directory(const char* tfmt, const char* targ, const char* det);
       static std::string directory(const char* typ, const char* det);

       static std::string path(const char* dir, const char* name);
       static std::string path(const char* typ, const char* tag, const char* det);
       static std::string path(const char* pfx, const char* gen, const char* tag, const char* det);
       void setVerbose(bool verbose=true);
       void setAllowPrealloc(bool allow=true); 

   public:
       // provided by subclass
       virtual void read(void* ptr) = 0;
       virtual void* getBytes() = 0 ;

       virtual void setQuad(const glm::vec4& vec, unsigned int i, unsigned int j, unsigned int k) = 0 ;
       virtual void setQuad(const glm::ivec4& vec, unsigned int i, unsigned int j, unsigned int k) = 0 ;

       virtual glm::vec4  getQuad(unsigned int i, unsigned int j, unsigned int k ) = 0 ; 
       virtual glm::ivec4 getQuadI(unsigned int i, unsigned int j, unsigned int k ) = 0 ; 

       virtual void save(const char* path) = 0;
       virtual void save(const char* dir, const char* name) = 0;
       virtual void save(const char* typ, const char* tag, const char* det) = 0;
       virtual void save(const char* tfmt, const char* targ, const char* tag, const char* det ) = 0;
 
    public:
       void Summary(const char* msg="NPYBase::Summary");
       std::string description(const char* msg="NPYBase::description");

   protected:
       void setNumItems(unsigned int ni);
   protected:
       unsigned int       m_dim ; 
       unsigned int       m_ni ; 
       unsigned int       m_nj ; 
       unsigned int       m_nk ; 
       unsigned int       m_nl ; 

       unsigned char      m_sizeoftype ; 
       Type_t             m_type ; 
       int                m_buffer_id ; 
       int                m_buffer_target ; 
       void*              m_aux ; 
       bool               m_verbose ; 
       bool               m_allow_prealloc ; 
 
   private:
       std::vector<int>   m_shape ; 
       std::string        m_metadata ; 
       bool               m_has_data ;
       bool               m_dynamic ;

};


inline NPYBase::NPYBase(std::vector<int>& shape, unsigned char sizeoftype, Type_t type, std::string& metadata, bool has_data) 
         :
         m_sizeoftype(sizeoftype),
         m_type(type),
         m_buffer_id(-1),
         m_buffer_target(-1),
         m_aux(NULL),
         m_verbose(false),
         m_allow_prealloc(false),
         m_shape(shape),
         m_metadata(metadata),
         m_has_data(has_data),
         m_dynamic(false)
{
   init();
} 




inline void NPYBase::setHasData(bool has_data)
{
    m_has_data = has_data ; 
}

inline bool NPYBase::hasData()
{
    return m_has_data ; 
}


// shape related

inline std::vector<int>& NPYBase::getShapeVector()
{
    return m_shape ; 
}



inline unsigned int NPYBase::getNumItems(int ifr, int ito)
{
    //  A) default ifr/ito  0/1 correponds to shape of 1st dimension
    //
    //  B) example ifr/ito  0/-1  gives number of items excluding last dimension 
    //               -->    0/2   --> shape(0)*shape(1)    for ndim 3 
    //
    //  C)       ifr/ito  0/3 for ndim 3   shape(0)*shape(1)*shape(2)
    //
    //  D)  ifr/ito 0/0     for any dimension
    //           -> 0/ndim     -> shape of all dimensions  
    //
    //
    int ndim = m_shape.size();
    if(ifr <  0) ifr += ndim ; 
    if(ito <= 0) ito += ndim ; 

    assert(ifr >= 0 && ifr < ndim);
    assert(ito >= 0 && ito <= ndim);

    unsigned int nit(1) ; 
    for(int i=ifr ; i < ito ; i++) nit *= getShape(i);
    return nit ;
}
inline unsigned int NPYBase::getNumElements()
{
    return getShape(m_shape.size()-1);
}


inline unsigned int NPYBase::getDimensions()
{
    return m_shape.size();
}
inline unsigned int NPYBase::getShape(unsigned int n)
{
    return n < m_shape.size() ? m_shape[n] : 0 ;
}



// OpenGL related

inline void NPYBase::setBufferId(int buffer_id)
{
    m_buffer_id = buffer_id  ;
}
inline int NPYBase::getBufferId()
{
    return m_buffer_id ;
}

inline void NPYBase::setBufferTarget(int buffer_target)
{
    m_buffer_target = buffer_target  ;
}
inline int NPYBase::getBufferTarget()
{
    return m_buffer_target ;
}





// used for CUDA OpenGL interop
inline void NPYBase::setAux(void* aux)
{
    m_aux = aux ; 
}
inline void* NPYBase::getAux()
{
    return m_aux ; 
}







inline void NPYBase::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}
inline void NPYBase::setAllowPrealloc(bool allow)
{
    m_allow_prealloc = allow ; 
}


inline unsigned int NPYBase::getValueIndex(unsigned int i, unsigned int j, unsigned int k, unsigned int l)
{
    //assert(m_dim == 3 ); 
    unsigned int nj = m_nj ;
    unsigned int nk = m_nk ;
    unsigned int nl = m_nl == 0 ? 1 : m_nl ;

    return  i*nj*nk*nl + j*nk*nl + k*nl + l ;
}

inline unsigned int NPYBase::getNumValues(unsigned int from_dim)
{
    unsigned int nvals = 1 ; 
    for(unsigned int i=from_dim ; i < m_shape.size() ; i++) nvals *= m_shape[i] ;
    return nvals ;  
}


// depending on sizeoftype

inline unsigned char NPYBase::getSizeOfType()
{
    return m_sizeoftype;
}
inline NPYBase::Type_t NPYBase::getType()
{
    return m_type;
}



inline unsigned int NPYBase::getNumBytes(unsigned int from_dim)
{
    return m_sizeoftype*getNumValues(from_dim);
}
inline unsigned int NPYBase::getByteIndex(unsigned int i, unsigned int j, unsigned int k, unsigned int l)
{
    return m_sizeoftype*getValueIndex(i,j,k,l);
}

inline void NPYBase::setDynamic(bool dynamic)
{
    m_dynamic = dynamic ; 
}
inline bool NPYBase::isDynamic()
{
    return m_dynamic ; 
}

