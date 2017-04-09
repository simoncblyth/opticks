#pragma once

#include <vector>
#include <string>
#include <cassert>

#include "NPY_FLAGS.hh"

// dont include NGLM.hpp here as it causes problem for thrustrap-
//#include "NGLM.hpp"
#include <glm/fwd.hpp>


class NParameters ; 
class NLookup ; 
class NPYSpec ; 
#include "NPY_API_EXPORT.hh"

class NPY_API NPYBase {
   public:
       typedef enum { FLOAT, SHORT, DOUBLE, INT, UINT, CHAR, UCHAR, ULONGLONG} Type_t ;
      // static const char* DEFAULT_DIR_TEMPLATE  ; 

       static const char* FLOAT_ ; 
       static const char* SHORT_ ; 
       static const char* DOUBLE_ ; 
       static const char* INT_ ; 
       static const char* UINT_ ; 
       static const char* CHAR_ ; 
       static const char* UCHAR_ ; 
       static const char* ULONGLONG_ ; 

       static const char* TypeName(Type_t type);

       static bool GLOBAL_VERBOSE ; 
       static int checkNumItems(NPYBase* data);
       static std::string path(const char* dir, const char* name);
       static std::string path(const char* dir, const char* reldir, const char* name);
       static std::string path(const char* det, const char* source, const char* tag, const char* tfmt);
   public:
        NPYBase(std::vector<int>& shape, unsigned char sizeoftype, Type_t type, std::string& metadata, bool has_data);
        virtual ~NPYBase();
        void setHasData(bool has_data=true);
        bool hasData();
        static void transfer(NPYBase* dst, NPYBase* src); 
   public:
       // shape related
       NPYSpec* getShapeSpec();
       NPYSpec* getItemSpec();
       std::vector<int>& getShapeVector();

       bool hasSameShape(NPYBase* other, unsigned fromdim=0);
       bool hasShape(int ni, int nj=0, int nk=0, int nl=0, int nm=0); // -1 for anything 
       bool hasItemShape(int nj, int nk=0, int nl=0, int nm=0);

       bool hasShapeSpec(NPYSpec* spec); 
       bool hasItemSpec(NPYSpec* spec); 

       std::string  getItemShape(unsigned int ifr=1);
       std::string  getDigestString();
       static std::string  getDigestString(void* bytes, unsigned int nbytes);
       bool isEqualTo(void* bytes, unsigned int nbytes);
       bool isEqualTo(NPYBase* other);
       unsigned int getNumQuads();  // product of all dimensions excluding the last, which must be 4 
       //unsigned int getLength();
       unsigned int getNumItems(int ifr=0, int ito=1);  // default ifr/ito=0/1 is size of 1st dimension
       unsigned int getNumElements();   // size of last dimension
       unsigned int getDimensions();
       std::string  getShapeString(unsigned int ifr=0);
       unsigned int getShape(unsigned int dim);
       unsigned int getValueIndex(unsigned i, unsigned j, unsigned k, unsigned l=0, unsigned m=0);
       unsigned int getNumValues(unsigned int from_dim=0);

       NParameters*  getParameters();
   public:
       // depending on sizeoftype
       Type_t        getType();
       bool          isIntegerType();
       bool          isFloatType();
       unsigned char getSizeOfType();
       unsigned int  getNumBytes(unsigned int from_dim=0);
       unsigned int  getByteIndex(unsigned i, unsigned j, unsigned k, unsigned l=0, unsigned m=0);
   public:
       void reshape(int ni, unsigned nj=0, unsigned nk=0, unsigned nl=0, unsigned nm=0);
   private:
       void init();
       void updateDimensions();
   public:
       // OpenGL related
       void         setBufferId(int buffer_id);
       int          getBufferId();  // either -1 if not uploaded, or the OpenGL buffer Id
       bool         isComputeBuffer();
       bool         isInteropBuffer();

       void         setBufferTarget(int buffer_target);
       int          getBufferTarget();  // -1 if unset

       void         setBufferControl(unsigned long long  buffer_control);
       unsigned long long getBufferControl();
       unsigned long long* getBufferControlPtr();

       void         setActionControl(unsigned long long  action_control);
       void         addActionControl(unsigned long long  action_control);
       unsigned long long  getActionControl();
       unsigned long long* getActionControlPtr();

       void     setLookup(NLookup* lookup);   // needed for legacy gensteps
       NLookup* getLookup();

       void         setBufferSpec(NPYSpec* spec);
       NPYSpec*     getBufferSpec();
       const char*  getBufferName();

       void         setAux(void* aux);
       void*        getAux();
       void         setDynamic(bool dynamic=true);
       bool         isDynamic();    // used by oglrap-/Rdr::upload

       bool isGenstepTranslated();
       void setGenstepTranslated(bool flag=true);

       unsigned getNumHit();
       void setNumHit(unsigned num_hit);
   private:
       void         setBufferName(const char* name);
   public:
       // NumPy static persistency path/dir providers moved to brap-/BOpticksEvent

       void setVerbose(bool verbose=true);
       void setAllowPrealloc(bool allow=true); 
       static void setGlobalVerbose(bool verbose=true);

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
       //virtual void save(const char* typ, const char* tag, const char* det) = 0;
       virtual void save(const char* tfmt, const char* targ, const char* tag, const char* det ) = 0;
 
    public:
       void Summary(const char* msg="NPYBase::Summary");
       std::string description(const char* msg="NPYBase::description");

   //protected:
    public:
       void setNumItems(unsigned int ni);

   protected:
       unsigned int       m_dim ; 

       unsigned int       m_ni ; 
       unsigned int       m_nj ; 
       unsigned int       m_nk ; 
       unsigned int       m_nl ; 
       unsigned int       m_nm ;
 
       NPYSpec*           m_shape_spec ; 
       NPYSpec*           m_item_spec ; 
       NPYSpec*           m_buffer_spec ; 

       unsigned char      m_sizeoftype ; 
       Type_t             m_type ; 
       int                m_buffer_id ; 
       int                m_buffer_target ; 
       unsigned long long m_buffer_control ; 
       const char*        m_buffer_name ; 
       unsigned long long m_action_control ; 
       void*              m_aux ; 
       bool               m_verbose ; 
       bool               m_allow_prealloc ; 
 
   private:
       std::vector<int>   m_shape ; 
       std::string        m_metadata ; 
       bool               m_has_data ;
       bool               m_dynamic ;
       NLookup*           m_lookup ;   // only needed for legacy gensteps 
       NParameters*        m_parameters ;  // for keeping notes, especially for gensteps

};


