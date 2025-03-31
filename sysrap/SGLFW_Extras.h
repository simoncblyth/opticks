#pragma once
/**
SGLFW_Extras.h : Toggle, GLboolean, bool, GLenum, Attrib, Buffer, VAO 
======================================================================

SGLFW__check

SGLFW_Toggle
   viz switches

SGLFW_GLboolean
   string parse

SGLFW_bool
   string parse

SGLFW_GLenum
   string parse

SGLFW_Attrib
   parse attribute metadata strings such as "4,GL_FLOAT,GL_FALSE,64,0,false"

SGLFW_Buffer  
   minimal OpenGL buffer wrapper

SGLFW_VAO
   minimal Vertex Array wrapper


**/

inline void SGLFW__check(const char* path, int line) // static
{
    GLenum err = glGetError() ;   
    bool ok = err == GL_NO_ERROR ;
    const char* s = NULL ; 
    switch(err)
    {   
        case GL_INVALID_ENUM:      s = "GL_INVALID_ENUM" ; break ; 
        case GL_INVALID_VALUE:     s = "GL_INVALID_VALUE" ; break ; 
        case GL_INVALID_OPERATION: s = "GL_INVALID_OPERATION" ; break ; 
        case GL_STACK_OVERFLOW:    s = "GL_STACK_OVERFLOW" ; break ;   
        case GL_STACK_UNDERFLOW:   s = "GL_STACK_UNDERFLOW" ; break ;   
        case GL_OUT_OF_MEMORY:     s = "GL_OUT_OF_MEMORY" ; break ;   
#ifdef GL_VERSION_4_5
        case GL_CONTEXT_LOST:      s = "GL_CONTEXT_LOST" ; break ;
#endif
        case GL_INVALID_FRAMEBUFFER_OPERATION: s = "GL_INVALID_FRAMEBUFFER_OPERATION" ; break ;
    }   
    if(!ok) std::cout << "SGLFW__check OpenGL ERROR " << path << " : " << line << " : " << std::hex << err << std::dec << " : " << s << std::endl ; 
    assert( ok );  
}



/**
HMM: move to SGLM_Toggle as no use of GLFW 

**/

struct SGLFW_Toggle
{
    bool zoom ; 
    bool tmin ;
    bool tmax ;
    bool lrot ;
    bool cuda ;
    bool norm ;
    bool tran ;

    std::string desc() const ; 
};

inline std::string SGLFW_Toggle::desc() const
{  
    std::stringstream ss ;  
    ss << "SGLFW_Toggle::desc"
       << " zoom:" << ( zoom ? "Y" : "N" )
       << " tmin:" << ( tmin ? "Y" : "N" )
       << " tmax:" << ( tmax ? "Y" : "N" )
       << " lrot:" << ( lrot ? "Y" : "N" )
       << " cuda:" << ( cuda ? "Y" : "N" )
       << " norm:" << ( norm ? "Y" : "N" )
       << " tran:" << ( tran ? "Y" : "N" )
       ;
    std::string str = ss.str(); 
    return str ; 
}





struct SGLFW_GLboolean
{
    static constexpr const char* GL_FALSE_  = "GL_FALSE" ; 
    static constexpr const char* GL_TRUE_   = "GL_TRUE" ;
    static GLboolean Value(const char* name); 
    static const char* Name(GLboolean value); 
}; 
inline GLboolean SGLFW_GLboolean::Value(const char* name)
{
    GLboolean value = GL_FALSE ; 
    if( strcmp( name, GL_FALSE_) == 0 ) value = GL_FALSE ; 
    if( strcmp( name, GL_TRUE_)  == 0 ) value = GL_TRUE ; 
    return value ; 
}
inline const char* SGLFW_GLboolean::Name(GLboolean value)
{
    const char* s = nullptr ; 
    switch(value)
    {
       case GL_FALSE: s = GL_FALSE_ ; break ; 
       case GL_TRUE:  s = GL_TRUE_ ; break ; 
    }
    return s ; 
}


struct SGLFW_bool
{
    static constexpr const char* false_ = "false" ; 
    static constexpr const char* true_  = "true" ;
    static bool Value(const char* name); 
    static const char* Name(bool value); 
}; 
inline bool SGLFW_bool::Value(const char* name)
{
    bool value = false ; 
    if( strcmp( name, false_) == 0 ) value = false ; 
    if( strcmp( name, true_)  == 0 ) value = true ; 
    return value ; 
}
inline const char* SGLFW_bool::Name(bool value)
{
    return value ? true_ : false_ ; 
}



struct SGLFW_GLenum
{
    static constexpr const char* GL_BYTE_           = "GL_BYTE" ; 
    static constexpr const char* GL_UNSIGNED_BYTE_  = "GL_UNSIGNED_BYTE" ; 
    static constexpr const char* GL_SHORT_          = "GL_SHORT" ; 
    static constexpr const char* GL_UNSIGNED_SHORT_ = "GL_UNSIGNED_SHORT" ; 
    static constexpr const char* GL_INT_            = "GL_INT" ; 
    static constexpr const char* GL_UNSIGNED_INT_   = "GL_UNSIGNED_INT" ; 
    static constexpr const char* GL_HALF_FLOAT_     = "GL_HALF_FLOAT" ; 
    static constexpr const char* GL_FLOAT_          = "GL_FLOAT" ; 
    static constexpr const char* GL_DOUBLE_         = "GL_DOUBLE" ; 

    static const char* Name(GLenum type); 
    static GLenum      Type(const char* name); 
};

inline const char* SGLFW_GLenum::Name(GLenum type)
{
    const char* s = nullptr ; 
    switch(type)
    {
        case GL_BYTE:           s = GL_BYTE_           ; break ; 
        case GL_UNSIGNED_BYTE:  s = GL_UNSIGNED_BYTE_  ; break ; 
        case GL_SHORT:          s = GL_SHORT_          ; break ; 
        case GL_UNSIGNED_SHORT: s = GL_UNSIGNED_SHORT_ ; break ; 
        case GL_INT:            s = GL_INT_            ; break ; 
        case GL_UNSIGNED_INT:   s = GL_UNSIGNED_INT_   ; break ; 
        case GL_HALF_FLOAT:     s = GL_HALF_FLOAT_     ; break ;
        case GL_FLOAT:          s = GL_FLOAT_          ; break ;
        case GL_DOUBLE:         s = GL_DOUBLE_         ; break ;
        default:                s = nullptr            ; break ;
    }
    return s ; 
}

inline GLenum SGLFW_GLenum::Type(const char* name)
{
    GLenum type = 0 ; 
    if( strcmp( name, GL_BYTE_) == 0 )           type = GL_BYTE ; 
    if( strcmp( name, GL_UNSIGNED_BYTE_) == 0 )  type = GL_UNSIGNED_BYTE ; 
    if( strcmp( name, GL_SHORT_) == 0 )          type = GL_SHORT ; 
    if( strcmp( name, GL_UNSIGNED_SHORT_) == 0 ) type = GL_UNSIGNED_SHORT ; 
    if( strcmp( name, GL_INT_) == 0 )            type = GL_INT ; 
    if( strcmp( name, GL_UNSIGNED_INT_) == 0 )   type = GL_UNSIGNED_INT ; 
    if( strcmp( name, GL_HALF_FLOAT_) == 0 )     type = GL_HALF_FLOAT ; 
    if( strcmp( name, GL_FLOAT_) == 0 )          type = GL_FLOAT ; 
    if( strcmp( name, GL_DOUBLE_) == 0 )         type = GL_DOUBLE ; 
    return type ; 
}


/**
SGLFW_Attrib
-----------------

Parse a string of the below form into 6 fields::

    rpos:4,GL_FLOAT,GL_FALSE,64,0,false

**/
struct SGLFW_Attrib
{
    const char* name ; 
    const char* spec ; 
    std::vector<std::string> field ; 

    GLuint index ;                 // set externally to result of SGLFW_Program::getAttribLocation(name)

    GLint size ;                   // field 0 : number of components of the attribute (aka item), must be one of : 1,2,3,4 
    GLenum type ;                  // field 1 : normally GL_FLOAT 
    GLboolean normalized ;         // field 2 : normalized means in range 0->1
    GLsizei stride ;               // field 3 : in bytes eg for 4,4 float photon/record struct stride is 4*4*4=64
    size_t   byte_offset ;         // field 4 : allows access to different parts of array of structs 
    bool     integer_attribute ;   // field 5       

    void*    byte_offset_pointer ; // derived from byte_offset 


    SGLFW_Attrib( const char* name, const char* spec ); 
    std::string desc() const ;  
};


inline SGLFW_Attrib::SGLFW_Attrib(const char* name_, const char* spec_)
    :
    name(strdup(name_)),
    spec(strdup(spec_)),
    index(0),
    size(0),
    type(0),
    normalized(false),
    stride(0),
    byte_offset(0),
    integer_attribute(false),
    byte_offset_pointer(nullptr)
{
    char delim = ',' ; 
    std::stringstream ss; 
    ss.str(spec)  ;
    std::string s;
    while (std::getline(ss, s, delim)) field.push_back(s) ; 
    assert( field.size() == 6 ); 

    size =  std::atoi(field[0].c_str()) ;           assert( size == 1 || size == 2 || size == 3 || size == 4 ) ; 
    type =  SGLFW_GLenum::Type(field[1].c_str()) ;  assert( type > 0 );    
    normalized = SGLFW_GLboolean::Value(field[2].c_str()) ; 
    stride = std::atoi( field[3].c_str() );          assert( stride > 0 ); 
    byte_offset = std::atoi( field[4].c_str() );     assert( byte_offset >= 0 ); 
    integer_attribute = SGLFW_bool::Value(field[5].c_str()) ; 

    byte_offset_pointer = (void*)byte_offset ; 
}

inline std::string SGLFW_Attrib::desc() const 
{
    std::stringstream ss ; 
    ss << "SGLFW_Attrib::desc" << std::endl 
       << std::setw(20) << "name"  << " : " << name << std::endl 
       << std::setw(20) << "spec"  << " : " << spec << std::endl 
       << std::setw(20) << "index" << " : " << index << std::endl 
       << std::setw(20) << "size"  << " : " << size << std::endl 
       << std::setw(20) << "type"  << " : " << SGLFW_GLenum::Name(type) << std::endl
       << std::setw(20) << "normalized" << " : " << SGLFW_GLboolean::Name(normalized) << std::endl
       << std::setw(20) << "stride" << " : " << stride << std::endl
       << std::setw(20) << "byte_offset" << " : " << byte_offset << std::endl
       << std::setw(20) << "integer_attribute" << " : " << SGLFW_bool::Name(integer_attribute) << std::endl
       << std::setw(20) << "byte_offset_pointer" << " : " << byte_offset_pointer << std::endl
       ;

    for(unsigned i=0 ; i < field.size() ; i++ ) ss << std::setw(20) << i << " : " << field[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}


/**
SGLFW_Buffer : minimal OpenGL buffer wrapper
---------------------------------------------

Old Opticks oglrap handled multi-buffers using RBuf held by Renderer
See::

   Renderer::createVertexArray

**/

struct SGLFW_Buffer
{
    int num_bytes ; 
    const void* data ;
    GLenum target ;
    GLenum usage ;
    GLuint id ; 

    SGLFW_Buffer( int num_bytes, const void* data, GLenum target, GLenum usage  ); 

    void bind(); 
    void upload(); 
    void unbind(); 
};

inline SGLFW_Buffer::SGLFW_Buffer( int num_bytes_, const void* data_ , GLenum target_, GLenum usage_ )
    :
    num_bytes(num_bytes_),
    data(data_),
    target(target_),
    usage(usage_),
    id(0)
{
    glGenBuffers(1, &id );                         SGLFW__check(__FILE__, __LINE__);
}

inline void SGLFW_Buffer::bind()
{
    glBindBuffer(target, id);                      SGLFW__check(__FILE__, __LINE__);     
}

inline void SGLFW_Buffer::upload()
{
    glBufferData(target, num_bytes, data, usage ); SGLFW__check(__FILE__, __LINE__);
}

inline void SGLFW_Buffer::unbind()
{
    glBindBuffer(target, 0);                      SGLFW__check(__FILE__, __LINE__);     
}



/**
SGLFW_VAO : Minimal Vertex Array wrapper
--------------------------------------------
**/

struct SGLFW_VAO
{
    GLuint id ; 

    SGLFW_VAO(); 
    void init(); 
    void bind(); 
    void unbind(); 
};

inline SGLFW_VAO::SGLFW_VAO()
    :
    id(-1)
{
    init(); 
}

inline void SGLFW_VAO::init()
{
    printf("SGLFW_VAO::init\n"); 
    glGenVertexArrays (1, &id);  SGLFW__check(__FILE__, __LINE__);
}

inline void SGLFW_VAO::bind()
{
    glBindVertexArray(id);        SGLFW__check(__FILE__, __LINE__);
}
inline void SGLFW_VAO::unbind()
{
    glBindVertexArray(0);        SGLFW__check(__FILE__, __LINE__);
}



