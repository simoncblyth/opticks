#pragma once
/**
SGLFW_Program.h : compile and link OpenGL pipeline using shader sources loaded from directory
===============================================================================================


**/
#include "ssys.h"


struct SGLFW_Program
{
    static constexpr const char* MVP_KEYS = "ModelViewProjection,MVP" ;

    int level ;
    const char* dir ;
    const char* vtx_attname ;
    const char* nrm_attname ;
    const char* ins_attname ;
    const char* mvp_uniname ;

    const char* vertex_shader_text ;
    const char* geometry_shader_text ;
    const char* fragment_shader_text ;

    GLuint program ;
    GLint  mvp_location ;
    const float* mvp ;
    bool dump ;

    SGLFW_Program(
        const char* _dir,
        const char* _vtx_attname,
        const char* _nrm_attname,
        const char* _ins_attname,
        const char* _mvp_uniname,
        const float* _mvp
        );
    void init();

    void createFromDir(const char* _dir);
    void createFromText(const char* vertex_shader_text, const char* geometry_shader_text, const char* fragment_shader_text );
    void use() const ;

    GLint getUniformLocation(const char* name) const ;
    GLint getAttribLocation(const char* name) const ;

    GLint findUniformLocation(const char* keys, char delim ) const ;
    void locateMVP(const char* key, const float* mvp );
    void updateMVP() const ;  // called from renderloop_head

    static void UniformMatrix4fv( GLint loc, const float* vv, bool dump );
    static void Uniform4fv(       GLint loc, const float* vv, bool dump );

    void enableVertexAttribArray( const char* name, const char* spec, bool dump=false ) const ;
    void enableVertexAttribArray_OfTransforms( const char* name ) const ;

    static void Print_shader_info_log(unsigned id);

    template<typename T>
    static std::string Desc(const T* tt, int num);

};

inline SGLFW_Program::SGLFW_Program(
    const char* _dir,
    const char* _vtx_attname,
    const char* _nrm_attname,
    const char* _ins_attname,
    const char* _mvp_uniname,
    const float* _mvp
    )
    :
    level(ssys::getenvint("SGLFW_Program_LEVEL", 0)),
    dir( _dir ? strdup(_dir) : nullptr ),
    vtx_attname( _vtx_attname ? strdup(_vtx_attname) : nullptr ),
    nrm_attname( _nrm_attname ? strdup(_nrm_attname) : nullptr ),
    ins_attname( _ins_attname ? strdup(_ins_attname) : nullptr ),
    mvp_uniname( _mvp_uniname ? strdup(_mvp_uniname) : nullptr ),
    vertex_shader_text(nullptr),
    geometry_shader_text(nullptr),
    fragment_shader_text(nullptr),
    program(0),
    mvp_location(-1),
    mvp(_mvp),
    dump(false)
{
    init();
}


inline void SGLFW_Program::init()
{
    if(dir) createFromDir(dir) ;
    use();
    if(mvp_uniname)
    {
        mvp_location = getUniformLocation(mvp_uniname);
        assert( mvp_location > -1 );
    }
}


/**
SGLFW_Program::locateMVP
-------------------------

Does not update GPU side, invoke SGLFW_Program::locateMVP
prior to the renderloop after shader program is
setup and the GLM maths has been instanciated
hence giving the pointer to the world2clip matrix
address.

**/

inline void SGLFW_Program::locateMVP(const char* key, const float* mvp_ )
{
    if(level > 0) std::cout << "SGLFW_Program::locateMVP backwards compat" << std::endl ;
    mvp_location = getUniformLocation(key);
    assert( mvp_location > -1 );
    mvp = mvp_ ;
}


/**
SGLFW_Program::updateMVP
-------------------------

When mvp_location is > -1 this is called from
the end of renderloop_head so any matrix updates
need to be done before then.

HMM: could just pass in the pointer ?

**/

inline void SGLFW_Program::updateMVP() const
{
    if( mvp_location <= -1 ) return ;
    assert( mvp != nullptr );
    UniformMatrix4fv(mvp_location, mvp, dump );
}



/**
SGLFW_Program::createFromDir
-------------------------------

Loads {vert/geom/frag}.glsl shader source files from provided directory
and invokes createFromText

**/

inline void SGLFW_Program::createFromDir(const char* _dir)
{
    const char* dir = U::Resolve(_dir);

    vertex_shader_text = U::ReadString(dir, "vert.glsl");
    geometry_shader_text = U::ReadString(dir, "geom.glsl");
    fragment_shader_text = U::ReadString(dir, "frag.glsl");

    if(level > 0) std::cout
        << "SGLFW_Program::createFromDir"
        << " _dir " << ( _dir ? _dir : "-" )
        << " dir "  << (  dir ?  dir : "-" )
        << " vertex_shader_text " << ( vertex_shader_text ? "YES" : "NO" )
        << " geometry_shader_text " << ( geometry_shader_text ? "YES" : "NO" )
        << " fragment_shader_text " << ( fragment_shader_text ? "YES" : "NO" )
        << std::endl
        ;

    createFromText( vertex_shader_text, geometry_shader_text, fragment_shader_text );
}


/**
SGLFW_Program::createFromText
------------------------------

Compiles and links shader strings into a program referred from integer *program*

On macOS with the below get "runtime error, unsupported version"::

    #version 460 core

On macOS with the below::

    #version 410 core

note that a trailing semicolon after the main curly brackets gives a syntax error,
that did not see on Linux with "#version 460 core"

**/

inline void SGLFW_Program::createFromText(const char* vertex_shader_text, const char* geometry_shader_text, const char* fragment_shader_text )
{
    if(level > 0) std::cout << "[SGLFW_Program::createFromText level " << level << std::endl ;
    if(level > 1) std::cout << " vertex_shader_text " << std::endl << vertex_shader_text << std::endl ;
    if(level > 1) std::cout << " geometry_shader_text " << std::endl << ( geometry_shader_text ? geometry_shader_text : "-" )  << std::endl ;
    if(level > 1) std::cout << " fragment_shader_text " << std::endl << fragment_shader_text << std::endl ;

    int params = -1;
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);                    SGLFW__check(__FILE__, __LINE__);
    glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);                SGLFW__check(__FILE__, __LINE__);
    glCompileShader(vertex_shader);                                             SGLFW__check(__FILE__, __LINE__);
    glGetShaderiv (vertex_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) Print_shader_info_log(vertex_shader) ;

    GLuint geometry_shader = 0 ;
    if( geometry_shader_text )
    {
        geometry_shader = glCreateShader(GL_GEOMETRY_SHADER);                       SGLFW__check(__FILE__, __LINE__);
        glShaderSource(geometry_shader, 1, &geometry_shader_text, NULL);            SGLFW__check(__FILE__, __LINE__);
        glCompileShader(geometry_shader);                                           SGLFW__check(__FILE__, __LINE__);
        glGetShaderiv (geometry_shader, GL_COMPILE_STATUS, &params);
        if (GL_TRUE != params) Print_shader_info_log(geometry_shader) ;
    }

    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);                SGLFW__check(__FILE__, __LINE__);
    glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);            SGLFW__check(__FILE__, __LINE__);
    glCompileShader(fragment_shader);                                           SGLFW__check(__FILE__, __LINE__);
    glGetShaderiv (fragment_shader, GL_COMPILE_STATUS, &params);
    if (GL_TRUE != params) Print_shader_info_log(fragment_shader) ;

    program = glCreateProgram();               SGLFW__check(__FILE__, __LINE__);
    glAttachShader(program, vertex_shader);    SGLFW__check(__FILE__, __LINE__);
    if( geometry_shader > 0 )
    {
        glAttachShader(program, geometry_shader); SGLFW__check(__FILE__, __LINE__);
    }
    glAttachShader(program, fragment_shader);  SGLFW__check(__FILE__, __LINE__);
    glLinkProgram(program);                    SGLFW__check(__FILE__, __LINE__);

    if(level > 0) std::cout << "]SGLFW_Program::createFromText level " << level << std::endl ;
}

inline void SGLFW_Program::use() const
{
    glUseProgram(program);
}




inline GLint SGLFW_Program::getUniformLocation(const char* name) const
{
    GLint loc = glGetUniformLocation(program, name);   SGLFW__check(__FILE__, __LINE__);
    return loc ;
}

inline GLint SGLFW_Program::getAttribLocation(const char* name) const
{
    GLint loc = glGetAttribLocation(program, name);   SGLFW__check(__FILE__, __LINE__);
    return loc ;
}




/**
SGLFW_Program::findUniformLocation
-----------------------------------

Returns the location int for the first uniform key found in the
shader program

**/

inline GLint SGLFW_Program::findUniformLocation(const char* keys, char delim ) const
{
    std::vector<std::string> kk ;

    std::stringstream ss;
    ss.str(keys)  ;
    std::string key;
    while (std::getline(ss, key, delim)) kk.push_back(key) ;

    GLint loc = -1 ;

    int num_key = kk.size();
    for(int i=0 ; i < num_key ; i++)
    {
        const char* k = kk[i].c_str();
        loc = getUniformLocation(k);
        if(loc > -1) break ;
    }
    return loc ;
}



template<typename T>
inline std::string SGLFW_Program::Desc(const T* tt, int num) // static
{
    std::stringstream ss ;
    for(int i=0 ; i < num ; i++)
        ss
            << ( i % 4 == 0 && num > 4 ? ".\n" : "" )
            << " " << std::fixed << std::setw(10) << std::setprecision(4) << tt[i]
            << ( i == num-1 && num > 4 ? ".\n" : "" )
            ;

    std::string s = ss.str();
    return s ;
}

inline void SGLFW_Program::UniformMatrix4fv( GLint loc, const float* vv, bool dump ) // static
{
    if(dump) std::cout
        << "SGLFW_Program::UniformMatrix4fv"
        << " loc " << loc
        << std::endl
        << Desc(vv, 16)
        << std::endl
        ;

    assert( loc > -1 );
    glUniformMatrix4fv(loc, 1, GL_FALSE, (const GLfloat*)vv );
}

inline void SGLFW_Program::Uniform4fv( GLint loc, const float* vv, bool dump ) // static
{
    if(dump) std::cout
        << "SGLFW_Program::Uniform4fv"
        << " loc " << loc
        << std::endl
        << Desc(vv, 4)
        << std::endl
        ;

    assert( loc > -1 );
    glUniform4fv(loc, 1, (const GLfloat*)vv );
}





/**
SGLFW_Program::enableVertexAttribArray
-------------------------------------------

Array attribute : connecting values from the array with attribute symbol in the shader program

Example rpos spec "4,GL_FLOAT,GL_FALSE,64,0,false"

NB when handling multiple buffers note that glVertexAttribPointer
binds to the buffer object bound to GL_ARRAY_BUFFER when called.
So that means have to repeatedly call this again after switching
buffers ?

* https://stackoverflow.com/questions/14249634/opengl-vaos-and-multiple-buffers
* https://antongerdelan.net/opengl/vertexbuffers.html

* NOTICE THAT index 0 IS NOT "NULL" for VertexAttribArray

**/

inline void SGLFW_Program::enableVertexAttribArray( const char* name, const char* spec, bool dump ) const
{
    if(dump) std::cout << "SGLFW_Program::enableVertexAttribArray name [" << name << "]" <<  std::endl ;

    SGLFW_Attrib att(name, spec);

    att.index = getAttribLocation( name );     SGLFW__check(__FILE__, __LINE__);

    if(dump) std::cout << "SGLFW_Program::enableVertexAttribArray att.desc [" << att.desc() << "]" <<  std::endl ;

    glEnableVertexAttribArray(att.index);      SGLFW__check(__FILE__, __LINE__);

    assert( att.integer_attribute == false );

    glVertexAttribPointer(att.index, att.size, att.type, att.normalized, att.stride, att.byte_offset_pointer );     SGLFW__check(__FILE__, __LINE__);
}


inline void SGLFW_Program::enableVertexAttribArray_OfTransforms( const char* name ) const
{
    assert( name );

    SGLFW_Attrib att(name, SMesh::MATROW_SPEC );

    att.index = getAttribLocation( name );     SGLFW__check(__FILE__, __LINE__);

    size_t qsize = att.stride/4 ;
    GLuint divisor = 1 ;
    // number of instances between updates of attribute , >1 will land that many instances on top of each other

    const void* offset0 = (void*)(qsize*0) ;
    const void* offset1 = (void*)(qsize*1) ;
    const void* offset2 = (void*)(qsize*2) ;
    const void* offset3 = (void*)(qsize*3) ;

    glEnableVertexAttribArray(att.index+0);                                                       SGLFW__check(__FILE__, __LINE__);
    glVertexAttribPointer(att.index+0, att.size, att.type, att.normalized, att.stride, offset0 ); SGLFW__check(__FILE__, __LINE__);
    glVertexAttribDivisor(att.index+0, divisor);                                                  SGLFW__check(__FILE__, __LINE__);

    glEnableVertexAttribArray(att.index+1);                                                       SGLFW__check(__FILE__, __LINE__);
    glVertexAttribPointer(att.index+1, att.size, att.type, att.normalized, att.stride, offset1 ); SGLFW__check(__FILE__, __LINE__);
    glVertexAttribDivisor(att.index+1, divisor);                                                  SGLFW__check(__FILE__, __LINE__);

    glEnableVertexAttribArray(att.index+2);                                                       SGLFW__check(__FILE__, __LINE__);
    glVertexAttribPointer(att.index+2, att.size, att.type, att.normalized, att.stride, offset2 ); SGLFW__check(__FILE__, __LINE__);
    glVertexAttribDivisor(att.index+2, divisor);                                                  SGLFW__check(__FILE__, __LINE__);

    glEnableVertexAttribArray(att.index+3);                                                       SGLFW__check(__FILE__, __LINE__);
    glVertexAttribPointer(att.index+3, att.size, att.type, att.normalized, att.stride, offset3 ); SGLFW__check(__FILE__, __LINE__);
    glVertexAttribDivisor(att.index+3, divisor);                                                  SGLFW__check(__FILE__, __LINE__);
}

inline void SGLFW_Program::Print_shader_info_log(unsigned id)  // static
{
    int max_length = 2048;
    int actual_length = 0;
    char log[2048];

    glGetShaderInfoLog(id, max_length, &actual_length, log);
    SGLFW__check(__FILE__, __LINE__ );

    printf("SGLFW_Program::Print_shader_info_log GL index %u:\n%s\n", id, log);
    assert(0);
}


