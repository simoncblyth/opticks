#pragma once

struct SGLFW_Program
{
    static constexpr const char* MVP_KEYS = "ModelViewProjection,MVP" ;  

    const char* vertex_shader_text ;
    const char* geometry_shader_text ; 
    const char* fragment_shader_text ;
    GLuint program ; 
    GLint  mvp_location ; 
    const float* mvp ; 
    bool dump ; 

    SGLFW_Program( const char* _dir ); 

    void createFromDir(const char* _dir); 
    void createFromText(const char* vertex_shader_text, const char* geometry_shader_text, const char* fragment_shader_text ); 
    void use(); 

    GLint getUniformLocation(const char* name) const ; 
    GLint getAttribLocation(const char* name) const ; 

    GLint findUniformLocation(const char* keys, char delim ) const ; 
    void locateMVP(const char* key, const float* mvp ); 
    void updateMVP();  // called from renderloop_head

    void UniformMatrix4fv( GLint loc, const float* vv ); 
    void Uniform4fv(       GLint loc, const float* vv ); 

    void enableAttrib( const char* name, const char* spec, bool dump=false ); 

    static void Print_shader_info_log(unsigned id); 

    template<typename T>
    static std::string Desc(const T* tt, int num); 

};

inline SGLFW_Program::SGLFW_Program( const char* _dir )
    :
    vertex_shader_text(nullptr),
    geometry_shader_text(nullptr),
    fragment_shader_text(nullptr),
    program(0),
    mvp_location(-1),
    mvp(nullptr),
    dump(true)
{
    createFromDir(_dir) ; 
}

inline void SGLFW_Program::createFromDir(const char* _dir)
{
    const char* dir = U::Resolve(_dir); 

    vertex_shader_text = U::ReadString(dir, "vert.glsl"); 
    geometry_shader_text = U::ReadString(dir, "geom.glsl"); 
    fragment_shader_text = U::ReadString(dir, "frag.glsl"); 

    std::cout 
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
    std::cout << "[SGLFW_Program::createFromText" << std::endl ; 
    //std::cout << " vertex_shader_text " << std::endl << vertex_shader_text << std::endl ;
    //std::cout << " geometry_shader_text " << std::endl << ( geometry_shader_text ? geometry_shader_text : "-" )  << std::endl ;
    //std::cout << " fragment_shader_text " << std::endl << fragment_shader_text << std::endl ;

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

    std::cout << "]SGLFW_Program::createFromText" << std::endl ; 
}

inline void SGLFW_Program::use()
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

**/

inline void SGLFW_Program::updateMVP()
{
    if( mvp_location <= -1 ) return ; 
    assert( mvp != nullptr ); 
    UniformMatrix4fv(mvp_location, mvp); 
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

inline void SGLFW_Program::UniformMatrix4fv( GLint loc, const float* vv )
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

inline void SGLFW_Program::Uniform4fv( GLint loc, const float* vv )
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
SGLFW_Program::enableAttrib
-----------------------------

Array attribute : connecting values from the array with attribute symbol in the shader program 

Example rpos spec "4,GL_FLOAT,GL_FALSE,64,0,false"


NB when handling multiple buffers note that glVertexAttribPointer
binds to the buffer object bound to GL_ARRAY_BUFFER when called. 
So that means have to repeatedly call this again after switching
buffers ? 

* https://stackoverflow.com/questions/14249634/opengl-vaos-and-multiple-buffers 
* https://antongerdelan.net/opengl/vertexbuffers.html

**/

inline void SGLFW_Program::enableAttrib( const char* name, const char* spec, bool dump )
{
    SGLFW_Attrib att(name, spec); 

    att.index = getAttribLocation( name );     SGLFW__check(__FILE__, __LINE__);

    if(dump) std::cout << "SGLFW_Program::enableArrayAttribute att.desc [" << att.desc() << "]" <<  std::endl ; 

    glEnableVertexAttribArray(att.index);      SGLFW__check(__FILE__, __LINE__);

    assert( att.integer_attribute == false ); 

    glVertexAttribPointer(att.index, att.size, att.type, att.normalized, att.stride, att.byte_offset_pointer );     SGLFW__check(__FILE__, __LINE__);
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


