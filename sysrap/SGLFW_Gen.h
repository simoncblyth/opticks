#pragma once
/**
SGLFW_Gen.h
===============


**/


struct SGLFW_Program ;
struct SGen ;

struct SGLFW_Gen
{
    static SGLFW_Gen* Create(const SGen* _genstep, const float* _timeparam_ptr  );

    const SGen*         genstep ;
    const float*        timeparam_ptr ;

    SGLFW_Buffer*       buf ;
    SGLFW_VAO*          vao ;
    GLint               param_location;

    SGLFW_Gen(const SGen* _genstep, const float* _timeparam_ptr  ) ;
    void init();

    void render(const SGLFW_Program* prog);
};


inline SGLFW_Gen* SGLFW_Gen::Create(const SGen* _genstep, const float* _timeparam_ptr )
{
    return _genstep ? new SGLFW_Gen(_genstep, _timeparam_ptr) : nullptr ;
}


inline SGLFW_Gen::SGLFW_Gen(const SGen* _genstep, const float* _timeparam_ptr )
    :
    genstep(_genstep),
    timeparam_ptr(_timeparam_ptr),
    buf(nullptr),
    vao(nullptr),
    param_location(0)
{
    init();
}

/**
SGLFW_Gen::init
----------------


**/


inline void SGLFW_Gen::init()
{
    vao = new SGLFW_VAO ;  // vao: establishes context for OpenGL attrib state and element array (not GL_ARRAY_BUFFER)
    vao->bind();

    buf = new SGLFW_Buffer( genstep->genstep->arr_bytes(), genstep->genstep->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW );
    buf->bind();
    buf->upload();
}


/**
SGLFW_Gen::render
---------------------

Called from renderloop.

**/

inline void SGLFW_Gen::render(const SGLFW_Program* prog)
{
    param_location = prog->getUniformLocation("Param");
    prog->use();
    vao->bind();

    buf->bind();
    prog->enableVertexAttribArray("rpos", SGen::RPOS_SPEC );
    prog->enableVertexAttribArray("rdel", SGen::RDEL_SPEC );


    if(param_location > -1 ) prog->Uniform4fv(param_location, timeparam_ptr, false );
    prog->updateMVP();  // ?

    GLenum mode = prog->geometry_shader_text ? GL_LINE_STRIP : GL_POINTS ;
    glDrawArrays(mode, genstep->genstep_first,  genstep->genstep_count);
}


