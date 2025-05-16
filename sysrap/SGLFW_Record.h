#pragma once
/**
SGLFW_Record.h
===============


**/


struct NP ;
struct SGLFW_Program ;
struct SRecordInfo ;
struct sfr;
struct SGLM ;

struct SGLFW_Record
{
    static SGLFW_Record* Create(SGLM& gm, const SRecordInfo* _sr);

    SGLM& gm ;
    bool  dump ;

    const SRecordInfo*  sr ;
    SGLFW_Buffer*       rpos ;
    SGLFW_VAO*          vao ;
    GLint          param_location;

    SGLFW_Record(SGLM& gm, const SRecordInfo* _sr) ;
    void init();


    void render(const SGLFW_Program* prog);
};


inline SGLFW_Record* SGLFW_Record::Create(SGLM& gm, const SRecordInfo* _sr)
{
    return _sr ? new SGLFW_Record(gm, _sr) : nullptr ;
}


inline SGLFW_Record::SGLFW_Record(SGLM& _gm, const SRecordInfo* _sr )
    :
    gm(_gm),
    dump(false),
    sr(_sr),
    rpos(nullptr),
    vao(nullptr),
    param_location(0)
{
    init();
}

/**
SGLFW_Record::init
----------------


**/


inline void SGLFW_Record::init()
{
    if(sr) gm.init_time( sr->get_t0(), sr->get_t1(), sr->get_num_time() );

    vao = new SGLFW_VAO ;  // vao: establishes context for OpenGL attrib state and element array (not GL_ARRAY_BUFFER)
    vao->bind();

    rpos = new SGLFW_Buffer( sr->record->arr_bytes(), sr->record->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW );
    rpos->bind();
    rpos->upload();
}


/**
SGLFW_Record::render
---------------------

Called from renderloop. At each call the simulation time
is bumped until the time exceeds t1 at which point it is
returned to t0.

**/

inline void SGLFW_Record::render(const SGLFW_Program* prog)
{
    param_location = prog->getUniformLocation("Param");
    prog->use();
    vao->bind();

    rpos->bind();
    prog->enableVertexAttribArray("rpos", SRecordInfo::RPOS_SPEC );

    gm.bump_time();

    if(param_location > -1 ) prog->Uniform4fv(param_location, glm::value_ptr(gm.timeparam), false );
    prog->updateMVP();  // ?

    GLenum mode = prog->geometry_shader_text ? GL_LINE_STRIP : GL_POINTS ;
    glDrawArrays(mode, sr->record_first,  sr->record_count);
}


