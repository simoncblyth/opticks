#pragma once

struct NP ; 
struct SMesh ; 
struct SGLFW_Program ; 


/**
SGLFW_Render
-------------

Try same prog for multiple mesh 

**/
struct SGLFW_Render
{
    const SMesh*   mesh ;  

    SGLFW_Buffer*  vtx ; 
    SGLFW_Buffer*  nrm ;
    SGLFW_Buffer*  ins ;

    SGLFW_VAO*     vao ;  
    SGLFW_Buffer*  idx ; 

    int           inst_num ; 
    const float*  inst_values ; 
    int           render_count ; 

    SGLFW_Render(const SMesh* mesh ) ; 
    void set_inst(const NP* _inst );
    void set_inst(int _inst_num, const float* _inst_values );
    bool has_inst() const ;

    std::string descInst() const ; 
    std::string desc() const ; 

    void render(const SGLFW_Program* prog); 
    void render_drawElements() const ; 
};

inline SGLFW_Render::SGLFW_Render(const SMesh* _mesh )
    :
    mesh(_mesh),
    vtx(nullptr),
    nrm(nullptr),
    ins(nullptr),
    vao(nullptr),
    idx(nullptr),
    inst_num(0),    
    inst_values(nullptr),    
    render_count(0)
{
    vtx = new SGLFW_Buffer( mesh->vtx->arr_bytes(), mesh->vtx->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    vtx->bind();
    vtx->upload(); 

    nrm = new SGLFW_Buffer( mesh->nrm->arr_bytes(), mesh->nrm->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    nrm->bind();
    nrm->upload(); 

    vao = new SGLFW_VAO ;  // vao: establishes context for OpenGL attrib state and element array (not vbuf,nbuf)
    vao->bind(); 

    idx = new SGLFW_Buffer( mesh->tri->arr_bytes(), mesh->tri->cvalues<int>()  , GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW ); 
    idx->bind();
    idx->upload(); 
}

inline void SGLFW_Render::set_inst(const NP* _inst )
{
    if(_inst == nullptr) return ; 
    assert( _inst->uifc == 'f' ); 
    assert( _inst->ebyte == 4 ); 
    assert( _inst->has_shape(-1,4,4)); 
    set_inst( _inst->num_items(), _inst->cvalues<float>() ); 
}

inline void SGLFW_Render::set_inst(int _inst_num, const float* _inst_values )
{
    inst_num = _inst_num ; 
    inst_values = _inst_values ; 

    int itemsize = 4*4*sizeof(float) ; 
    int num_bytes = inst_num*itemsize ;
    ins = new SGLFW_Buffer( num_bytes, inst_values, GL_ARRAY_BUFFER,  GL_STATIC_DRAW ); 
    ins->bind();
    ins->upload(); 
}

inline bool SGLFW_Render::has_inst() const
{
    return inst_num > 0 && inst_values != nullptr ; 
}


inline std::string SGLFW_Render::desc() const
{
    std::stringstream ss ; 
    ss << descInst() ; 
    std::string str = ss.str() ; 
    return str ; 
}
inline std::string SGLFW_Render::descInst() const
{
    int edge_items = 10 ; 
    std::stringstream ss ; 
    ss << "[SGLFW_Render::descInst inst_num " << inst_num << std::endl ; 
    ss << stra<float>::DescItems( inst_values, 16, inst_num, edge_items ); 
    std::string str = ss.str() ; 
    return str ; 
}

/**
SGLFW_Render::render
---------------------

NB: careful that the intended buffer is bound (making it the active GL_ARRAY_BUFFER)
when the vertex attrib is enabled.  Getting this wrong can for example easily cause
normals to appear in position slots causing perplexing renders. 

**/

inline void SGLFW_Render::render(const SGLFW_Program* prog)
{
   prog->use(); 
   vao->bind(); 

   vtx->bind();
   prog->enableVertexAttribArray( prog->vtx_attname, SMesh::VTX_SPEC ); 

   nrm->bind();
   prog->enableVertexAttribArray( prog->nrm_attname, SMesh::NRM_SPEC ); 

   if(ins)
   {
       ins->bind(); 
       prog->enableVertexAttribArray_OfTransforms( prog->ins_attname ) ; 
   }

   idx->bind();
   prog->updateMVP();

   render_drawElements(); 
   render_count += 1 ; 
}

inline void SGLFW_Render::render_drawElements() const 
{
    GLenum mode = GL_TRIANGLES ; 
  	GLsizei count = mesh->indices_num() ;  // number of elements to render (eg 3 for 1 triangle)
  	GLenum type = GL_UNSIGNED_INT ; 
  	const void * indices = (GLvoid*)(sizeof(GLuint) * mesh->indices_offset() ) ;
  	GLsizei instancecount = ins ? inst_num : 0 ; 

    if(instancecount > 0)
    {

#ifdef __APPLE__
        glDrawElementsInstanced(mode, count, type, indices, instancecount );
        if(render_count < 10 ) std::cout 
            << "SGLFW_Render::render_drawElements.glDrawElementsInstanced" 
            << " render_count " << render_count
            << " instancecount " << instancecount
            << std::endl
            ;

#else
        GLint basevertex = 0 ; 
        GLuint baseinstance = 0 ; 
        glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, basevertex, baseinstance ); 
        // SEGV on laptop, OK on worktation 
        // https://github.com/moderngl/moderngl/issues/346
        if(render_count < 10 ) std::cout 
            << "SGLFW_Render::render_drawElements.glDrawElementsInstancedBaseVertexBaseInstance" 
            << " render_count " << render_count
            << " instancecount " << instancecount
            << std::endl
            ;
#endif 

    }
    else
    {
        glDrawElements(mode, count, type, indices );
    }
}



