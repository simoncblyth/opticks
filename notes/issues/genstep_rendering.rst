genstep_rendering
====================


Overview
----------

1. tried adaption of oglrap/gl/p2l but get OPENGL INVALID


Review old oglrap rendering
--------------------------------

oglrap/Scene :  manages all the Shaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     592     m_axis_renderer = new Rdr(m_device, "axis", m_shader_dir, m_shader_incl_path );
     593 
     594     m_genstep_renderer = new Rdr(m_device, "p2l", m_shader_dir, m_shader_incl_path);
     595 
     596     m_nopstep_renderer = new Rdr(m_device, "nop", m_shader_dir, m_shader_incl_path);
     597     m_nopstep_renderer->setPrimitive(Rdr::LINE_STRIP);
     598 
     599     m_photon_renderer = new Rdr(m_device, "pos", m_shader_dir, m_shader_incl_path );
     600 
     601     m_source_renderer = new Rdr(m_device, "pos", m_shader_dir, m_shader_incl_path );
     602 
     603 
     604     //
     605     // RECORD RENDERING USES AN UNPARTIONED BUFFER OF ALL RECORDS
     606     // SO THE GEOMETRY SHADERS HAVE TO THROW INVALID STEPS AS DETERMINED BY
     607     // COMPARING THE TIMES OF THE STEP PAIRS  
     608     // THIS MEANS SINGLE VALID STEPS WOULD BE IGNORED..
     609     // THUS MUST SUPPLY LINE_STRIP SO GEOMETRY SHADER CAN GET TO SEE EACH VALID
     610     // VERTEX IN A PAIR
     611     //
     612     // OTHERWISE WILL MISS STEPS
     613     //
     614     //  see explanations in gl/altrec/geom.glsl
     615     //
     616     m_record_renderer = new Rdr(m_device, "rec", m_shader_dir, m_shader_incl_path );
     617     m_record_renderer->setPrimitive(Rdr::LINE_STRIP);
     618 
     619     m_altrecord_renderer = new Rdr(m_device, "altrec", m_shader_dir, m_shader_incl_path);
     620     m_altrecord_renderer->setPrimitive(Rdr::LINE_STRIP);
     621 
     622     m_devrecord_renderer = new Rdr(m_device, "devrec", m_shader_dir, m_shader_incl_path);
     623     m_devrecord_renderer->setPrimitive(Rdr::LINE_STRIP);
     624 



     890 void Scene::uploadEvent(OpticksEvent* evt)
     891 {
     892     if(!evt)
     893     {
     894        LOG(fatal) << "no evt " ;
     895        assert(evt);
     896     }
     897 
     898     // The Rdr call glBufferData using bytes and size from the associated NPY 
     899     // the bytes used is NULL when npy->hasData() == false
     900     // corresponding to device side only OpenGL allocation
     901 
     902     if(m_genstep_renderer)
     903         m_genstep_renderer->upload(evt->getGenstepAttr());
     904 
     905     if(m_nopstep_renderer)
     906          m_nopstep_renderer->upload(evt->getNopstepAttr());
     907 
     908     if(m_photon_renderer)
     909          m_photon_renderer->upload(evt->getPhotonAttr());
     910 
     911     if(m_source_renderer)
     912          m_source_renderer->upload(evt->getSourceAttr());
     913 
     914 
     915     uploadRecordAttr(evt->getRecordAttr());
     916 
     917     // Note that the above means that the same record renderers are 
     918     // uploading mutiple things from different NPY.
     919     // For this to work the counts must match.
     920     //
     921     // This is necessary for the photon records and the selection index.
     922     //
     923     // All renderers ready to roll so can live switch between them, 
     924     // data is not duplicated thanks to Device register of uploads
     925 }
     926 


Rdr::upload
~~~~~~~~~~~~~~


::

    193 void Rdr::upload(MultiViewNPY* mvn, bool debug)
    194 {
    195 
    196     if(!mvn) return ;
    197 
    198     m_uploads.push_back(mvn);
    199 
    200     // MultiViewNPY are constrained to all refer to the same underlying NPY 
    201     // so only do upload and m_buffer creation for the first 
    202 
    203     const char* tag = getShaderTag();
    204 
    205     if(debug)
    206     {
    207         LOG(info) << "tag [" << tag << "] mvn [" << mvn->getName() << "]" ;
    208         mvn->Summary("Rdr::upload mvn");
    209     }
    210 
    211     // need to compile and link shader for access to attribute locations
    212     if(m_first_upload)
    213     {
    214         prepare_vao(); // seems needed by oglrap-/tests/AxisTest 
    215         make_shader();
    216         glUseProgram(m_program);
    217         check_uniforms();
    218         log("Rdr::upload FIRST m_program:",m_program);
    219 
    220     }
    221 
    222     unsigned int count(0);
    223     NPYBase* npy(NULL);
    224 
    225     for(unsigned int i=0 ; i<mvn->getNumVecs() ; i++)
    226     {
    227         ViewNPY* vnpy = (*mvn)[i] ;
    228         if(npy == NULL)
    229         {
    230             count = vnpy->getCount();
    231             if(m_first_upload)
    232             {
    233                 if(debug)
    234                 LOG(info) << "Rdr::upload"
    235                           << " mvn " << mvn->getName()
    236                           << " (first)count " << count
    237                            ;
    238                 setCountDefault(count);
    239             }
    240             else
    241             {
    242                 bool count_match = count == getCountDefault() ;
    243                 if(!count_match)
    244                 {
    245                     LOG(fatal) << "Rdr::upload COUNT MISMATCH "
    246                                << " tag " << tag
    247                                << " mvn " << mvn->getName()
    248                                << " expected  " << getCountDefault()
    249                                << " found " << count
    250                                ;
    251                     dump_uploads_table();
    252                 }
    253                 assert(count_match && "all buffers fed to the Rdr pipeline must have the same counts");
    254             }
    255 
    256             npy = vnpy->getNPY();
    257             upload(npy, vnpy);      // duplicates are not re-uploaded
    258         }
    259         else
    260         {
    261             assert(npy == vnpy->getNPY());
    262             LOG(verbose) << "Rdr::upload counts, prior: " << count << " current: " << vnpy->getCount() ;
    263             assert(count == vnpy->getCount());
    264         }
    265         address(vnpy);
    266     }
    267 
    268 
    269     if(m_first_upload)
    270     {
    271         m_first_upload = false ;
    272     }
    273 }
                                                                                                     


Keeping multiviews of same buffer simple but avoiding duplicated uploads::

    314 void Rdr::upload(NPYBase* npy, ViewNPY* vnpy)
    315 {
    316     prepare_vao();
    317 
    318     MultiViewNPY* parent = vnpy->getParent();
    319     assert(parent);
    320 
    321     bool dynamic = npy->isDynamic();
    322 
    323     if(m_device->isUploaded(npy))
    324     {
    325         GLuint buffer_id = npy->getBufferId();
    326         log("Rdr::upload BindBuffer to preexisting buffer_id:",buffer_id)  ;
    327         assert(buffer_id > 0);
    328         glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    329     }
    330     else
    331     {
    332         void* data = npy->getBytes();
    333         unsigned int nbytes = npy->getNumBytes(0) ;
    334 
    335         char repdata[16] ;
    336         snprintf( repdata, 16, "%p", data );
    337 
    338         GLuint buffer_id ;
    339         glGenBuffers(1, &buffer_id);
    340         glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    341 
    342         LOG(debug)
    343             << std::setw(15) << parent->getName()
    344             << std::setw(5)  << vnpy->getName()
    345             << " cn " << std::setw(8) << vnpy->getCount()
    346             << " sh " << std::setw(20) << vnpy->getShapeString()
    347             << " id " << std::setw(5) << buffer_id
    348             << " dt " << std::setw(16) << repdata
    349             << " hd " << std::setw(5) << ( npy->hasData() ? "Y" : "N" )
    350             << " nb " << std::setw(10) << nbytes
    351             << " " << (dynamic ? "GL_DYNAMIC_DRAW" : "GL_STATIC_DRAW" )
    352             ;
    353 
    354         glBufferData(GL_ARRAY_BUFFER, nbytes, data, dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW  );
    355 
    356         npy->setBufferId(buffer_id);
    357         m_device->add(npy);         //  (void*)npy used by Device::isUploaded to prevent re-uploads  
    358 
    359 
    360         NGPU::GetInstance()->add( nbytes, vnpy->getName(), parent->getName(), "Rdr:upl" );
    361     }



Rdr::address
~~~~~~~~~~~~~

::

    404 void Rdr::address(ViewNPY* vnpy)
    405 {
    406     const char* name = vnpy->getName();
    407     GLint location = m_shader->attribute(name, false);
    408     if(location == -1)
    409     {
    410          LOG(debug)<<"Rdr::address failed to find active attribute for ViewNPY named " << name
    411                      << " in shader " << getShaderTag() ;
    412          return ;
    413     }
    414 
    415     GLenum type = GL_FLOAT  ;              //  of each component in the array
    416     switch(vnpy->getType())
    417     {
    418         case ViewNPY::BYTE:                         type = GL_BYTE           ; break ;
    419         case ViewNPY::UNSIGNED_BYTE:                type = GL_UNSIGNED_BYTE  ; break ;
    420         case ViewNPY::SHORT:                        type = GL_SHORT          ; break ;
    421         case ViewNPY::UNSIGNED_SHORT:               type = GL_UNSIGNED_SHORT ; break ;
    422         case ViewNPY::INT:                          type = GL_INT            ; break ;
    423         case ViewNPY::UNSIGNED_INT:                 type = GL_UNSIGNED_INT   ; break ;
    424         case ViewNPY::HALF_FLOAT:                   type = GL_HALF_FLOAT     ; break ;
    425         case ViewNPY::FLOAT:                        type = GL_FLOAT          ; break ;
    426         case ViewNPY::DOUBLE:                       type = GL_DOUBLE         ; break ;
    427         case ViewNPY::FIXED:                        type = GL_FIXED                        ; break ;
    428         case ViewNPY::INT_2_10_10_10_REV:           type = GL_INT_2_10_10_10_REV           ; break ;
    429         case ViewNPY::UNSIGNED_INT_2_10_10_10_REV:  type = GL_UNSIGNED_INT_2_10_10_10_REV  ; break ;
    430         //case ViewNPY::UNSIGNED_INT_10F_11F_11F_REV: type = GL_UNSIGNED_INT_10F_11F_11D_REV ; break ; 
    431         default: assert(0)                                                                 ; break ;
    432     }
    433 
    434 
    435     GLuint       index = location  ;            //  generic vertex attribute to be modified
    436     GLint         size = vnpy->getSize() ;      //  number of components per generic vertex attribute, must be 1,2,3,4
    437     GLboolean     norm = vnpy->getNorm() ;
    438     GLsizei       stride = vnpy->getStride();   // byte offset between consecutive generic vertex attributes, or 0 for tightly packed
    439 
    440     uintptr_t stride_ = stride ;
    441     uintptr_t offset_ = vnpy->getOffset() ;
    442 
    443     const GLvoid* offset = (const GLvoid*)offset_ ;
    444 
    445     // offset of the first component of the first generic vertex attribute 
    446     // in the array in the data store of the buffer currently bound to GL_ARRAY_BUFFER target
    447 
    448     LOG(verbose)
    449         << std::setw(10) << getShaderTag()
    450         << " name " << name
    451         << " type " << std::setw(20) << vnpy->getTypeName()
    452         << " index " << index
    453         << " norm " << norm
    454         << " size " << size
    455         << " stride " << stride
    456         << " offset_ " << offset_
    457         ;
    458 
    459     assert( offset_ < stride_ && "offset_ should always be less than the stride_, see ggv-/issues/gui_broken_photon_record_colors");
    460 
    461     if( vnpy->getIatt() )
    462     {
    463         glVertexAttribIPointer(index, size, type, stride, offset);
    464     }
    465     else
    466     {
    467         glVertexAttribPointer(index, size, type, norm, stride, offset);
    468     }
    469     glEnableVertexAttribArray(index);
    470 
    471 }






::

     471 MultiViewNPY* OpticksEvent::getNopstepAttr(){ return m_nopstep_attr ; }
     472 MultiViewNPY* OpticksEvent::getPhotonAttr(){ return m_photon_attr ; }
     473 MultiViewNPY* OpticksEvent::getSourceAttr(){ return m_source_attr ; }
     474 MultiViewNPY* OpticksEvent::getRecordAttr(){ return m_record_attr ; }
     475 MultiViewNPY* OpticksEvent::getDeluxeAttr(){ return m_deluxe_attr ; }
     476 MultiViewNPY* OpticksEvent::getPhoselAttr(){ return m_phosel_attr ; }
     477 MultiViewNPY* OpticksEvent::getRecselAttr(){ return m_recsel_attr ; }
     478 MultiViewNPY* OpticksEvent::getSequenceAttr(){ return m_sequence_attr ; }
     479 MultiViewNPY* OpticksEvent::getBoundaryAttr(){ return m_boundary_attr ; }
     480 MultiViewNPY* OpticksEvent::getSeedAttr(){   return m_seed_attr ; }
     481 MultiViewNPY* OpticksEvent::getHitAttr(){    return m_hit_attr ; }
     482 MultiViewNPY* OpticksEvent::getHiyAttr(){    return m_hiy_attr ; }



With Genstep rendering need to access the rpos and rdel (DeltaPosition)  from the 
same genstep buffer. But unclear how to do that. Record rendering only
needs one attribute from the buffer::


     56 inline void SGLFW_Record::init()
     57 {
     58     vao = new SGLFW_VAO("SGLFW_Record.vao") ;  // vao: establishes context for OpenGL attrib state and element array (not GL_ARRAY_BUFFER)
     59     vao->bind();
     60 
     61     buf = new SGLFW_Buffer("SGLFW_Record.buf", record->record->arr_bytes(), record->record->cvalues<float>(), GL_ARRAY_BUFFER,  GL_STATIC_DRAW );
     62     buf->bind();
     63     buf->upload();
     64 }
     65 
     66 
     67 /**
     68 SGLFW_Record::render
     69 ---------------------
     70 
     71 Called from renderloop.
     72 
     73 **/
     74 
     75 inline void SGLFW_Record::render(const SGLFW_Program* prog)
     76 {
     77     param_location = prog->getUniformLocation("Param");
     78     prog->use();
     79     vao->bind();
     80 
     81     buf->bind();
     82     prog->enableVertexAttribArray("rpos", SRecord::RPOS_SPEC );
     83 
     84     if(param_location > -1 ) prog->Uniform4fv(param_location, timeparam_ptr, false );
     85     prog->updateMVP();  // ?
     86 
     87     GLenum mode = prog->geometry_shader_text ? GL_LINE_STRIP : GL_POINTS ;
     88     glDrawArrays(mode, record->record_first,  record->record_count);
     89 }


     75 inline void SGLFW_Gen::render(const SGLFW_Program* prog)
     76 {
     77     param_location = prog->getUniformLocation("Param");
     78     prog->use();
     79     vao->bind();
     80 
     81     buf->bind();
     82     prog->enableVertexAttribArray("rpos", SGen::RPOS_SPEC );
     83 
     84     buf->bind();
     85     prog->enableVertexAttribArray("rdel", SGen::RDEL_SPEC );
     86 
     87 
     88     if(param_location > -1 ) prog->Uniform4fv(param_location, timeparam_ptr, false );
     89     prog->updateMVP();  // ?
     90 
     91     GLenum mode = prog->geometry_shader_text ? GL_LINE_STRIP : GL_POINTS ;
     92     glDrawArrays(mode, genstep->genstep_first,  genstep->genstep_count);
     93 }





SGLFW_Program::enableVertexAttribArray
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    384 inline void SGLFW_Program::enableVertexAttribArray( const char* name, const char* spec, bool dump ) const
    385 {
    386     if(dump) std::cout << "SGLFW_Program::enableVertexAttribArray name [" << name << "]" <<  std::endl ;
    387 
    388     SGLFW_Attrib att(name, spec);
    389 
    390     att.index = getAttribLocation( name );     SGLFW__check(__FILE__, __LINE__);
    391 
    392     if(dump) std::cout << "SGLFW_Program::enableVertexAttribArray att.desc [" << att.desc() << "]" <<  std::endl ;
    393 
    394     glEnableVertexAttribArray(att.index);      SGLFW__check(__FILE__, __LINE__);
    395 
    396     assert( att.integer_attribute == false );
    397 
    398     glVertexAttribPointer(att.index, att.size, att.type, att.normalized, att.stride, att.byte_offset_pointer );     SGLFW__check(__FILE__, __LINE__);
    399 }











