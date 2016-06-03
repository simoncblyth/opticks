Borked Photon Record Colors
============================

Following OpticksEvent simplifications and associated changes to ViewNPY the 
photon record colors selectable by M key are incorrect, exceping the all white mode.

All flavors of genstep show the issue::

    op
    op -c
    op -s


where do the colors come from
--------------------------------

oglrap-/gl/fcolor.h is incl into the record shaders, all modes other than zero for all white are afflicted::

     01 //
      2 //  ColorDomain
      3 //         x : 0.
      4 //         y : total number of colors in color buffer (*)
      5 //         z : number of psychedelic colors 
      6 //         w : 0.
      7 // 
      8 // (*) total number of colors is required to convert a buffer index 
      9 //     into float 0->1 for texture access
     10 //
     11 // offsets into the color buffer are obtained from dynamic define
     12 // in order to match those of  GColors::setupCompositeColorBuffer
     13 //
     14 
     15     switch(ColorParam.x)
     16     {
     17        case 0:
     18               fcolor = vec4(1.0,1.0,1.0,1.0) ; break;
     19        case 1:
     20               fcolor = texture(Colors, (float(flq[0].x) + MATERIAL_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break;
     21        case 2:
     22               fcolor = texture(Colors, (float(flq[1].x) + MATERIAL_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break;
     23        case 3:
     24               fcolor = texture(Colors, (float(flq[0].w) + FLAG_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break;
     25        case 4:
     26               fcolor = texture(Colors, (float(flq[1].w) + FLAG_COLOR_OFFSET - 1.0 + 0.5)/ColorDomain.y ) ; break;
     27        case 5:
     28               fcolor = vec4(vec3(polarization[0]), 1.0) ; break;
     29        case 6:
     30               fcolor = vec4(vec3(polarization[1]), 1.0) ; break;
     31     }
     32 


Polarization modes are effected : suggests problem is with 
attribute access rather than the Colors texture.

* **Offsets should always be less than stride**

Suspect that the offsets other than 0 for rpos incorrect. 
Before fix::

    [2016-Jun-03 10:34:16.888453]:info: Rdr::address (glVertexAttribPointer)        rec name rpos type 2 index 0 norm  size 4 stride 16 offset 0
    [2016-Jun-03 10:34:16.888600]:info: Rdr::address (glVertexAttribPointer)        rec name rpol type 1 index 1 norm  size 4 stride 16 offset 16
    [2016-Jun-03 10:34:16.888761]:info: Rdr::address (glVertexAttribPointer)        rec name rflg type 3 index 2 norm  size 2 stride 16 offset 32
    [2016-Jun-03 10:34:16.888910]:info: Rdr::address (glVertexAttribPointer)        rec name rflq type 1 index 4 norm  size 4 stride 16 offset 32

    [2016-Jun-03 10:34:16.900436]:info: Rdr::address (glVertexAttribPointer)     altrec name rpos type 2 index 0 norm  size 4 stride 16 offset 0
    [2016-Jun-03 10:34:16.900578]:info: Rdr::address (glVertexAttribPointer)     altrec name rpol type 1 index 1 norm  size 4 stride 16 offset 16
    [2016-Jun-03 10:34:16.900693]:info: Rdr::address (glVertexAttribPointer)     altrec name rflg type 3 index 2 norm  size 2 stride 16 offset 32
    [2016-Jun-03 10:34:16.900805]:info: Rdr::address (glVertexAttribPointer)     altrec name rflq type 1 index 4 norm  size 4 stride 16 offset 32

    [2016-Jun-03 10:34:16.913666]:info: Rdr::address (glVertexAttribPointer)     devrec name rpos type 2 index 0 norm  size 4 stride 16 offset 0
    [2016-Jun-03 10:34:16.913822]:info: Rdr::address (glVertexAttribPointer)     devrec name rpol type 1 index 1 norm  size 4 stride 16 offset 16
    [2016-Jun-03 10:34:16.913942]:info: Rdr::address (glVertexAttribPointer)     devrec name rflg type 3 index 2 norm  size 2 stride 16 offset 32
    [2016-Jun-03 10:34:16.914058]:info: Rdr::address (glVertexAttribPointer)     devrec name rflq type 1 index 4 norm  size 4 stride 16 offset 32


After fix::

    [2016-Jun-03 11:09:11.866200]:info: Rdr::address (glVertexAttribPointer)        rec name rpos type                SHORT index 0 norm  size 4 stride 16 offset 0
    [2016-Jun-03 11:09:11.866378]:info: Rdr::address (glVertexAttribPointer)        rec name rpol type        UNSIGNED_BYTE index 1 norm  size 4 stride 16 offset 8
    [2016-Jun-03 11:09:11.866539]:info: Rdr::address (glVertexAttribPointer)        rec name rflg type       UNSIGNED_SHORT index 2 norm  size 2 stride 16 offset 12
    [2016-Jun-03 11:09:11.866699]:info: Rdr::address (glVertexAttribPointer)        rec name rflq type        UNSIGNED_BYTE index 4 norm  size 4 stride 16 offset 12
    [2016-Jun-03 11:09:11.878231]:info: Rdr::address (glVertexAttribPointer)     altrec name rpos type                SHORT index 0 norm  size 4 stride 16 offset 0
    [2016-Jun-03 11:09:11.878409]:info: Rdr::address (glVertexAttribPointer)     altrec name rpol type        UNSIGNED_BYTE index 1 norm  size 4 stride 16 offset 8
    [2016-Jun-03 11:09:11.878542]:info: Rdr::address (glVertexAttribPointer)     altrec name rflg type       UNSIGNED_SHORT index 2 norm  size 2 stride 16 offset 12
    [2016-Jun-03 11:09:11.878684]:info: Rdr::address (glVertexAttribPointer)     altrec name rflq type        UNSIGNED_BYTE index 4 norm  size 4 stride 16 offset 12
    [2016-Jun-03 11:09:11.889749]:info: Rdr::address (glVertexAttribPointer)     devrec name rpos type                SHORT index 0 norm  size 4 stride 16 offset 0
    [2016-Jun-03 11:09:11.889904]:info: Rdr::address (glVertexAttribPointer)     devrec name rpol type        UNSIGNED_BYTE index 1 norm  size 4 stride 16 offset 8
    [2016-Jun-03 11:09:11.890067]:info: Rdr::address (glVertexAttribPointer)     devrec name rflg type       UNSIGNED_SHORT index 2 norm  size 2 stride 16 offset 12
    [2016-Jun-03 11:09:11.890228]:info: Rdr::address (glVertexAttribPointer)     devrec name rflq type        UNSIGNED_BYTE index 4 norm  size 4 stride 16 offset 12


oglrap-/Rdr::address::

    303 
    304     GLuint       index = location  ;            //  generic vertex attribute to be modified
    305     GLint         size = vnpy->getSize() ;      //  number of components per generic vertex attribute, must be 1,2,3,4
    306     GLboolean     norm = vnpy->getNorm() ;
    307     GLsizei       stride = vnpy->getStride();   // byte offset between consecutive generic vertex attributes, or 0 for tightly packed
    308     const GLvoid* offset = (const GLvoid*)vnpy->getOffset() ;
    309 
    310     // offset of the first component of the first generic vertex attribute 
    311     // in the array in the data store of the buffer currently bound to GL_ARRAY_BUFFER target
    312 
    313     LOG(info) << "Rdr::address (glVertexAttribPointer) "
    314               << std::setw(10) << getShaderTag()
    315               << " name " << name
    316               << " type " << std::setw(20) << vnpy->getTypeName()
    317               << " index " << index
    318               << " norm " << norm
    319               << " size " << size
    320               << " stride " << stride
    321               << " offset " << vnpy->getOffset()
    322               ;
    323 
    324 
    325     if( vnpy->getIatt() )
    326     {
    327         glVertexAttribIPointer(index, size, type, stride, offset);
    328     }
    329     else
    330     {
    331         glVertexAttribPointer(index, size, type, norm, stride, offset);
    332     }
    333     glEnableVertexAttribArray(index);


npy-/ViewNPY::

    .69 void ViewNPY::init()
     70 {
     71     m_bytes    = m_npy->getBytes() ;
     72 
     73     assert(m_item_from_dim == 1 || m_item_from_dim == 2);
     74 
     75     // these dont require the data, just the shape
     76     m_numbytes = m_npy->getNumBytes(0) ;
     77     m_stride   = m_npy->getNumBytes(m_item_from_dim) ;
     78     m_offset   = m_npy->getByteIndex(0,m_j,m_k,m_l) ;  //  i*nj*nk*nl + j*nk*nl + k*nl + l     scaled by sizeoftype
     79 
     80     if( m_npy->hasData() )
     81     {
     82         addressNPY();
     83     }
     84 }

    121 unsigned int ViewNPY::getValueOffset()
    122 {
    123     //   i*nj*nk + j*nk + k ;    i=0
    124     //
    125     // serial offset of the qty within each rec 
    126     // obtained from first rec (i=0)
    127     //
    128     return m_npy->getValueIndex(0,m_j,m_k,m_l);
    129 }


optickscore-/OpticksEvent::

     422 void OpticksEvent::setRecordData(NPY<short>* record_data)
     423 {
     424     m_record_data = record_data  ;
     425 
     426     //                                               j k l sz   type                  norm   iatt   item_from_dim
     427     ViewNPY* rpos = new ViewNPY("rpos",m_record_data,0,0,0,4,ViewNPY::SHORT          ,true,  false, 2);
     428     ViewNPY* rpol = new ViewNPY("rpol",m_record_data,1,0,0,4,ViewNPY::UNSIGNED_BYTE  ,true,  false, 2);
     /// 
     ///    because item_from_dim is 2, must shift the j,k,l spec one to the right and set j=0 
     ///
     429   
     430     ViewNPY* rflg = new ViewNPY("rflg",m_record_data,1,2,0,2,ViewNPY::UNSIGNED_SHORT ,false, true,  2);
     431     // NB k=2, value offset from which to start accessing data to fill the shaders uvec4 x y (z, w)  
     432 
     433     ViewNPY* rflq = new ViewNPY("rflq",m_record_data,1,2,0,4,ViewNPY::UNSIGNED_BYTE  ,false, true,  2);
     434     // NB k=2 again : try a UBYTE view of the same data for access to boundary,m1,history-hi,history-lo
     435 
     436     // structured record array => item_from_dim=2 the count comes from product of 1st two dimensions
     437 
     438 
     439     // ViewNPY::TYPE need not match the NPY<T>,
     440     // OpenGL shaders will view the data as of the ViewNPY::TYPE, 
     441     // informed via glVertexAttribPointer/glVertexAttribIPointer 
     442     // in oglrap-/Rdr::address(ViewNPY* vnpy)
     443 
     444     // standard byte offsets obtained from from sizeof(T)*value_offset 
     445     //rpol->setCustomOffset(sizeof(unsigned char)*rpol->getValueOffset());
     446     // this is not needed
     447 
     448     m_record_attr = new MultiViewNPY("record_attr");
     449 
     450     m_record_attr->add(rpos);
     451     m_record_attr->add(rpol);
     452     m_record_attr->add(rflg);
     453     m_record_attr->add(rflq);


::

    Records NPY<short> have shape  (

    rx_raw :   (500000, 10, 2, 4) : (records) photon step records 

    sizeof(SHORT) == 2 bytes


The "item" has dimension (2,4) ie 8*2 = 16 bytes, so the strides are OK (they must be as rpos at offset 0 works)::

    In [6]: evt.rx_raw.shape
    Out[6]: (500000, 10, 2, 4)

    In [7]: evt.rx_raw.reshape(-1,2,4).shape
    Out[7]: (5000000, 2, 4)

    In [8]: evt.rx_raw.reshape(-1,2,4)
    Out[8]: 
    A()sliced
    A([[[ -6627,  10244,   2210,     16],
            [ 26808,  27408,    769,   3352]],

           [[ -6637,  10248,   2204,     19],
            [-27688,  30935,    769,   1304]],

           [[ -6084,  14152,    773,    827],
            [-27688,  30935,    769,   1048]],

           ..., 
           [[  -427,  17532,  -1714,   2049],
            [-14995, -31256,    772,   2266]],

           [[     0,      0,      0,      0],
            [     0,      0,      0,      0]],

           [[     0,      0,      0,      0],
            [     0,      0,      0,      0]]], dtype=int16)

    In [9]: evt.rx_raw.reshape(-1,2,4)[0]
    Out[9]: 
    A()sliced
    A([[-6627, 10244,  2210,    16],
           [26808, 27408,   769,  3352]], dtype=int16)


