OptiX Changing Event Buffer
=============================

Objective
------------

Minimize what needs to be done per-event by placing everything
possible into once only initialization.


Observation : trying to prelaunch in init fails
---------------------------------------------------

Clearly split:

* once-only setup, all the way to pre-launch 
* per-event just final launch 

Do multi-event propagations to test the split.


Context validation fails without evt buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    Application Specific Information:
    terminating with uncaught exception of type optix::Exception: Invalid value (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Non-initialized variable record_buffer:  Buffer(1d, 8 byte element), file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Context/ValidationManager.cpp, line: 118)
    abort() called

    Thread 0 Crashed:: Dispatch queue: com.apple.main-thread
    8   libOptiXRap.dylib               0x000000010afa8eb9 optix::ContextObj::checkError(RTresult) const + 121 (optixpp_namespace.h:1832)
    9   libOptiXRap.dylib               0x000000010afa8f17 optix::ContextObj::validate() + 55 (optixpp_namespace.h:1877)
    10  libOptiXRap.dylib               0x000000010afbae9e OContext::launch(unsigned int, unsigned int, unsigned int, unsigned int, OTimes*) + 542 (OContext.cc:237)
    11  libOptiXRap.dylib               0x000000010afce036 OPropagator::prelaunch() + 390 (OPropagator.cc:328)
    12  libOptiXRap.dylib               0x000000010afcae69 OEngineImp::preparePropagator() + 1081 (OEngineImp.cc:188)
    13  libOptiXRap.dylib               0x000000010afc8f42 OEngineImp::init() + 66 (OEngineImp.cc:82)
    14  libOptiXRap.dylib               0x000000010afc8ed4 OEngineImp::OEngineImp(OpticksHub*) + 228 (OEngineImp.cc:73)
    15  libOptiXRap.dylib               0x000000010afc8f6d OEngineImp::OEngineImp(OpticksHub*) + 29 (OEngineImp.cc:73)
    16  libOpticksOp.dylib              0x000000010b4b52df OpEngine::OpEngine(OpticksHub*) + 95 (OpEngine.cc:23)
    17  libOpticksOp.dylib              0x000000010b4b534d OpEngine::OpEngine(OpticksHub*) + 29 (OpEngine.cc:27)
    18  libGGeoView.dylib               0x000000010b5aac9f OKPropagator::OKPropagator(OpticksHub*, OpticksIdx*, OpticksViz*) + 143 (OKPropagator.cc:38)
    19  libGGeoView.dylib               0x000000010b5aaefd OKPropagator::OKPropagator(OpticksHub*, OpticksIdx*, OpticksViz*) + 45 (OKPropagator.cc:46)
    20  libGGeoView.dylib               0x000000010b5aa12f OKMgr::OKMgr(int, char**) + 575 (OKMgr.cc:37)
    21  libGGeoView.dylib               0x000000010b5aa453 OKMgr::OKMgr(int, char**) + 35 (OKMgr.cc:45)
    22  OKTest                          0x00000001073fba38 main + 1368 (OKTest.cc:60)
    23  libdyld.dylib                   0x00007fff8a86f5fd start + 1



Changing Event Buffers ? 
------------------------------

OptiX 400 Guide, p14 on OptiX Buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host access to the data stored within a buffer is performed with the
rtBufferMap function. This function returns a pointer to a one dimensional
array representation of the buffer data. All buffers must be unmapped via
rtBufferUnmap before context validation will succeed.


* this implies can change content then launch again without ceremony


OptiX 400 Guide, p60 on Interop Buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenGL buffer objects like PBOs and VBOs can be encapsulated for use in OptiX
with rtBufferCreateFromGLBO. 

* The resulting buffer is only a reference to the OpenGL data; 

* the size of the OptiX buffer as well as the format have to be set
  via rtBufferSetSize and rtBufferSetFormat. 

* When the OptiX buffer is destroyed, the state of the OpenGL buffer object is unaltered. 

* Once an OptiX buffer is created, the original GL buffer object is immutable, 
  meaning the properties of the GL object like its size cannot be changed while registered with OptiX.
  However, it is still possible to read and write buffer data to the GL buffer
  object using the appropriate GL functions. 

* If it is necessary to change properties of an object, 
  first call rtBufferGLUnregister before making changes.
  After the changes are made the object has to be registered again with
  rtBufferGLRegister. This is necessary to allow OptiX to access the object’s
  data again. 

* Registration and unregistration calls are expensive and should be avoided if possible.


Thoughts
~~~~~~~~~~

* compute mode performance is what matters

* performance in interop mode doesnt matter much, as just for debugging anyhow, 
  maybe continue to prelaunch for every launch in interop but try to 
  update and modify preexisting compute buffers

* maybe can set maximal buffer size, and just use partial ? 


Experience
~~~~~~~~~~~~~~

* https://devtalk.nvidia.com/default/topic/525863/changing-the-size-of-an-optix-buffer/

You can take a look at most of the resize function for the windowing in the Optix SDK::

    _context["eyeHitBuffer"]->getBuffer()->setSize(NEW_WIDTH, NEW_HEIGHT);


Changing OpenGL Buffers
------------------------

* https://www.opengl.org/wiki/Vertex_Specification_Best_Practices

If the contents of your VBO will be dynamic, should you call glBufferData or
glBufferSubData (or glMapBuffer)?  If you will be updating a small section, use
glBufferSubData. If you will update the entire VBO, use glBufferData (this
information reportedly comes from a nVidia document). However, another approach
reputed to work well when updating an entire buffer is to call glBufferData
with a NULL pointer, and then glBufferSubData with the new contents. The NULL
pointer to glBufferData lets the driver know you don't care about the previous
contents so it's free to substitute a totally different buffer, and that helps
the driver pipeline uploads more efficiently.

Another thing you can do is double buffered VBO. This means you make 2 VBOs. On
frame N, you update VBO 2 and you render with VBO 1. On frame N+1, you update
VBO 1 and you render from VBO 2. This also gives a nice boost in performance
for nVidia and ATI/AMD.


glBufferData
~~~~~~~~~~~~~

* http://stackoverflow.com/questions/15821969/what-is-the-proper-way-to-modify-opengl-vertex-buffer

To resize, needs to call glBufferData again, 

* whilst bound to the old buffer id ?

* within Opticks would have to pass buffer id between OpticksEvent constituents
  so the next OpticksEvent takes over the old id ? Prior to  


oglrap- Scene::uploadEvent
-----------------------------

::


     718 void Scene::uploadEvent(OpticksEvent* evt)
     719 {
     720     if(!evt)
     721     {
     722        LOG(fatal) << "Scene::uploadEvt no evt " ;
     723        assert(evt);
     724     }
     725 
     726     // The Rdr call glBufferData using bytes and size from the associated NPY 
     727     // the bytes used is NULL when npy->hasData() == false
     728     // corresponding to device side only OpenGL allocation
     729 
     730     if(m_genstep_renderer)
     731         m_genstep_renderer->upload(evt->getGenstepAttr());
     732 
     733     if(m_nopstep_renderer)
     734          m_nopstep_renderer->upload(evt->getNopstepAttr(), false);
     735 
     736     if(m_photon_renderer)
     737          m_photon_renderer->upload(evt->getPhotonAttr());
     738 
     739 
     740     uploadRecordAttr(evt->getRecordAttr());



The upload creates new OpenGL buffer object and copies to it::

    272 void Rdr::upload(NPYBase* npy, ViewNPY* vnpy)
    273 {
    274     // handles case of multiple mvn referring to the same buffer without data duplication,
    275     // by maintaining a list of NPYBase which have been uploaded to the Device
    276 
    277     prepare_vao();
    278 
    279     MultiViewNPY* parent = vnpy->getParent();
    280     assert(parent);
    281 
    282     bool dynamic = npy->isDynamic();
    283 
    ///
    ///   hmm notion of buffer identity  used to see if 
    //    uploaded already is coming from the host npy
    ///   not from the buffer_id 
    ///
    284     if(m_device->isUploaded(npy))
    285     {
    286         GLuint buffer_id = npy->getBufferId();
    287         log("Rdr::upload BindBuffer to preexisting buffer_id:",buffer_id)  ;
    288         assert(buffer_id > 0);
    289         glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    290     }
    291     else
    292     {
    293         void* data = npy->getBytes();
    294         unsigned int nbytes = npy->getNumBytes(0) ;
    295 
    296         char repdata[16] ;
    297         snprintf( repdata, 16, "%p", data );
    298 
    299         GLuint buffer_id ;
    300         glGenBuffers(1, &buffer_id);
    301         glBindBuffer(GL_ARRAY_BUFFER, buffer_id);
    302 
    303         LOG(info) << " up "
    304                   << std::setw(15) << parent->getName()
    305                   << std::setw(5)  << vnpy->getName()
    306                   << " count " << std::setw(8) << vnpy->getCount()
    307                   << " shape " << std::setw(20) << vnpy->getShapeString()
    308                   << " buffer_id " << std::setw(5) << buffer_id
    309                   << " data " << std::setw(16) << repdata
    310                   << " hasData " << std::setw(5) << ( npy->hasData() ? "Y" : "N" )
    311                   << " nbytes " << std::setw(10) << nbytes
    312                   << " " << (dynamic ? "GL_DYNAMIC_DRAW" : "GL_STATIC_DRAW" )
    313                   ;
    314 
    315         glBufferData(GL_ARRAY_BUFFER, nbytes, data, dynamic ? GL_DYNAMIC_DRAW : GL_STATIC_DRAW  );
    316 
    317         npy->setBufferId(buffer_id);
    318         m_device->add(npy);
    319     }
    320 }







