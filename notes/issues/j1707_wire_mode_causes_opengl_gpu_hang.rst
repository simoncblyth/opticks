j1707 wire mode causes opengl GPU hang
=========================================


Huh : I though I disabled wire mode already BUT
---------------------------------------------------

::

    simon:issues blyth$ op --j1707 --tracer --gltf 3

    ...
    Scene::nextGeometryStyle : none 
    Interactor::key_pressed 265 
    Interactor::key_pressed 265 
    Interactor::key_pressed 265 
    Scene::nextGeometryStyle : wire 
    GPU hang occurred, msgtracer returned -1
    /Users/blyth/opticks/bin/op.sh: line 689: 12950 Abort trap: 6           /usr/local/opticks/lib/OTracerTest --j1707 --tracer --gltf 3
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:issues blyth$ 



Scene
--------


::
    1152 unsigned int Scene::getNumGeometryStyle()
    1153 {
    1154     return m_num_geometry_style == 0 ? int(NUM_GEOMETRY_STYLE) : m_num_geometry_style ;
    1155 }


    1177 void Scene::nextGeometryStyle()
    1178 {
    1179     int next = (m_geometry_style + 1) % getNumGeometryStyle();
    1180     setGeometryStyle( (GeometryStyle_t)next );
    1181 
    1182     const char* stylename = getGeometryStyleName();
    1183     printf("Scene::nextGeometryStyle : %s \n", stylename);
    1184 }

::

     82    public:
     83         // disabled styles after NUM_GEOMETRY_STYLE
     84         typedef enum { BBOX, NORM, NONE, WIRE, NUM_GEOMETRY_STYLE, NORM_BBOX } GeometryStyle_t ;
     85         void setGeometryStyle(Scene::GeometryStyle_t style);
     86         unsigned int getNumGeometryStyle(); // allows ro override the enum
     87         void setNumGeometryStyle(unsigned int num_geometry_style); // used to disable WIRE style for JUNO
     88         void applyGeometryStyle();
     89         static const char* getGeometryStyleName(Scene::GeometryStyle_t style);
     90         const char* getGeometryStyleName();
     91         void nextGeometryStyle();



::

    simon:oglrap blyth$ opticks-find setNumGeometryStyle
    ./oglrap/OpticksViz.cc:            m_scene->setNumGeometryStyle(Scene::WIRE); 
    ./oglrap/Scene.cc:void Scene::setNumGeometryStyle(unsigned int num_geometry_style)
    ./oglrap/Scene.hh:        void setNumGeometryStyle(unsigned int num_geometry_style); // used to disable WIRE style for JUNO
    simon:opticks blyth$ 

::

    172 void OpticksViz::prepareScene(const char* rendermode)
    173 {
    174     if(rendermode)
    175     {
    176         LOG(warning) << "OpticksViz::prepareScene using non-standard rendermode " << rendermode ;
    177         m_scene->setRenderMode(rendermode);
    178     }
    179     else if(m_ok->isJuno())
    180     {
    181         LOG(warning) << "disable GeometryStyle  WIRE for JUNO as too slow " ;
    182 
    183         if(!hasOpt("jwire")) // use --jwire to enable wireframe with JUNO, do this only on workstations with very recent GPUs
    184         {
    185             m_scene->setNumGeometryStyle(Scene::WIRE);
    186         }
    187 
    188         m_scene->setNumGlobalStyle(Scene::GVISVEC); // disable GVISVEC, GVEC debug styles
    189 
    190         m_scene->setRenderMode("bb0,bb1,-global");
    191         std::string rmode = m_scene->getRenderMode();
    192         LOG(info) << "App::prepareViz " << rmode ;
    193     }
    194     else if(m_ok->isDayabay())
    195     {
    196         m_scene->setNumGlobalStyle(Scene::GVISVEC);   // disable GVISVEC, GVEC debug styles
    197     }



