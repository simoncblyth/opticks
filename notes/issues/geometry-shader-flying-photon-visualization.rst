geometry-shader-flying-photon-visualization
=============================================

* reference on geometry shaders https://open.gl/geometry


oglrap/Scene.cc::

     550 void Scene::initRenderers()
     551 {
     ...
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



oglrap geometry shaders::

    epsilon:oglrap blyth$ l gl/*/geom.glsl
    8 -rw-r--r--  1 blyth  staff  2922 May 16  2020 gl/rec/geom.glsl
    8 -rw-r--r--  1 blyth  staff  1259 May 16  2020 gl/p2l/geom.glsl
    8 -rw-r--r--  1 blyth  staff  2369 May 16  2020 gl/nrmvec/geom.glsl
    8 -rw-r--r--  1 blyth  staff  1997 May 16  2020 gl/nop/geom.glsl
    8 -rw-r--r--  1 blyth  staff  2946 May 16  2020 gl/inrmcull/geom.glsl
    8 -rw-r--r--  1 blyth  staff  2922 May 16  2020 gl/devrec/geom.glsl
    8 -rw-r--r--  1 blyth  staff  1238 May 16  2020 gl/axis/geom.glsl
    8 -rw-r--r--  1 blyth  staff  2615 May 16  2020 gl/altrec/geom.glsl



Review oglrap flying point rendering 
--------------------------------------

oglrap/gl/rec/geom.glsl::
  
    47 layout (lines) in;
    48 layout (points, max_vertices = 1) out;

    59     uint photon_id = gl_PrimitiveIDIn/MAXREC ;     // implies flat step-records are fed in 
    63     vec4 p0 = gl_in[0].gl_Position  ;
    64     vec4 p1 = gl_in[1].gl_Position  ; 
    65     float tc = Param.w / TimeDomain.y ;
    67     uint valid  = (uint(p0.w >= 0.)  << 0) + (uint(p1.w >= 0.) << 1) + (uint(p1.w > p0.w) << 2) ; // sensible ascending times 
    68     uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.x == 0 || photon_id % Pick.x == 0) << 2) ;
    69     uint vselect = valid & select ;



        
