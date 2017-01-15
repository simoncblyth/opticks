Sliced Composited Raytraced Rasterized
=========================================

Test geometries
----------------

Single PMT::

     tpmt-
     tpmt--    # makes the event in tmp
     tpmt-v    # loads event from tmp  

Dyb cerenkov with analytic PMTs::

     op -c --analyticmesh 1   


Oglrap
---------

oglrap/gl/tex/frag.glsl::
 
    15  void main ()
    16 {
    17    frag_colour = texture(ColorTex, texcoord);
    18    float depth = frag_colour.w ;  // alpha is hijacked for depth in pinhole_camera.cu material1_radiance.cu
    19    frag_colour.w = 1.0 ;
    20 
    21    gl_FragDepth = depth  ;
    22 
    23    if(NrmParam.z == 1)
    24    {
    25         if(depth < ScanParam.x || depth > ScanParam.y ) discard ;
    26    }


With GUI "Composition/(nrm) scanmode" can ENABLE NrmParam.z and then use 
sliders to adjust ScanParam x and y.
Hmm the x and y are not independently variable, they are tied to z and w. 

::

    simon:oglrap blyth$ grep getScanParamPtr *.*
    GUI.cc:    float* scanparam = composition->getScanParamPtr() ;
    Rdr.cc:        glUniform4fv(m_scanparam_location, 1, m_composition->getScanParamPtr());
    Renderer.cc:        glUniform4fv(m_scanparam_location, 1, m_composition->getScanParamPtr());
    simon:oglrap blyth$ 

oglrap/GUI.cc::

    496     ImGui::SliderInt( "nrmparam.z", np + 2,  0, 1  );
    497     ImGui::Text(" (nrm) scanmode : %s ",  *(np + 2) == 0 ? "DISABLED" : "ENABLED" );
    498 
    499     float* scanparam = composition->getScanParamPtr() ;
    500     ImGui::SliderFloat( "scanparam.x", scanparam + 0,  0.f, 1.0f, "%0.3f", 2.0f );
    501     ImGui::SliderFloat( "scanparam.y", scanparam + 1,  0.f, 1.0f, "%0.3f", 2.0f );
    502     ImGui::SliderFloat( "scanparam.z", scanparam + 2,  0.f, 1.0f, "%0.3f", 2.0f );
    503     ImGui::SliderFloat( "scanparam.w", scanparam + 3,  0.f, 1.0f, "%0.3f", 2.0f );
    504 
    505     *(scanparam + 0) = fmaxf( 0.0f , *(scanparam + 2) - *(scanparam + 3) ) ;
    506     *(scanparam + 1) = fminf( 1.0f , *(scanparam + 2) + *(scanparam + 3) ) ;
    507 

I suppose the problem here is that to produce something visible need to give two depth values 
where one is constrained to be greater than the other and they are both must be between 0 and 1.
There is also a related problem of coarse/fine ranges as its quite difficult to get what 
you want when all the action is over a small portion of the depth range.

Note that adjusting near/far to tightly contain some geometry of interest
then allows the scan parameters to be much easier to control. 


The slicing applies to ray trace, but usually gives "conic section" cuts as the view
is rarely precisely aligned with the geometry. 

Using orthogonal projection, avoiding rotation and fiddling with ScanParam w/z while watching x/y 
allows to create cross section "sliver" views.

Maybe use a base and offset size ? But that has problems of going beyond the range.

Looks like ImGui may soon have a double ended range slider.

* https://github.com/ocornut/imgui/issues/76

Also a DragFloat looks like worth investigating, will probably need to get current with ImGui.

* https://github.com/ocornut/imgui/issues/180

