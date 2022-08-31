nmskSolidMaskTail_small_line_of_spurious_at_upper_middle
============================================================


::

     geom_ # set default to nmskSolidMaskTail

     gc
     GEOM=nmskSolidMaskTail ./translate.sh 

::

     gx
     ./gxt.sh         # workstation
     ./gxt.sh grab    # laptop
     ./gxt.sh ana     # laptop


Central spurious are in line x=-1:1 z=-39 

Morton finder fails to select them as too many together, so increase the SPURIOUS_CUT to find them::

    MASK=t SPURIOUS=1 ./gxt.sh ana 
    MASK=t SPURIOUS=4 ./gxt.sh ana 

Looking closer at the sides edge shows nasty lips, with disconnected lines of spurious at z=-39.

Will need to look at constituents, add the below to gc:mtranslate.sh::

    geomlist_nmskSolidMaskTail(){ cat << EOL
    nmskSolidMaskTail

    nmskTailOuter
    nmskTailOuterIEllipsoid
    nmskTailOuterITube
    nmskTailOuterI
    nmskTailOuterIITube

    nmskTailInner
    nmskTailInnerIEllipsoid
    nmskTailInnerITube
    nmskTailInnerI
    nmskTailInnerIITube 

    EOL
    }



TODO : use extg4/xxs.sh to get the Geant4 view on this nmsk solids
---------------------------------------------------------------------- 

* will need to update extg4/tests/X4IntersectSolidTest.cc to the gxt way of doing things

started with::

   x4
   ./xxs0.sh 



Checking nmskTailInner with uncoincide_z 1mm : has issues on top edge
-------------------------------------------------------------------------

HMM: I need a way to have both uncoincide and not under different names, 
or have some versioning. Currently the manager prefix eg "nmsk" "nnvt" "hama" "hmsk" 
is skipped over, so could use suffix like "__opt" to be interpreted into
variations like uncoincide. 

::

    #geom=nmskSolidMask
    #geom=nmskMaskOut
    #geom=nmskMaskIn

    #geom=nmskSolidMaskTail

    #geom=nmskTailOuter
    #geom=nmskTailOuterIEllipsoid
    #geom=nmskTailOuterITube
    #geom=nmskTailOuterI
    #geom=nmskTailOuterIITube

    geom=nmskTailInner
    #geom=nmskTailInnerIEllipsoid
    #geom=nmskTailInnerITube
    #geom=nmskTailInnerI
    #geom=nmskTailInnerIITube 



    


::

    831 G4VSolid* NNVTMaskManager::getSolid(const char* name)
    832 {
    833     if (logicMaskVirtual == nullptr )
    834     {
    835         std::cout << "NNVTMaskManager::getSolid booting with getLV " << name << std::endl ;
    836         getLV();
    837     }
    838 
    839     G4VSolid* solid = nullptr ;
    840     // makeMaskOutLogical 
    841     if(strcmp(name, "SolidMaskVirtual") == 0 ) solid = SolidMaskVirtual ;
    842 
    843     // makeMaskLogical
    844     if(strcmp(name, "TopOut") == 0 )     solid = Top_out ;
    845     if(strcmp(name, "BottomOut") == 0 )  solid = Bottom_out ;
    846     if(strcmp(name, "MaskOut") == 0 )    solid = Mask_out ;
    847 
    848     if(strcmp(name, "TopIn") == 0 )      solid = Top_in ;




Checking nmskTailOuter : no sign of spurious
---------------------------------------------

Union of 4:

* very thin tubs on top from -39.0 to -39.3 mm
* big ellipsoid "bowl"
* medium cylinder base of bowl 
* sliver ellipsoid base : would make the bowl rotate around 

The thin tubs on top is a small feature on "big" solid, 
this causes intersect simtrace viz only issue : no rays hit the 0.3 mm thin edge. 

Compare the inner and outer::

     epsilon:g4cx blyth$ ./cf_gxt.sh 

Shows all problems are on the top edge. And situation there is confused due to very thin lip with both inner and outer.
Need to blast three area with rays to see whats what. 
Regions to illuminate::

    X -270 -> -240 
    Z  -35 -> -45  

    X  240 ->  270 
    Z  -35 -> -45  

    X   -10 ->  10 
    Z  -35 -> -45  


How to illuminate regions ?
-----------------------------

::

    363 void G4CXOpticks::simtrace()
    364 {
    365 #ifdef __APPLE__
    366      LOG(fatal) << " APPLE skip " ;
    367      return ;
    368 #endif
    369     LOG(LEVEL) << "[" ;
    370     assert(cx);
    371     assert(qs);
    372     assert( SEventConfig::IsRGModeSimtrace() );
    373 
    374     SEvt* sev = SEvt::Get();  assert(sev);
    375 
    376     sframe fr = fd->getFrame() ;  // depends on MOI, fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 
    377     sev->setFrame(fr);   // 
    378 

::

     256 void SEvt::setFrame(const sframe& fr )
     257 {
     258     frame = fr ;
     259 
     260     if(SEventConfig::IsRGModeSimtrace())
     261     {
     262         addGenstep( SFrameGenstep::MakeCenterExtentGensteps(frame) );
     263     }

::

    108 NP* SFrameGenstep::MakeCenterExtentGensteps(sframe& fr)
    109 {
    110     const float4& ce = fr.ce ;
    111     float gridscale = SSys::getenvfloat("GRIDSCALE", 0.1 ) ;
    112 
    113     // CSGGenstep::init
    114     std::vector<int> cegs ;
    115     SSys::getenvintvec("CEGS", cegs, ':', "16:0:9:1000" );
    116 


Doing the below again with different ranges seems simplest, so 
can then NP::Concatenate multiple genstep arrays::

    252     for(int ip=0 ; ip < num_offset ; ip++)   // planes
    253     {
    254         const float3& offset = ce_offset[ip] ;
    255 
    256         gs.q1.f.x = offset.x ;
    257         gs.q1.f.y = offset.y ;
    258         gs.q1.f.z = offset.z ;
    259         gs.q1.f.w = 1.f ;
    260 
    261         for(int ix=ix0 ; ix < ix1+1 ; ix++ )
    262         for(int iy=iy0 ; iy < iy1+1 ; iy++ )
    263         for(int iz=iz0 ; iz < iz1+1 ; iz++ )
    264         {
    265             double tx = double(ix)*local_scale ;
    266             double ty = double(iy)*local_scale ;
    267             double tz = double(iz)*local_scale ;
    268 
    269             const Tran<double>* local_translate = Tran<double>::make_translate( tx, ty, tz );
    270             // grid shifts 
    271 
    272             bool reverse = false ;
    273             const Tran<double>* transform = Tran<double>::product( geotran, local_translate, reverse );
    274 
    275             qat4* qc = Tran<double>::ConvertFrom( transform->t ) ;
    276 
    277             unsigned gsid = SGenstep::GenstepID(ix,iy,iz,ip) ;
    278 
    279             SGenstep::ConfigureGenstep(gs, OpticksGenstep_FRAME, gridaxes, gsid, photons_per_genstep );
    280 
    281             qc->write(gs);  // copy qc into gs.q2,q3,q4,q5
    282 
    283             gensteps.push_back(gs);
    284             photon_offset += std::abs(photons_per_genstep) ;
    285         }
    286     }


The default CEGS 16:0:9:1000 leads to a grid system from -16->16 and -9->9 so can use 
those basis grid coordinates to pick where to put extra gensteps. 


So for "+" original grid a highlighted cell gives three more:: 


       +         +        + 

       1    1    3   3    

       +    1    +   3    +

       0    0    2   2        

       +    0    +   2    +


