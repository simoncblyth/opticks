simtrace_cegs_optimization_ideas
===================================

Simtrace cross section plots are very useful for 
debugging geometry issues.  But in areas where lots 
of surfaces are close together there is a need for very 
high resolution genstep positions to define the surfaces 
but in "desert" regions far from any intersects there 
is no need for genstep sources. 

Essentially want to be able to increase resolution without 
directly scaling the time/storage costs of huge intersect arrays. 

Can think of three possible approaches:

1. calculate sdf of a set of prim at each candidate genstep position
   and apply sdf distance cuts

   * this has disadvantage of needing sdf implementation for all shapes

2. use a 2D histogram of intersects from a former run to guide which 
   genstep positions to use in a subsequent run

   * has advantage that spurious intersects in the histogram can be found
     and those regions examined 

   * needs multi-resolution 2D histo addressing (division level style), 
     as need to use histo from a coarser grid simtrace to inform 
     the choice of gensteps in a higher resolution grid

   * see npy/tests/NGrid3Test.cc for some work on multi-resolution addressing 
     using Morton codes : looks to allow shifting between resolution 
     levels by bit shifting the morton index

3. use uniqing of bit mask manipulated morton codes of 2d intersect positions 
   to act as a sparse histogram without any traditional+slow "histogramming", see::

       npy/tests/mortonlib/morton2d_test.sh
       npy/tests/mortonlib/domain2d_test.sh
          

Favor approach 3 for its low resource nature, just bit manipulations 
on a vector of uint64_t means can work fast with very high resolutions
and low overhead.  


simtrace
-----------

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
    379     cx->setFrame(fr);
    380     // Q: why does cx need the frame ?
    381     // A: rendering viewpoint is based on the frame center, extent and transforms 
    382 
    383     qs->simtrace();
    384     LOG(LEVEL) << "]" ;
    385 }


    0256 void SEvt::setFrame(const sframe& fr )
     257 {
     258     frame = fr ;
     259 
     260     if(SEventConfig::IsRGModeSimtrace())
     261     {
     262         addGenstep( SFrameGenstep::MakeCenterExtentGensteps(frame) );
     263     }


    108 NP* SFrameGenstep::MakeCenterExtentGensteps(sframe& fr)
    109 {
    110     const float4& ce = fr.ce ;
    111     float gridscale = SSys::getenvfloat("GRIDSCALE", 0.1 ) ;
    112 
    113     // CSGGenstep::init
    114     std::vector<int> cegs ;
    115     SSys::getenvintvec("CEGS", cegs, ':', "16:0:9:1000" );
    116 
    117     StandardizeCEGS(ce, cegs, gridscale );  // ce is informational here 
    118     assert( cegs.size() == 7 );
    119 
    120     fr.set_grid(cegs, gridscale);
    121 
    122 
    123     std::vector<float3> ce_offset ;
    124     CE_OFFSET(ce_offset, ce);
    125 
    126     LOG(info)
    127         << " ce " << ce
    128         << " ce_offset.size " << ce_offset.size()
    129         ;
    130 
    131 
    132     int ce_scale = SSys::getenvint("CE_SCALE", 1) ; // TODO: ELIMINATE AFTER RTP CHECK 
    133     if(ce_scale == 0) LOG(fatal) << "warning CE_SCALE is not enabled : NOW THINK THIS SHOULD ALWAYS BE ENABLED " ;
    134 
    135 
    136     Tran<double>* geotran = Tran<double>::FromPair( &fr.m2w, &fr.w2m, 1e-6 );
    137 
    138     NP* gs = MakeCenterExtentGensteps(ce, cegs, gridscale, geotran, ce_offset, ce_scale );
    139 
    140     //gs->set_meta<std::string>("moi", moi );
    141     gs->set_meta<int>("midx", fr.midx() );
    142     gs->set_meta<int>("mord", fr.mord() );
    143     gs->set_meta<int>("iidx", fr.iidx() );
    144     gs->set_meta<float>("gridscale", fr.gridscale() );
    145     gs->set_meta<int>("ce_scale", int(ce_scale) );
    146 
    147     return gs ;
    148 }

::

    211 
    212 NP* SFrameGenstep::MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran, const std::vector<float3>& ce_offset, bool ce_scale ) // sta    tic
    213 {
    ...
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




How to proceed ? smortonhist2d.h smortonhist2d.py
-----------------------------------------------------

* simplify tech from npy/tests/NGrid3Test.cc into something more specialized "sysrap/smortonhist2d.h"
* will need to save/load from C++ and from python, as the isect histos will initially come from python  
* SFrameGenstep::MakeCenterExtentGensteps then needs to be optionally guided by an envvar pointing 
  to a persisted smortonhist2d directory from a prior run (python ana run initially) 
* then the current simple grid loop of SFrameGenstep::MakeCenterExtentGensteps 
  can be guided by where the intersects from the prior run actually are 

Whacky alternative sparse approach
-------------------------------------

* a simple list of morton indices at some resolution level can act as a sparse histogram
* the morton uint64_t indices could be stored within the simtrace array
* then to follow the intersects just need to find uniques after some bitshift

  * actually its more convenient not to bitshift, just zero-ing the least significant bits 
    in groups of 2 (for morton2d) will give coarser coordinates without needing to be 
    concerned with scale changes   

* the unique indices directly give coordinates for the gensteps that are close to the intersects  
* this approach avoids some of the costs of going to higher resolution as there 
  is no need for a mostly empty histo array and slow grid loops over it 


See::

    npy/mortonlib/morton2d_test.sh
  

bit shift morton index
--------------------------

* :google:`bit shift morton index`

* https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/






 



