ellipsoid_not_maintaining_shape_within_boolean_combination
=============================================================

Reproduce
-----------

::

    ## update the PMTSim lib used by the below : PMTSim is standalone provider of test G4VSolid
    jps       # cd ~/j/PMTSim   
    om

    ## check the Geant4 intersects using X4IntersectTest  

    x4         # cd ~/opticks/extg4
    ./xxs.sh   #   using UnionOfHemiEllipsoids or UnionOfHemiEllipsoids-50


    ## perform GeoChain geometry conversion

    gc   # cd ~/opticks/GeoChain
    om

    ## with GeoChainSolidTest

    ./run.sh # using UnionOfHemiEllipsoids 
    ./run.sh # using UnionOfHemiEllipsoids-50 
    ./run.sh # using pmt_solid    (creates /tmp/blyth/opticks/GeoChain/pmt_solid/CSGFoundry/)

    ## try OptiX pre-7 render

    cx  
    om    # pre-7 build

    ./cxr_geochain.sh     ## of UnionOfHemiEllipsoids  and UnionOfHemiEllipsoids-50 


    ## try OptiX 7 2D render
    cx
    ./b7    # OptiX 7 build not yet standardly done by the "om" build

    CXS=pmt_solid ./cxs.sh 

    ## view the OptiX 7 2D render in laptop

    cx
    ./grab.sh 
    CXS=pmt_solid ./cxs.sh 


    

UnionOfHemiEllipsoids        
   looks fine, like full ellipsoid

UnionOfHemiEllipsoids-50   
   lower hemi-ellipsoid becomes smaller than upper 

   * looks like the translation transform is stomping on the scale transform

pmt_solid
   before fix : lower side of bulb smaller than upper, after fix : looks correct

 


X4Solid::convertBooleanSolid
------------------------------

Looks like the handling of the boolean transform happens at the 
level of the right child, not here.  


X4Solid::convertDisplacedSolid LOOKS TO BE STOMPING ON THE ELLIPSOID SCALE TRANSFORM
----------------------------------------------------------------------------------------

* possible fix below makes UnionOfHemiEllipsoids-50 look as expected 


::

     269 /**
     270 X4Solid::convertDisplacedSolid
     271 -------------------------------
     272 
     273 The constituents of BooleanSolid which have displacements 
     274 are represented by a G4DisplacedSolid
     275 
     276 Note possibly fixed issue notes/issues/ellipsoid_not_maintaining_shape_within_boolean_combination.rst
     277 was addressed by combining the prior scale transform with the displaced transform from 
     278 the boolean combination. Although it seems to work the testing has not yet used rotation
     279 so there could be transform combination glitches with other transforms.
     280  
     281 **/
     282 
     283 void X4Solid::convertDisplacedSolid()
     284 {   
     285     const G4DisplacedSolid* const disp = static_cast<const G4DisplacedSolid*>(m_solid);
     286     G4VSolid* moved = disp->GetConstituentMovedSolid() ;
     287     assert( dynamic_cast<G4DisplacedSolid*>(moved) == NULL ); // only a single displacement is handled
     288     
     289     bool top = false ;  // never top of tree : expect to always be a boolean RHS
     290     X4Solid* xmoved = new X4Solid(moved, m_ok, top);
     291     setDisplaced(xmoved);
     292     
     293     nnode* a = xmoved->root();
     294     
     295     LOG(LEVEL)
     296         << " a.csgname " << a->csgname()
     297         << " a.transform " << a->transform
     298         ;
     299     
     300     glm::mat4 xf_disp = X4Transform3D::GetDisplacementTransform(disp);
     301     const nmat4triple* disp_transform = new nmat4triple(xf_disp) ;
     302     
     303     if( a->transform == nullptr )  // no preexisting transform, no stomp worries
     304     {
     305         a->transform = disp_transform ;
     306     }   
     307     else
     308     {
     309         // TODO: CHECK TO SEE IF THIS AVOIDS STOMPING ON PRIOR TRANSFORM SUCH AS ELLIPSOID SCALE 
     310         const nmat4triple* prior_transform = a->transform ; 
     311         
     312         bool reverse = false ;   // adhoc guess of transform order : to be checked by comparing results with G4 
     313         const nmat4triple* comb_transform = nmat4triple::product( disp_transform, prior_transform, reverse ); 
     314         a->transform = comb_transform ; 
     315         
     316         DumpTransform("prior_transform", prior_transform );
     317         DumpTransform("disp_transform", disp_transform );  
     318         DumpTransform("comb_transform", comb_transform );
     319     }
     320 
     321     setRoot(a);
     322 }




