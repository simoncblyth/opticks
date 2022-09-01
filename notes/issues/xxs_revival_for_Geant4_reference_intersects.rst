xxs_revival_for_Geant4_reference_intersects
=============================================

From :doc:`nmskSolidMaskTail_small_line_of_spurious_at_upper_middle`

Confusion with mask lips makes it necessary to get xxs.sh running 
again to provide a reference truth geometry to understand what the problem is.  

* will need to update extg4/tests/X4IntersectSolidTest.cc to the gxt way of doing things


Started with::

   x4
   ./xxs0.sh 


X4Intersect currently using SCenterExtentGenstep not SFrameGenstep like gxt.sh does::

     20 /**
     21 X4Intersect::Scan
     22 -------------------
     23 
     24 Used from tests/X4IntersectSolidTest.cc which is used by xxs.sh 
     25 
     26 **/
     27 
     28 void X4Intersect::Scan(const G4VSolid* solid, const char* name, const char* basedir )  // static
     29 {
     30     assert( solid && "X4Intersect::Scan requires solid");
     31     
     32     X4Intersect* x4i = new X4Intersect(solid);
     33     x4i->scan(); 
     34     
     35     const std::string& solidname = solid->GetName() ;
     36     
     37     const char* outdir = SPath::Resolve(basedir, name, "X4Intersect", DIRPATH );
     38     
     39     LOG(info)
     40         << "x4i.desc " << x4i->desc()
     41         << " solidname " << solidname.c_str()
     42         << " name " << name 
     43         << " outdir " << outdir
     44         ;  
     45         
     46     SCenterExtentGenstep* cegs = x4i->cegs ;
     47     cegs->set_meta<std::string>("name", name);
     48     cegs->set_meta<int>("iidx", 0 ); 
     49     cegs->save(outdir); 
     50 }   
i

::

    185 /**
    186 X4Intersect::scan_
    187 -------------------
    188 
    189 Using the *pp* vector of "photon" positions and directions
    190 calulate distances to the solid.  Collect intersections
    191 into *ss* vector. 
    192 
    193 TODO: 
    194 
    195 * adopt simtrace layout, even although some aspects like surface normal 
    196   will  be missing from it. This means cegs->pp and cegs->ii will kinda merge 
    197 
    198 **/
    199 
    200 void X4Intersect::scan_()
    201 {
    202     const std::vector<quad4>& pp = cegs->pp ;
    203     std::vector<quad4>& ii = cegs->ii ;
    204 
    205     bool dump = false ;
    206     for(unsigned i=0 ; i < pp.size() ; i++)
    207     {
    208         const quad4& p = pp[i];
    209 
    210         G4ThreeVector pos(p.q0.f.x, p.q0.f.y, p.q0.f.z);
    211         G4ThreeVector dir(p.q1.f.x, p.q1.f.y, p.q1.f.z);
    212 
    213         G4double t = Distance( solid, pos, dir, dump );
    214 
    215         if( t == kInfinity ) continue ;
    216         G4ThreeVector ipos = pos + dir*t ;
    217 
    218         quad4 isect ;
    219         isect.zero();
    220 
    221         isect.q0.f.x = float(ipos.x()) ;
    222         isect.q0.f.y = float(ipos.y()) ;
    223         isect.q0.f.z = float(ipos.z()) ;
    224         isect.q0.f.w = float(t) ;






SCenterExtentGenstep vs SFrameGenstep
----------------------------------------

* primary difference is sframe with is used to carry 
  the config in a more encapsulated way 

* major user is from python side 


simtrace 
------------------

::

    362 void G4CXOpticks::simtrace()
    363 {
    373     SEvt* sev = SEvt::Get();  assert(sev);
    375     sframe fr = fd->getFrame() ;  // depends on MOI, fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 
    376     sev->setFrame(fr);   // 
    378     cx->setFrame(fr);
    382     qs->simtrace();
    384 }

::

    256 void SEvt::setFrame(const sframe& fr )
    257 {
    258     frame = fr ;
    259 
    260     if(SEventConfig::IsRGModeSimtrace())
    261     {
    262         addGenstep( SFrameGenstep::MakeCenterExtentGensteps(frame) );
    263     } 







