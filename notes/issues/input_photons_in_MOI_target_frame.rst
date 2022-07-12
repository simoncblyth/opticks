input_photons_in_MOI_target_frame
=====================================

From :doc:`U4Stack_Linux_Darwin_difference`

To simplify histories and allow AB comparison in full geometry 
need to be able to transform input photons into any target frame
so can shoot specific PMTs from starting points in the water. 

This is rather similar to what FrameGensteps do but 
with the difference need to apply the transform to the photons. 

Remember want to do things in a way common to A and B 
so that means that the transform needs to be applied to the 
input photon array on CPU 


How to implement : transform of pos, mom, pol 
-------------------------------------------------

* review input photons to see where best to introduce the transform

* do not want to have loads of input photon files so the 
  transform needs to be applied on the fly 

* HMM access to the transform means that the B side will need to 
  access A side geometry

  * HMM: should B side depend on CSG ? Or just U4CF python type 
    access to transforms ?
 

Added stran.h Tran::Apply applying transforms to photon pos/mom/pol
---------------------------------------------------------------------

::

    113 void test_apply_ph()
    114 {    
    115      double data[16] = { 
    116            1., 0., 0.,   0.,
    117            0., 1., 0.,   0.,
    118            0., 0.,-1.,   0.,  
    119            0., 0., 100., 1. } ;
    120      
    121      const Tran<T>* t = Tran<T>::ConvertFromData(&data[0]) ; 
    122      //const Tran<T>* t = Tran<T>::make_translate(0., 0., 100.) ; 
    123      
    124      
    125      const char* name = "RandomDisc100_f8.npy" ; 
    126      const char* path = SPath::Resolve("$HOME/.opticks/InputPhotons" , name, NOOP );
    127      NP* a = NP::Load(path); 
    128      NP* b = Tran<T>::Apply(a, t);
    129      
    130      const char* FOLD = SPath::Resolve("$TMP/stranTest", DIRPATH );
    131      std::cout << " FOLD " << FOLD << std::endl ;
    132      
    133      a->save(FOLD,  "a.npy");
    134      b->save(FOLD,  "b.npy");
    135      t->save(FOLD,  "t.npy");


Now how to access the instance transform
-------------------------------------------

::

     26     static NP* MakeCenterExtentGensteps(sframe& fr);
     27     static NP* MakeCenterExtentGensteps(const float4& ce, const std::vector<int>& cegs, float gridscale, const Tran<double>* geotran, const std::vector<float3>& ce_offset, bool ce_scale ) ;


::

    082 void CSGGenstep::create(const char* moi_, bool ce_offset, bool ce_scale )
     83 {
     84     moi = strdup(moi_);
     85 
     86     LOG(info) << " moi " << moi << " ce_offset " << ce_offset << " ce_scale " << ce_scale ;
     87 
     93     else
     94     {
     95         locate(moi);        // sets: ce, geotran 
     96         configure_grid();   // standardize cegs 
     97         gs = SEvent::MakeCenterExtentGensteps(ce, cegs, gridscale, geotran, ce_offset, ce_scale );
     98     }
     99 

    136 void CSGGenstep::locate(const char* moi_)
    137 {
    138     moi = strdup(moi_) ;
    139 
    140     foundry->parseMOI(midx, mord, iidx, moi );
    141 
    142     LOG(info) << " moi " << moi << " midx " << midx << " mord " << mord << " iidx " << iidx ;
    143     if( midx == -1 )
    144     {   
    145         LOG(fatal)
    146             << " failed CSGFoundry::parseMOI for moi [" << moi << "]"
    147             ;
    148         return ;
    149     }
    150 
    151 
    152     int rc = foundry->getCenterExtent(ce, midx, mord, iidx, m2w, w2m ) ;
    153 
    154     LOG(info) << " rc " << rc << " MOI.ce ("
    155               << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << ")" ;
    156 
    157     LOG(info) << "m2w" << *m2w ;
    158     LOG(info) << "w2m" << *w2m ;
    159 
    160     geotran = Tran<double>::FromPair( m2w, w2m, 1e-6 );    // Tran from stran.h 
    161 
    162     //override_locate(); 
    163 }


    117 int CSGTarget::getFrame(sframe& fr, int inst_idx ) const
    118 {
    119     const qat4* _t = foundry->getInst(inst_idx);
    120 
    121     unsigned ins_idx, gas_idx, ias_idx ;
    122     _t->getIdentity(ins_idx, gas_idx, ias_idx )  ;
    123 
    124     assert( int(ins_idx) == inst_idx );
    125     fr.set_inst(inst_idx);
    126 
    127     // HMM: these values are already there inside the matrices ? 
    128     fr.set_ins_gas_ias(ins_idx, gas_idx, ias_idx ) ;
    129 
    130 
    131     qat4 t(_t->cdata());   // copy the instance (transform and identity info)
    132     const qat4* v = Tran<double>::Invert(&t);     // identity gets cleared in here 
    133 
    134     qat4::copy(fr.m2w,  t);
    135     qat4::copy(fr.w2m, *v);
    136 
    137     const CSGSolid* solid = foundry->getSolid(gas_idx);
    138     fr.ce = solid->center_extent ;
    139 
    140     // although there can be multiple CSGPrim within the CSGSolid
    141     // there is not way from the inst_idx to tell which one is needed
    142     // so use the CSGSolid one as that should combined the ce of all the CSGPrim
    143 
    144     return 0 ;
    145 }
    146 
    147 





Review Input Photons
---------------------


::

     129 /**
     130 SEvt::initInputPhoton
     131 -----------------------
     132 
     133 This is invoked by SEvt::init on instanciating the SEvt instance  
     134 The default "SEventConfig::InputPhoton()" is nullptr meaning no input photons.
     135 This can be changed by setting an envvar in the script that runs the executable, eg::
     136 
     137    export OPTICKS_INPUT_PHOTON=CubeCorners.npy
     138    export OPTICKS_INPUT_PHOTON=$HOME/reldir/path/to/inphoton.npy
     139  
     140 Or within the code of the executable, typically in the main prior to SEvt instanciation, 
     141 using eg::
     142 
     143    SEventConfig::SetInputPhoton("CubeCorners.npy")
     144    SEventConfig::SetInputPhoton("$HOME/reldir/path/to/inphoton.npy")
     145 
     146 When non-null it is resolved into a path and the array loaded at SEvt instanciation
     147 by SEvt::LoadInputPhoton
     148 
     149 **/
     150 
     151 void SEvt::initInputPhoton()
     152 {
     153     const char* ip = SEventConfig::InputPhoton();
     154     if( ip == nullptr ) return ;
     155     NP* a = LoadInputPhoton(ip) ;
     156     setInputPhoton(a);    // this adds placeholder genstep of gentype OpticksGenstep_INPUT_PHOTON
     157 }


::

    1048 /**
    1049 SEvt::setInputPhoton
    1050 ---------------------
    1051 
    1052 Also adds placeholder genstep of gentype OpticksGenstep_INPUT_PHOTON
    1053 
    1054 **/
    1055 
    1056 void SEvt::setInputPhoton(NP* p)
    1057 {
    1058     input_photon = p ;
    1059     assert( input_photon->has_shape(-1,4,4) );
    1060     int numphoton = input_photon->shape[0] ;
    1061     assert( numphoton > 0 );
    1062 
    1063     assert( genstep.size() == 0 ) ; // cannot mix input photon running with genstep running  
    1064 
    1065     quad6 ipgs ;
    1066     ipgs.zero();
    1067     ipgs.set_gentype( OpticksGenstep_INPUT_PHOTON );
    1068     ipgs.set_numphoton( numphoton );
    1069 
    1070     addGenstep(ipgs);
    1071 }


The placeholder genstep has room for the transform. 

::

    202 /**
    203 QEvent::setInputPhoton
    204 ------------------------
    205 
    206 This is a private method invoked only from QEvent::setGenstep
    207 which gets the input photon array from SEvt and uploads 
    208 it to the device. 
    209 When the input_photon array is in double precision it is 
    210 narrowed here prior to upload. 
    211 
    212 **/
    213 
    214 void QEvent::setInputPhoton()
    215 {
    216     input_photon = SEvt::GetInputPhoton();
    217     if( input_photon == nullptr )
    218         LOG(fatal)
    219             << " INCONSISTENT : OpticksGenstep_INPUT_PHOTON by no input photon array "
    220             ;
    221 
    222     assert(input_photon);
    223     assert(input_photon->has_shape( -1, 4, 4) );
    224     assert(input_photon->ebyte == 4 || input_photon->ebyte == 8);
    225 
    226     int num_photon = input_photon->shape[0] ;
    227     assert( evt->num_seed == num_photon );
    228 
    229     NP* narrow_input_photon = input_photon->ebyte == 8 ? NP::MakeNarrow(input_photon) : input_photon ;
    230 
    231     setNumPhoton( num_photon );
    232     QU::copy_host_to_device<sphoton>( evt->photon, (sphoton*)narrow_input_photon->bytes(), num_photon );
    233 }



Within the below, have the geometry. 

::

    128 void G4CXOpticks::simulate()
    129 {
    130     LOG(LEVEL) << desc() ;
    131     assert(cx);
    132     assert(qs);
    133     assert( SEventConfig::IsRGModeSimulate() );
    134     qs->simulate();
    135 }
    136 
    137 void G4CXOpticks::simtrace()
    138 {
    139     assert(cx);
    140     assert(qs);
    141     assert( SEventConfig::IsRGModeSimtrace() );
    142 
    143 
    144     SEvt* sev = SEvt::Get();  assert(sev);
    145 
    146     sev->fr = fd->getFrame() ;  // depends on MOI, fr.ce fr.m2w fr.w2m set by CSGTarget::getFrame 
    147     LOG(LEVEL) << sev->fr ;
    148     SEvt::AddGenstep( SFrameGenstep::MakeCenterExtentGensteps(sev->fr) );
    149     cx->setFrame(sev->fr);
    150 
    151     // where to get the frame could be implicit at this level  
    152 
    153 
    154 
    155     qs->simtrace();
    156 }
    157 






B Side
---------




::

    049 template<typename P>
     50 inline void U4VPrimaryGenerator::GetPhotonParam(
     51      G4ThreeVector& position_mm, G4double& time_ns,
     52      G4ThreeVector& direction,  G4double& wavelength_nm,
     53      G4ThreeVector& polarization, const P& p )
     54 {    
     55      position_mm.set(p.pos.x, p.pos.y, p.pos.z);
     56      time_ns = p.time ;
     57      
     58      direction.set(p.mom.x, p.mom.y, p.mom.z ); 
     59      polarization.set(p.pol.x, p.pol.y, p.pol.z );
     60      wavelength_nm = p.wavelength ;
     61 }
     62 


     95 inline void U4VPrimaryGenerator::GeneratePrimaries(G4Event* event)
     96 {
     97     NP* ph = SGenerate::GeneratePhotons();
     98     if(ph == nullptr) std::cerr
     99          << "U4VPrimaryGenerator::GeneratePrimaries : FATAL : NO PHOTONS " << std::endl
    100          << "compile with MOCK_CURAND to use SGenerate.h curand on CPU" << std::endl
    101          ;
    102     if(ph == nullptr) return ;
    103 
    104     //std::cout << "U4VPrimaryGenerator::GeneratePrimaries" << " ph " << ( ph ? ph->brief() : "-" ) << std::endl ;  
    105 
    106     if( ph->ebyte == 4 )
    107     {
    108         sphoton* pp = (sphoton*)ph->bytes() ;
    109         for(int i=0 ; i < ph->shape[0] ; i++)
    110         {
    111             const sphoton& p = pp[i];
    112             //if(i < 10) std::cout << "U4VPrimaryGenerator::GeneratePrimaries p.desc " << p.desc() << std::endl ; 
    113             G4PrimaryVertex* vertex = MakePrimaryVertexPhoton<sphoton>( p );
    114             event->AddPrimaryVertex(vertex);
    115         }
    116     }
    117     else if( ph->ebyte == 8 )
    118     {
    119         sphotond* pp = (sphotond*)ph->bytes() ;
    120         for(int i=0 ; i < ph->shape[0] ; i++)
    121         {
    122             const sphotond& p = pp[i];
    123             G4PrimaryVertex* vertex = MakePrimaryVertexPhoton<sphotond>( p );
    124             event->AddPrimaryVertex(vertex);
    125         }
    126     }
    127 }


    036 NP* SGenerate::GeneratePhotons()
     37 {
     38     NP* gs = SEvt::GetGenstep();  // user code needs to instanciate SEvt and AddGenstep 
     39     NP* ph = nullptr ;
     40     if(OpticksGenstep_::IsInputPhoton(SGenstep::GetGencode(gs,0)))
     41     {
     42         //std::cout << "SGenerate::GeneratePhotons SEvt::GetInputPhoton " << std::endl ; 
     43         ph = SEvt::GetInputPhoton();
     44     }
     45     else
     46     {
     47         ph = GeneratePhotons(gs);
     48     }
     49     //std::cout << "SGenerate::GeneratePhotons ph " << ( ph ? ph->brief() : "-" ) << std::endl ; 
     50     return ph ;
     51 }


