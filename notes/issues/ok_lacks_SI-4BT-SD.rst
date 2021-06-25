ok_lacks_SI-4BT-SD
======================

Summary
---------



After Skipping Degenerate Pyrex///Pyrex
-------------------------------------------

* continuing from prior issue :doc:`ok_less_SA_more_AB`


Looks like 3BT to 4BT mismatch, CRecorder G4 microStep skipping perhaps not always working would explain 3 vs 4 mismatch 

* this issue was there previously, just now it becomes the most prominent one  
* hmm could angular efficiency be an issue too regards relative SA/SD 


::

    In [3]: ab.his[:40]
    Out[3]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11142     11142       310.47/62 =  5.01  (pval:1.000 prob:0.000)  
       n             iseq         a         b    a-b               c2          a/b                   b/a           [ns] label
    0000               42      1666      1621     45             0.62        1.028 +- 0.025        0.973 +- 0.024  [2 ] SI AB
    0001            7ccc2      1367      1258    109             4.53        1.087 +- 0.029        0.920 +- 0.026  [5 ] SI BT BT BT SD       ## OK EXCESS SI-3BT-SD
    0002            8ccc2       624       766   -142            14.51        0.815 +- 0.033        1.228 +- 0.044  [5 ] SI BT BT BT SA       ## OK LACKS SI-3BT-SA 
    0003           7ccc62       680       543    137            15.35        1.252 +- 0.048        0.799 +- 0.034  [6 ] SI SC BT BT BT SD    ## OK EXCESS SI-SC-3BT-SD    
    0004             8cc2       570       496     74             5.14        1.149 +- 0.048        0.870 +- 0.039  [4 ] SI BT BT SA
    0005              452       408       495    -87             8.38        0.824 +- 0.041        1.213 +- 0.055  [3 ] SI RE AB
    0006           7ccc52       405       385     20             0.51        1.052 +- 0.052        0.951 +- 0.048  [6 ] SI RE BT BT BT SD
    0007              462       399       351     48             3.07        1.137 +- 0.057        0.880 +- 0.047  [3 ] SI SC AB
    0008           8ccc62       270       258     12             0.27        1.047 +- 0.064        0.956 +- 0.059  [6 ] SI SC BT BT BT SA
    0009          7ccc662       255       195     60             8.00        1.308 +- 0.082        0.765 +- 0.055  [7 ] SI SC SC BT BT BT SD
    0010            8cc62       195       180     15             0.60        1.083 +- 0.078        0.923 +- 0.069  [5 ] SI SC BT BT SA
    0011           8ccc52       176       191    -15             0.61        0.921 +- 0.069        1.085 +- 0.079  [6 ] SI RE BT BT BT SA
    0012          7ccc652       186       165     21             1.26        1.127 +- 0.083        0.887 +- 0.069  [7 ] SI RE SC BT BT BT SD
    0013               41       156       160     -4             0.05        0.975 +- 0.078        1.026 +- 0.081  [2 ] CK AB
    0014             4552       118       152    -34             4.28        0.776 +- 0.071        1.288 +- 0.104  [4 ] SI RE RE AB
    0015            8cc52       136       133      3             0.03        1.023 +- 0.088        0.978 +- 0.085  [5 ] SI RE BT BT SA
    0016          7ccc552       123       120      3             0.04        1.025 +- 0.092        0.976 +- 0.089  [7 ] SI RE RE BT BT BT SD
    0017             4662       125       114     11             0.51        1.096 +- 0.098        0.912 +- 0.085  [4 ] SI SC SC AB
    0018             4652       118       108     10             0.44        1.093 +- 0.101        0.915 +- 0.088  [4 ] SI RE SC AB
    0019             4cc2       121       104     17             1.28        1.163 +- 0.106        0.860 +- 0.084  [4 ] SI BT BT AB                  ## NOW CONSISTENT
    0020           7cccc2        50       151   -101            50.75        0.331 +- 0.047        3.020 +- 0.246  [6 ] SI BT BT BT BT SD            ## OK LACKS SI-4BT-SD 
    0021          8ccc662        78        99    -21             2.49        0.788 +- 0.089        1.269 +- 0.128  [7 ] SI SC SC BT BT BT SA
    0022         7ccc6662        76        82     -6             0.23        0.927 +- 0.106        1.079 +- 0.119  [8 ] SI SC SC SC BT BT BT SD
    0023          8ccc652        72        72      0             0.00        1.000 +- 0.118        1.000 +- 0.118  [7 ] SI RE SC BT BT BT SA
    0024          8ccc552        62        69     -7             0.37        0.899 +- 0.114        1.113 +- 0.134  [7 ] SI RE RE BT BT BT SA
    0025           8cc662        57        65     -8             0.52        0.877 +- 0.116        1.140 +- 0.141  [6 ] SI SC SC BT BT SA
    0026         7ccc6652        57        63     -6             0.30        0.905 +- 0.120        1.105 +- 0.139  [8 ] SI RE SC SC BT BT BT SD
    0027             4562        51        57     -6             0.33        0.895 +- 0.125        1.118 +- 0.148  [4 ] SI SC RE AB
    0028         7ccc6552        50        57     -7             0.46        0.877 +- 0.124        1.140 +- 0.151  [8 ] SI RE RE SC BT BT BT SD
    0029          7ccc562        68        38     30             8.49        1.789 +- 0.217        0.559 +- 0.091  [7 ] SI SC RE BT BT BT SD
    0030              4c2        58        46     12             1.38        1.261 +- 0.166        0.793 +- 0.117  [3 ] SI BT AB
    0031           8cc652        48        54     -6             0.35        0.889 +- 0.128        1.125 +- 0.153  [6 ] SI RE SC BT BT SA
    0032          7cccc62        19        77    -58            35.04        0.247 +- 0.057        4.053 +- 0.462  [7 ] SI SC BT BT BT BT SD       ## OK LACKS SI-SC-4BT-SD
    0033           8cccc2        28        67    -39            16.01        0.418 +- 0.079        2.393 +- 0.292  [6 ] SI BT BT BT BT SA          ## OK LACKS SI-4BT-SA
    0034            4cc62        50        40     10             1.11        1.250 +- 0.177        0.800 +- 0.126  [5 ] SI SC BT BT AB
    0035            46662        40        43     -3             0.11        0.930 +- 0.147        1.075 +- 0.164  [5 ] SI SC SC SC AB
    0036           8cc552        30        50    -20             5.00        0.600 +- 0.110        1.667 +- 0.236  [6 ] SI RE RE BT BT SA
    0037            4ccc2        56        17     39            20.84        3.294 +- 0.440        0.304 +- 0.074  [5 ] SI BT BT BT AB
    0038         7ccc5552        33        40     -7             0.67        0.825 +- 0.144        1.212 +- 0.192  [8 ] SI RE RE RE BT BT BT SD
    .                              11142     11142       310.47/62 =  5.01  (pval:1.000 prob:0.000)  

    In [4]: 



Local frame dx of the 3BT and 4BT may be informative
--------------------------------------------------------

* potentially an angle of incidence effect
* dx with double precision step points is G4 only 
* need nidx in G4 pflags in order to get the transform corresponding to final photon positions
  so can then see local positions within various psel 


what about skipping and the all_volume arrays, does that mean the indices will be non-contiguous ?  
------------------------------------------------------------------------------------------------------

NO, the skipping changes what is in the GMergedMesh it does not change the **all_volume** arrays.


::

    epsilon:GNodeLib blyth$ ipython -i $(which np.py) *.npy
    a :                                          all_volume_bbox.npy :       (283812, 2, 4) : 91346e2be89bf2562e00f46025cf6d3a : 20210615-1403 
    b :                                 all_volume_center_extent.npy :          (283812, 4) : 339773b74d10be3ea97c1e34fc99e6a0 : 20210615-1403 
    c :                                      all_volume_identity.npy :          (283812, 4) : 6d9254dc39abca7829416d89198a82a5 : 20210615-1403 
    d :                            all_volume_inverse_transforms.npy :       (283812, 4, 4) : 1e936a997a5a200dc83cf0539f812530 : 20210615-1403 
    e :                                      all_volume_nodeinfo.npy :          (283812, 4) : 546376acb0ba68d868799f2d83eaa698 : 20210615-1403 
    f :                                    all_volume_transforms.npy :       (283812, 4, 4) : 93680ce18d4ed44c55d39d3489f38941 : 20210615-1403 


    In [1]: c      ## no gaps in all_volume_identity
    Out[1]: 
    array([[       0,        0,  8257536,        0],
           [       1,        1,   786433,        0],
           [       2,        2,   720898,        0],
           ...,
           [  283809, 67723011,  7864343,        0],
           [  283810, 67723012,  7733282,    45612],
           [  283811, 67723013,  7798819,        0]], dtype=uint32)

    In [4]: c0 = np.arange(len(c), dtype=np.uint32)

    In [5]: c0
    Out[5]: array([     0,      1,      2, ..., 283809, 283810, 283811], dtype=uint32)

    In [6]: np.all( c[:,0] == c0 )
    Out[6]: True


These arrays are collected in GNodeLib::addVolume::

    403 /**
    404 GNodeLib::addVolume (precache)
    405 --------------------------------
    406 
    407 Collects all volume information.
    408 
    409 The triplet identity is only available on the volumes after 
    410 GInstancer does the recursive labelling. So volume collection
    411 is now done by GInstancer::collectNodes_r rather than the former 
    412 X4PhysicalVolume::convertStructure.
    413 
    414 **/
    415 
    416 void GNodeLib::addVolume(const GVolume* volume)
    417 {   
    418     unsigned index = volume->getIndex();
    419     m_volumes.push_back(volume); 
    420     assert( m_volumes.size() - 1 == index && "indices of the geometry volumes added to GNodeLib must follow the sequence : 0,1,2,... " ); // formerly only for m_test
    421     m_volumemap[index] = volume ;
    422     
    423     glm::mat4 transform = volume->getTransformMat4();
    424     m_transforms->add(transform);
    425     
    426     glm::mat4 inverse_transform = volume->getInverseTransformMat4();
    427     m_inverse_transforms->add(inverse_transform);
    428 
    429     
    430     nbbox* bb = volume->getVerticesBBox();
    431     glm::vec4 min(bb->min, 1.f);
    432     glm::vec4 max(bb->max, 1.f); 
    433     m_bounding_box->add( min, max);
    434     
    435     glm::vec4 ce = bb->ce(); 
    436     m_center_extent->add(ce);
    437     
    438     m_lvlist->add(volume->getLVName());
    439     m_pvlist->add(volume->getPVName()); 
    440     // NB added in tandem, so same counts and same index as the volumes  
    441     
    442     glm::uvec4 id = volume->getIdentity();
    443     m_identity->add(id);
    444     
    445     glm::uvec4 ni = volume->getNodeInfo();
    446     m_nodeinfo->add(ni);
    447     
    448     const GVolume* check = getVolume(index);
    449     assert(check == volume);


    764 /**
    765 GInstancer::collectNodes
    766 ------------------------
    767 
    768 Populates GNodeLib. Invoked from GInstancer::createInstancedMergedMeshes immediately 
    769 after tree labelling and merged mesh creation.  
    770 The node collection needs to be after this labelling to capture the triplet identity. 
    771 
    772 **/
    773 
    774 void GInstancer::collectNodes()
    775 {
    776     assert(m_root);
    777     collectNodes_r(m_root, 0);
    778 }
    779 void GInstancer::collectNodes_r(const GNode* node, unsigned depth )
    780 {
    781     const GVolume* volume = dynamic_cast<const GVolume*>(node);
    782     m_nodelib->addVolume(volume);
    783     for(unsigned i = 0; i < node->getNumChildren(); i++) collectNodes_r(node->getChild(i), depth + 1 );
    784 }






identity info to allow getting local frame coords
--------------------------------------------------- 

::

    032 RT_PROGRAM void closest_hit_propagate()
     33 {
     34      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     35      float cos_theta = dot(n,ray.direction);
     36 
     37      prd.distance_to_boundary = t ;   // standard semantic attrib for this not available in raygen, so must pass it
     38 
     39      unsigned boundaryIndex = ( instanceIdentity.z & 0xffff ) ;
     40      prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;
     41      prd.identity = instanceIdentity ;
     42      prd.surface_normal = cos_theta > 0.f ? -n : n ;
     43 }

generate.cu::

    788         // use boundary index at intersection point to do optical constant + material/surface property lookups 
    789         fill_state(s, prd.boundary, prd.identity, p.wavelength );
    790 

state.h::

     70 
     71     s.identity = identity ;
     72 
     73 }       



::

    217 /**2
    218 FLAGS Macro 
    219 ------------
    220 
    221 Sets the photon flags p.flags using values from state s and per-ray-data prd
    222 
    223 p.flags.u.x 
    224    packed signed int boundary and unsigned sensorIndex which are 
    225    assumed to fit in 16 bits into 32 bits, see SPack::unsigned_as_int 
    226 
    227 p.flags.u.y
    228    now getting s.identity.x (nodeIndex) thanks to the packing 
    229 
    230 s.identity.x
    231     node index 
    232 
    233 s.identity.w 
    234     sensor index arriving from GVolume::getIdentity.w
    235 
    236 ::
    237 
    238     256 glm::uvec4 GVolume::getIdentity() const
    239     257 {
    240     258     glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ;
    241     259     return id ;
    242     260 }
    243 
    244 NumPy array access::
    245 
    246     boundary    = (( flags[:,0].view(np.uint32) & 0xffff0000 ) >> 16 ).view(np.int16)[1::2] 
    247     sensorIndex = (( flags[:,0].view(np.uint32) & 0x0000ffff ) >>  0 ).view(np.int16)[0::2] 
    248 
    249 
    250 Formerly::
    251 
    252     p.flags.i.x = prd.boundary ;  \
    253     p.flags.u.y = s.identity.w ;  \
    254     p.flags.u.w |= s.flag ; \
    255 
    256 2**/
    257 
    258 #define FLAGS(p, s, prd) \
    259 { \
    260     p.flags.u.x = ( ((prd.boundary & 0xffff) << 16) | (s.identity.w & 0xffff) )  ;  \
    261     p.flags.u.y = s.identity.x ;  \
    262     p.flags.u.w |= s.flag ; \
    263 } \
    264 


::

    epsilon:GNodeLib blyth$ ipython 

    In [1]: t = np.load("all_volume_transforms.npy")

    In [2]: t.shape
    Out[2]: (283812, 4, 4)



The G4 CRecorder emulation lacks the node index::

    In [9]: b.ox[:,3,1].view(np.uint32)
    Out[9]: A([0, 0, 0, ..., 0, 0, 0], dtype=uint32)

::

    386 void CWriter::writePhoton_(const G4StepPoint* point, unsigned record_id  )
    387 {
    388     assert( m_photons_buffer );
    389     writeHistory_(record_id);
    390 
    391     const G4ThreeVector& pos = point->GetPosition();
    392     const G4ThreeVector& dir = point->GetMomentumDirection();
    393     const G4ThreeVector& pol = point->GetPolarization();
    394 
    395     G4double time = point->GetGlobalTime();
    396     G4double energy = point->GetKineticEnergy();
    397     G4double wavelength = h_Planck*c_light/energy ;
    398     G4double weight = 1.0 ;
    399 
    400     // emulating the Opticks GPU written photons 
    401     m_photons_buffer->setQuad(record_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    402     m_photons_buffer->setQuad(record_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
    403     m_photons_buffer->setQuad(record_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );
    404 
    405     
    406     unsigned mskhis = m_photon._mskhis ; // narrowing from "unsigned long long" but 32-bits is enough   
    407     unsigned pflags = mskhis | m_ctx._hitflags ;
    408     
    409     
    410     // TODO: these are nothing like the OK flags  
    411     m_photons_buffer->setUInt(record_id, 3, 0, 0, m_photon._slot_constrained );
    412     m_photons_buffer->setUInt(record_id, 3, 0, 1, 0u );
    413     m_photons_buffer->setUInt(record_id, 3, 0, 2, m_photon._c4.u );
    414     m_photons_buffer->setUInt(record_id, 3, 0, 3, pflags );
    415 }


How to get the node index in G4 ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

jsd::

    0372     // == volume name
     373     std::string volname = track->GetVolume()->GetName(); // physical volume
     374     // == position

g4-cls G4Track::

    148   // volume, material, touchable
    149    G4VPhysicalVolume* GetVolume() const;
    150    G4VPhysicalVolume* GetNextVolume() const;

    133 // volume
    134    inline G4VPhysicalVolume* G4Track::GetVolume() const
    135    { if ( fpTouchable ==0 ) return 0;
    136      return fpTouchable->GetVolume(); }
    137 
    138    inline G4VPhysicalVolume* G4Track::GetNextVolume() const
    139    {  if ( fpNextTouchable ==0 ) return 0;
    140      return fpNextTouchable->GetVolume(); }
    141 

    159 // touchable
    160    inline const G4VTouchable* G4Track::GetTouchable() const
    161    { return fpTouchable(); }
    162 
    163    inline const G4TouchableHandle& G4Track::GetTouchableHandle() const
    164    { return fpTouchable; }
    165 
    166    inline void G4Track::SetTouchableHandle( const G4TouchableHandle& apValue)
    167    { fpTouchable = apValue; }
    168 
    169    inline const  G4VTouchable* G4Track::GetNextTouchable() const
    170    { return fpNextTouchable(); }
    171 
    172    inline const  G4TouchableHandle& G4Track::GetNextTouchableHandle() const
    173    { return fpNextTouchable; }
    174 
    175    inline void G4Track::SetNextTouchableHandle( const G4TouchableHandle& apValue)
    176    { fpNextTouchable = apValue; }
    177 
    178    inline const  G4VTouchable* G4Track::GetOriginTouchable() const
    179    { return fpOriginTouchable(); }
    180 
    181    inline const  G4TouchableHandle& G4Track::GetOriginTouchableHandle() const
    182    { return fpOriginTouchable; }
    183 
    184    inline void G4Track::SetOriginTouchableHandle( const G4TouchableHandle& apValue)
    185    { fpOriginTouchable = apValue; }

::

    epsilon:ggeo blyth$ g4-cc SetTouchable

    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/tracking/src/G4SteppingManager.cc:   fTrack->SetTouchableHandle(fTrack->GetNextTouchableHandle());
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/tracking/src/G4SteppingManager.cc:     fTrack->SetTouchableHandle( fTouchableHandle );
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/tracking/src/G4SteppingManager.cc:        fTrack->SetTouchableHandle( fTouchableHandle );
    epsilon:ggeo blyth$ 

    116 G4StepStatus G4SteppingManager::Stepping()
    117 //////////////////////////////////////////
    118 {
    ...
    134 // Store last PostStepPoint to PreStepPoint, and swap current and nex
    135 // volume information of G4Track. Reset total energy deposit in one Step. 
    136    fStep->CopyPostToPreStepPoint();
    137    fStep->ResetTotalEnergyDeposit();
    138 
    139 // Switch next touchable in track to current one
    140    fTrack->SetTouchableHandle(fTrack->GetNextTouchableHandle());
    ...
    147 //JA Set the volume before it is used (in DefineStepLength() for User Limit) 
    148    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
    149 
    150 // Reset the step's auxiliary points vector pointer
    151    fStep->SetPointerToVectorOfAuxiliaryPoints(0);
    152 
    ...
    230 // Send G4Step information to Hit/Dig if the volume is sensitive
    231    fCurrentVolume = fStep->GetPreStepPoint()->GetPhysicalVolume();
    232    StepControlFlag =  fStep->GetControlFlag();
    233    if( fCurrentVolume != 0 && StepControlFlag != AvoidHitInvocation) {
    234       fSensitive = fStep->GetPreStepPoint()->
    235                                    GetSensitiveDetector();
    236       if( fSensitive != 0 ) {
    237         fSensitive->Hit(fStep);
    238       }
    239    }

    136 inline
    137  G4VPhysicalVolume* G4StepPoint::GetPhysicalVolume() const
    138  { return fpTouchable->GetVolume(); }
    139 
    140 inline
    141  const G4VTouchable* G4StepPoint::GetTouchable() const
    142  { return fpTouchable(); }
    143 
    144 inline
    145  const G4TouchableHandle& G4StepPoint::GetTouchableHandle() const
    146  { return fpTouchable; }
    147 
    148 inline
    149  void G4StepPoint::SetTouchableHandle(const G4TouchableHandle& apValue)
    150  { fpTouchable = apValue; }
    151 




ggeo::

    310 void GNodeLib::getNodeIndicesForPVNameStarting(std::vector<unsigned>& nidx, const char* pvname_start) const
    311 {
    312     if( pvname_start == NULL ) return ;
    313     m_pvlist->getIndicesWithKeyStarting(nidx, pvname_start);
    314 }




very different number of unique nidx between G4 and OK ?
------------------------------------------------------------

::

    In [3]: len(np.unique(b.ox[:,3,1].view(np.uint32)))
    Out[3]: 298

    In [4]: len(np.unique(a.ox[:,3,1].view(np.uint32)))
    Out[4]: 5208


Huh after using getting nidx in CCtx::postTrack see less uniques::

    In [3]: np.unique(b.ox[:,3,1].view(np.uint32)).shape
    Out[3]: (101,)

::

    In [11]: als
    Out[11]: 
    SI BT BT BT SD
    SI RE AB
    SI BT BT BT SD
    SI BT BT SA
    SI RE BT AB
    SI RE RE SC AB
    SI RE AB
    SI SC SC RE BT BT BT SD
    SI SC SC AB
    SI AB

    In [12]: for i in range(10): print(gg.pv[an[i]])
    HamamatsuR12860_inner1_phys0x3aa0c00
    pTarget0x3358bb0
    NNVTMCPPMT_inner1_phys0x3a933a0
    pInnerWater0x3358a70
    pAcrylic0x3358b10
    pTarget0x3358bb0
    pTarget0x3358bb0
    NNVTMCPPMT_inner1_phys0x3a933a0
    pTarget0x3358bb0
    pTarget0x3358bb0

    In [13]: bls
    Out[13]: 
    SI RE BT BT BT SD
    SI BT BT BT SD
    SI SC SC SC SC SC BT BT BT SA
    SI RE AB
    SI AB
    SI SC BT BT BT SD
    SI BT BT BT SA
    SI SC BT BT SA
    SI BT BT BT SD
    SI AB

    In [14]: for i in range(10): print(gg.pv[bn[i]])
    HamamatsuR12860_inner1_phys0x3aa0c00
    NNVTMCPPMT_inner1_phys0x3a933a0
    lFasteners_phys0x33d0700
    pTarget0x3358bb0
    pTarget0x3358bb0
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    pCentralDetector0x3358c60
    NNVTMCPPMT_inner1_phys0x3a933a0
    pTarget0x3358bb0




nidx within the G4 4BT selection shows only 3 uniques 
---------------------------------------------------------

::


    In [6]: b.sel = "SI BT BT BT BT SD"

    In [7]: b.ox.shape
    Out[7]: (151, 4, 4)

    In [8]: b.ox[:,3,1].view(np.uint32)
    Out[8]: 
    A([141402, 141406, 141406, 141402, 269405, 269405, 141402, 141402, 141406, 141406, 141406, 141406, 141406, 141406, 141406, 141406, 141402, 141406, 141406, 141406, 141402, 141402, 141402, 269405,
       141406, 269405, 269405, 269405, 269405, 269405, 141402, 269405, 141406, 141406, 269405, 141406, 141406, 141402, 141406, 141406, 269405, 141406, 141406, 141406, 141406, 269405, 141402, 141402,
       141406, 269405, 269405, 141406, 141406, 269405, 141402, 269405, 141406, 141402, 269405, 269405, 269405, 269405, 269405, 141406, 269405, 141406, 141406, 141406, 269405, 141402, 141406, 141402,
       141406, 141406, 269405, 141406, 141406, 269405, 141406, 141402, 141402, 141406, 269405, 141402, 269405, 141406, 141402, 141406, 269405, 269405, 269405, 141406, 141406, 141406, 141406, 269405,
       269405, 269405, 141406, 141406, 141406, 269405, 269405, 141406, 141406, 141406, 141402, 141406, 141406, 269405, 269405, 269405, 141406, 141402, 141406, 141406, 141406, 141406, 269405, 141406,
       269405, 141406, 141406, 141402, 141402, 269405, 141406, 141402, 141406, 141406, 141402, 141402, 141406, 141402, 141406, 269405, 141406, 141402, 141406, 269405, 141402, 141406, 269405, 141406,
       141406, 141406, 141406, 141406, 269405, 141406, 141406], dtype=uint32)


    In [9]: nb = b.ox[:,3,1].view(np.uint32)

    In [10]: np.unique(nb)
    Out[10]: A([141402, 141406, 269405], dtype=uint32)


    In [15]: gg.pv[141402]
    Out[15]: 'HamamatsuR12860_inner1_phys0x3aa0c00'

    In [16]: gg.pv[141406]
    Out[16]: 'NNVTMCPPMT_inner1_phys0x3a933a0'

    In [18]: gg.pv[269405]
    Out[18]: 'PMT_3inch_inner1_phys0x421eca0'



Change to setting nidx in CCtx::postTrack::

    In [11]: np.unique(nb)
    Out[11]: A([141401, 141405, 269404], dtype=uint32)

    In [12]: gg.pv[141401]
    Out[12]: 'HamamatsuR12860_body_phys0x3aa0b80'

    In [13]: gg.pv[141405]
    Out[13]: 'NNVTMCPPMT_body_phys0x3a93320'

    In [14]: gg.pv[269404]
    Out[14]: 'PMT_3inch_body_phys0x421ec20'


They are well spread in position::

    In [21]: pos[:,0].min()
    Out[21]: A(-19343.75, dtype=float32)

    In [22]: pos[:,0].max()
    Out[22]: A(18575.697, dtype=float32)

    In [23]: pos[:,1].min()
    Out[23]: A(-19078.795, dtype=float32)

    In [24]: pos[:,1].max()
    Out[24]: A(19100.252, dtype=float32)

    In [25]: pos[:,2].min()
    Out[25]: A(-19166.809, dtype=float32)

    In [26]: pos[:,2].max()
    Out[26]: A(19353.016, dtype=float32)



    In [29]: np.sqrt(np.sum(pos*pos, axis=1)).min()
    Out[29]: A(19250.707, dtype=float32)

    In [30]: np.sqrt(np.sum(pos*pos, axis=1)).max()
    Out[30]: A(19435.08, dtype=float32)






OK 3BT
---------

::


    In [32]: a.sel = "SI BT BT BT SD"

    In [33]: a.ox.shape
    Out[33]: (1367, 4, 4)


    In [34]: pos = a.ox[:, 0, :3]

    In [35]: pos
    Out[35]: 
    A([[-17866.793,   7413.646,    244.02 ],
       [    41.153,  14882.913, -12431.25 ],
       [-15591.934,  -8364.082,  -7656.699],
       ...,
       [-16284.078,  -2305.71 , -10127.313],
       [-14134.6  ,  13035.886,   1194.539],
       [ -6101.833,  17138.096,   6436.572]], dtype=float32)

    In [36]: np.sqrt(np.sum(pos*pos, axis=1))
    Out[36]: A([19345.387, 19391.719, 19279.299, ..., 19314.502, 19265.205, 19297.049], dtype=float32)

    In [37]: np.sqrt(np.sum(pos*pos, axis=1)).min()
    Out[37]: A(19232.432, dtype=float32)

    In [38]: np.sqrt(np.sum(pos*pos, axis=1)).max()
    Out[38]: A(19435.096, dtype=float32)

    In [39]: pos[:,0].min(), pos[:,0].max(), pos[:,1].min(), pos[:,1].max(), pos[:,2].min(), pos[:,2].max()
    Out[39]: 
    (A(-19231.441, dtype=float32),
     A(19302.283, dtype=float32),
     A(-19332.838, dtype=float32),
     A(19350.223, dtype=float32),
     A(-19277.975, dtype=float32),
     A(19346.82, dtype=float32))

    In [41]: an
    Out[41]: A([106122, 129818, 120550, ..., 125134, 104414,  94170], dtype=uint32)

    In [42]: an.shape
    Out[42]: (1367,)

    In [45]: np.unique(an).shape
    Out[45]: (1316,)



    In [47]: for i in range(100): print(gg.pv[an[i]])
    HamamatsuR12860_inner1_phys0x3aa0c00
    NNVTMCPPMT_inner1_phys0x3a933a0
    HamamatsuR12860_inner1_phys0x3aa0c00
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    HamamatsuR12860_inner1_phys0x3aa0c00
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    HamamatsuR12860_inner1_phys0x3aa0c00
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    HamamatsuR12860_inner1_phys0x3aa0c00
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0
    NNVTMCPPMT_inner1_phys0x3a933a0




Hmm the writer is writing from collected points, I suspect the 
collection G4StepPoint::GetPhysicalVolume is relying on something 
outside the point that aint persisted. Causing the lack of unique nidx::

    389 void CWriter::writePhoton_(const G4StepPoint* point, unsigned record_id  )
    390 {   
    391     assert( m_photons_buffer );
    392     writeHistory_(record_id);
    393     
    394     const G4ThreeVector& pos = point->GetPosition();
    395     const G4ThreeVector& dir = point->GetMomentumDirection();
    396     const G4ThreeVector& pol = point->GetPolarization();
    397     
    398     G4double time = point->GetGlobalTime();
    399     G4double energy = point->GetKineticEnergy();
    400     G4double wavelength = h_Planck*c_light/energy ;
    401     G4double weight = 1.0 ;
    402     
    403     // emulating the Opticks GPU written photons 
    404     m_photons_buffer->setQuad(record_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    405     m_photons_buffer->setQuad(record_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
    406     m_photons_buffer->setQuad(record_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );
    407 
    408     
    409     unsigned mskhis = m_photon._mskhis ; // narrowing from "unsigned long long" but 32-bits is enough   
    410     unsigned pflags = mskhis | m_ctx._hitflags ;
    411     
    412     const G4VPhysicalVolume* pv = point->GetPhysicalVolume() ;
    413     const void* origin = (void*)pv ; 
    414     int nidx = GGeo::Get()->findNodeIndex(origin);
    415     



Trying getting the nidx at postTrack then there is no persisting involved its direct from G4Track::

    388 void CCtx::postTrack()
    389 {
    390     const G4VPhysicalVolume* pv = _track->GetVolume() ;
    391     const void* origin = (void*)pv ;
    392     _nidx = GGeo::Get()->findNodeIndex(origin);
    393 }
    394 

Then the writer can just grab from ctx::

    411     unsigned nidx = m_ctx._nidx > -1 ? unsigned(m_ctx._nidx) : ~0u ;
    412     
    413     m_photons_buffer->setUInt(record_id, 3, 0, 0, m_photon._slot_constrained );
    414     m_photons_buffer->setUInt(record_id, 3, 0, 1, nidx );
    415     m_photons_buffer->setUInt(record_id, 3, 0, 2, m_photon._c4.u );
    416     m_photons_buffer->setUInt(record_id, 3, 0, 3, pflags );
    417     
    418     // TODO: make these match OK flags better 
    419 }







Why not ~300k origin node ? Only 51k ? NEED TO USE (pv,copyNo) as the key ?
--------------------------------------------------------------------------------

::

    2021-06-25 19:15:26.568 INFO  [170343] [GGeo::prepareVolumes@1346] GNodeLib::descOriginMap m_origin2index.size 51017


::

    1409 GVolume* X4PhysicalVolume::convertNode(const G4VPhysicalVolume* const pv, GVolume* parent, int depth, const G4VPhysicalVolume* const pv_p, bool& recursive_select )
    1410 {
    1411 #ifdef X4_PROFILE
    1412     float t00 = BTimeStamp::RealTime();
    1413 #endif
    1414 
    1415     // record copynumber in GVolume, as thats one way to handle pmtid
    1416     const G4PVPlacement* placement = dynamic_cast<const G4PVPlacement*>(pv);
    1417     assert(placement);
    1418     G4int copyNumber = placement->GetCopyNo() ;
    ...
    1564     G4PVPlacement* _placement = const_cast<G4PVPlacement*>(placement) ;
    1565     void* origin_node = static_cast<void*>(_placement) ;
    1566     GVolume* volume = new GVolume(ndIdx, gtransform, mesh, origin_node );


g4-cls G4PVPlacement::

     51 class G4PVPlacement : public G4VPhysicalVolume

g4-cls G4VPhysicalVolume::

    82 class G4VPhysicalVolume







