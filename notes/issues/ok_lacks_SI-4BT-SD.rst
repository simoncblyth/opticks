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



After increase microstep suppression cut to 0.004, around 50 G4 4BT -> 3BT  
------------------------------------------------------------------------------

* 3BT/4BT difference remains, but its less significant 

::

    In [3]: ab.his[:40]                                                                                                                                                                             
    Out[3]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11142     11142       224.22/63 =  3.56  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               42      1666      1621     45              0.62         1.028 +- 0.025        0.973 +- 0.024  [2 ] SI AB
    0001            7ccc2      1367      1311     56              1.17         1.043 +- 0.028        0.959 +- 0.026  [5 ] SI BT BT BT SD

    0002            8ccc2       624       793   -169             20.16         0.787 +- 0.032        1.271 +- 0.045  [5 ] SI BT BT BT SA      ###

    0003           7ccc62       680       563    117             11.01         1.208 +- 0.046        0.828 +- 0.035  [6 ] SI SC BT BT BT SD
    0004             8cc2       570       496     74              5.14         1.149 +- 0.048        0.870 +- 0.039  [4 ] SI BT BT SA
    0005              452       408       495    -87              8.38         0.824 +- 0.041        1.213 +- 0.055  [3 ] SI RE AB
    0006           7ccc52       405       399      6              0.04         1.015 +- 0.050        0.985 +- 0.049  [6 ] SI RE BT BT BT SD
    0007              462       399       351     48              3.07         1.137 +- 0.057        0.880 +- 0.047  [3 ] SI SC AB
    0008           8ccc62       270       270      0              0.00         1.000 +- 0.061        1.000 +- 0.061  [6 ] SI SC BT BT BT SA
    0009          7ccc662       255       200     55              6.65         1.275 +- 0.080        0.784 +- 0.055  [7 ] SI SC SC BT BT BT SD
    0010            8cc62       195       180     15              0.60         1.083 +- 0.078        0.923 +- 0.069  [5 ] SI SC BT BT SA
    0011           8ccc52       176       197    -21              1.18         0.893 +- 0.067        1.119 +- 0.080  [6 ] SI RE BT BT BT SA
    0012          7ccc652       186       170     16              0.72         1.094 +- 0.080        0.914 +- 0.070  [7 ] SI RE SC BT BT BT SD
    0013               41       156       160     -4              0.05         0.975 +- 0.078        1.026 +- 0.081  [2 ] CK AB
    0014             4552       118       152    -34              4.28         0.776 +- 0.071        1.288 +- 0.104  [4 ] SI RE RE AB
    0015            8cc52       136       133      3              0.03         1.023 +- 0.088        0.978 +- 0.085  [5 ] SI RE BT BT SA
    0016          7ccc552       123       124     -1              0.00         0.992 +- 0.089        1.008 +- 0.091  [7 ] SI RE RE BT BT BT SD
    0017             4662       125       114     11              0.51         1.096 +- 0.098        0.912 +- 0.085  [4 ] SI SC SC AB
    0018             4652       118       108     10              0.44         1.093 +- 0.101        0.915 +- 0.088  [4 ] SI RE SC AB
    0019             4cc2       121       104     17              1.28         1.163 +- 0.106        0.860 +- 0.084  [4 ] SI BT BT AB
    0020          8ccc662        78       101    -23              2.96         0.772 +- 0.087        1.295 +- 0.129  [7 ] SI SC SC BT BT BT SA
    0021         7ccc6662        76        85     -9              0.50         0.894 +- 0.103        1.118 +- 0.121  [8 ] SI SC SC SC BT BT BT SD

    0022           7cccc2        50       101    -51             17.23         0.495 +- 0.070        2.020 +- 0.201  [6 ] SI BT BT BT BT SD           ####

    0023          8ccc652        72        75     -3              0.06         0.960 +- 0.113        1.042 +- 0.120  [7 ] SI RE SC BT BT BT SA
    0024          8ccc552        62        72    -10              0.75         0.861 +- 0.109        1.161 +- 0.137  [7 ] SI RE RE BT BT BT SA
    0025         7ccc6652        57        69    -12              1.14         0.826 +- 0.109        1.211 +- 0.146  [8 ] SI RE SC SC BT BT BT SD
    0026           8cc662        57        65     -8              0.52         0.877 +- 0.116        1.140 +- 0.141  [6 ] SI SC SC BT BT SA
    0027         7ccc6552        50        58     -8              0.59         0.862 +- 0.122        1.160 +- 0.152  [8 ] SI RE RE SC BT BT BT SD
    0028             4562        51        57     -6              0.33         0.895 +- 0.125        1.118 +- 0.148  [4 ] SI SC RE AB
    0029          7ccc562        68        40     28              7.26         1.700 +- 0.206        0.588 +- 0.093  [7 ] SI SC RE BT BT BT SD
    0030              4c2        58        46     12              1.38         1.261 +- 0.166        0.793 +- 0.117  [3 ] SI BT AB
    0031           8cc652        48        54     -6              0.35         0.889 +- 0.128        1.125 +- 0.153  [6 ] SI RE SC BT BT SA
    0032            4cc62        50        40     10              1.11         1.250 +- 0.177        0.800 +- 0.126  [5 ] SI SC BT BT AB
    0033            46662        40        43     -3              0.11         0.930 +- 0.147        1.075 +- 0.164  [5 ] SI SC SC SC AB
    0034           8cc552        30        50    -20              5.00         0.600 +- 0.110        1.667 +- 0.236  [6 ] SI RE RE BT BT SA
    0035          7cccc62        19        58    -39             19.75         0.328 +- 0.075        3.053 +- 0.401  [7 ] SI SC BT BT BT BT SD
    0036         7ccc5552        33        42     -9              1.08         0.786 +- 0.137        1.273 +- 0.196  [8 ] SI RE RE RE BT BT BT SD

    0037            4ccc2        56        18     38             19.51         3.111 +- 0.416        0.321 +- 0.076  [5 ] SI BT BT BT AB

    0038            45552        30        43    -13              2.32         0.698 +- 0.127        1.433 +- 0.219  [5 ] SI RE RE RE AB
    .                              11142     11142       224.22/63 =  3.56  (pval:0.000 prob:1.000)  







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


Single inheritance so casting should not change the pointer value ?


g4-cls G4PVPlacement::

     51 class G4PVPlacement : public G4VPhysicalVolume

g4-cls G4VPhysicalVolume::

    82 class G4VPhysicalVolume


* https://stackoverflow.com/questions/56706847/when-does-a-cast-return-a-different-pointer-in-case-of-single-inheritance





Geant4 makes it difficult to establish node/volume identity 
----------------------------------------------------------------

With Opticks you just read off the nidx label present on every volume, 
but not with Geant4.

* a pv pointer by itself does not identify a node 
* its seems even with copyNo it not enough
* the placement of that pv/copyNo in the full heirarcy of lv and pv/copyNo 
  is by definition unique, so creating a digest that traverses up the tree
  and includes copyNo should work 

  * but would it be usable from what G4Track provides


g4-cls G4Track::

    158    const G4VTouchable*      GetTouchable() const;
    159    const G4TouchableHandle& GetTouchableHandle() const;
    160    void SetTouchableHandle( const G4TouchableHandle& apValue);
    161 
    162    const G4VTouchable*      GetNextTouchable() const;
    163    const G4TouchableHandle& GetNextTouchableHandle() const;
    164    void SetNextTouchableHandle( const G4TouchableHandle& apValue);
    165 
    166    const G4VTouchable*      GetOriginTouchable() const;
    167    const G4TouchableHandle& GetOriginTouchableHandle() const;
    168    void SetOriginTouchableHandle( const G4TouchableHandle& apValue);
    169 

g4-cls G4PhysicalVolumeModel
g4-cls G4TouchableHistory



G4 nidx would be useful, but its not essential for debugging the current 3BT 4BT issue
------------------------------------------------------------------------------------------

::

    In [7]: b.sel = "SI BT BT BT BT SD"

    In [8]: b.dx.shape
    Out[8]: (151, 10, 2, 4)

    In [10]: b.dx[0]
    Out[10]: 
    A([[[   63.8113,   -79.8528,  -230.8417,     7.6382],          SI 
        [   -0.9492,     0.3023,    -0.0875,   479.4415]],

       [[-1803.5128,  -782.7069, 17590.4722,    99.2858],          BT   in Ac
        [   -0.411 ,    -0.9082,    -0.0789,   479.4415]],

       [[-1816.0073,  -787.4131, 17709.7289,    99.8992],          BT   Ac->Wa
        [   -0.411 ,    -0.9082,    -0.0789,   479.4415]],

       [[-1824.9983,  -790.7997, 17795.5455,   100.3407],          BT   Wa->Py
        [   -0.4165,    -0.9089,    -0.0191,   479.4415]],

       [[-1927.5772,  -773.954 , 19227.2446,   106.9218],
        [   -0.3278,    -0.9446,    -0.0174,   479.4415]],

       [[-1927.3906,  -774.1523, 19234.4925,   106.9584],
        [   -0.3278,    -0.9446,    -0.0174,   479.4415]],

       [[    0.    ,     0.    ,     0.    ,     0.    ],
        [    0.    ,     0.    ,     0.    ,     0.    ]],

       [[    0.    ,     0.    ,     0.    ,     0.    ],
        [    0.    ,     0.    ,     0.    ,     0.    ]],

       [[    0.    ,     0.    ,     0.    ,     0.    ],
        [    0.    ,     0.    ,     0.    ,     0.    ]],

       [[    0.    ,     0.    ,     0.    ,     0.    ],
        [    0.    ,     0.    ,     0.    ,     0.    ]]])


::

    In [26]: for j in range(5): print(" i%2d : j%2d : %10.4f %10.4f %10.4f %10.4f  " % tuple([i,j] + list(map(float,b.dx[i,j+1,0] - b.dx[i,j,0])))) 
     i 0 : j 0 : -1867.3241  -702.8541 17821.3139    91.6476  
     i 0 : j 1 :   -12.4945    -4.7062   119.2567     0.6134      Crossing acrylic
     i 0 : j 2 :    -8.9910    -3.3866    85.8166     0.4414      Whats this ? 
     i 0 : j 3 :  -102.5789    16.8457  1431.6991     6.5811      Crossing water 
     i 0 : j 4 :     0.1865    -0.1983     7.2480     0.0366       

    In [27]: 





Look at OK boundaries for the 4BT, reveals -29:Water///Water prior to landing on 3inch PMTs
----------------------------------------------------------------------------------------------------

* hmm it would be nice to have boundary histories with G4 , but do have seqmat 

::

    In [32]: a.sel = "SI BT BT BT BT SD"

    In [35]: a.rpost()
    Out[35]: 
    A([[[    64.0889,    -80.5689,   -230.7199,      1.2818],
        [ 15465.5599,  -2662.4348,   8188.7265,     91.9584],
        [ 15569.9332,  -2678.9148,   8245.4909,     92.581 ],
        [ 16895.6572,  -2900.4791,   8972.4418,     99.6124],
        [ 16908.475 ,  -2902.3103,   8979.7662,     99.6857],
        [ 16908.475 ,  -2902.3103,   8979.7662,     99.6857]],

       [[    64.0889,    -80.5689,   -230.7199,     20.3619],
        [-13777.2759,  -9999.6948,   4845.1186,    111.4048],
        [-13872.4937, -10065.6148,   4879.9097,    112.0273],
        [-15064.5466, -10918.9123,   5319.3762,    119.0588],
        [-15075.5333, -10926.2368,   5323.0384,    119.0954],
        [-15077.3644, -10928.0679,   5323.0384,    119.1321]],

       [[    65.92  ,    -69.5822,   -217.9022,      8.3132],
        [ 12065.1875, -12942.2895,    450.4532,     98.8433],
        [ 12147.5875, -13030.1828,    454.1154,     99.4659],
        [ 13196.8139, -14158.1469,    514.5421,    106.534 ],
        [ 13204.1383, -14163.6402,    514.5421,    106.5706],
        [ 13205.9694, -14165.4714,    514.5421,    106.6073]],

       ...,


Boundaries all the same::

    In [39]: a.bn.view(np.int8).reshape(-1,16)
    Out[39]: 
    A([[ 18,  17, -29, -23, -30,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -29, -23, -30,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -29, -23, -30,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -29, -23, -30,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
       [ 18,  17, -29, -23, -30,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],

::

    In [41]: print(a.blib.format( a.bn.view(np.int8).reshape(-1,16)[0] ))
     18 : Acrylic///LS
     17 : Water///Acrylic
    -29 : Water///Water
    -23 : Water///Pyrex
    -30 : Pyrex/PMT_3inch_photocathode_logsurf2/PMT_3inch_photocathode_logsurf1/Vacuum


::

    In [42]: a.seqmat_ana.table
    Out[42]: 
    seqmat_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                                 50         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000           deffb1        1.000          50        [6 ] LS Ac Wa Wa Py Va
       n             iseq         frac           a    a-b      [ns] label
    .                                 50         1.00 


    In [43]: b.seqmat_ana.table
    Out[43]: 
    seqmat_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                                151         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000           defbb1        0.351          53        [6 ] LS Ac Ac Wa Py Va
    0001           deefb1        0.351          53        [6 ] LS Ac Wa Py Py Va
    0002           deffb1        0.298          45        [6 ] LS Ac Wa Wa Py Va
       n             iseq         frac           a    a-b      [ns] label
    .                                151         1.00 

    In [44]: b.sel
    Out[44]: 'SI BT BT BT BT SD'

    In [45]: a.sel
    Out[45]: 'SI BT BT BT BT SD'


What causes G4: double Ac ? inconclusive : not microsteps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In [46]: b.selmat = "LS Ac Ac Wa Py Va"

    In [47]: b.seqhis
    Out[47]: A([     8178770,       511170, 604724684386, ...,    147613010,         1122,        36034], dtype=uint64)

    In [48]: b.seqhis_ana.table
    Out[48]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                                 78         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000           7cccc2        0.679          53        [6 ] SI BT BT BT BT SD
    0001           8cccc2        0.321          25        [6 ] SI BT BT BT BT SA
       n             iseq         frac           a    a-b      [ns] label
    .                                 78         1.00 

    In [49]: 



These are not microsteps::

    In [56]: i = 0 

    In [57]: for j in range(5): print(" i%2d : j%2d : %10.4f %10.4f %10.4f %10.4f  " % tuple([i,j] + list(map(float,b.dx[i,j+1,0] - b.dx[i,j,0])))) 
     i 0 : j 0 : -1867.3241  -702.8541 17821.3139    91.6476  
     i 0 : j 1 :   -12.4945    -4.7062   119.2567     0.6134  
     i 0 : j 2 :    -8.9910    -3.3866    85.8166     0.4414  
     i 0 : j 3 :  -102.5789    16.8457  1431.6991     6.5811  
     i 0 : j 4 :     0.1865    -0.1983     7.2480     0.0366  

    In [58]: i = 1

    In [59]: for j in range(5): print(" i%2d : j%2d : %10.4f %10.4f %10.4f %10.4f  " % tuple([i,j] + list(map(float,b.dx[i,j+1,0] - b.dx[i,j,0])))) 
     i 1 : j 0 : -10152.3894  5739.6408 -13166.0754    89.7826  
     i 1 : j 1 :   -69.2672    39.1588   -89.8386     0.6140  
     i 1 : j 2 :   -68.4883    38.7184   -88.8284     0.6071  
     i 1 : j 3 :  -719.3436   463.0135 -1061.3044     6.2678  
     i 1 : j 4 :    -3.4529     1.9782    -4.4161     0.0301  



What about G4 Py Py ? They all look to be microsteps that evaded the suppression cut at 0.002 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* increasing microstep suppression cut to 0.004 will migrate those "Py Py" to "Py", 4BT -> 3BT 


::

    In [2]: b.sel                                                                                                                                                                                   
    Out[2]: 'SI BT BT BT BT SD'

    In [3]: b.seqmat_ana.table                                                                                                                                                                      
    Out[3]: 
    seqmat_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                                151         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000           defbb1        0.351          53        [6 ] LS Ac Ac Wa Py Va   ## UNKNOWN CAUSE
    0001           deefb1        0.351          53        [6 ] LS Ac Wa Py Py Va     
    0002           deffb1        0.298          45        [6 ] LS Ac Wa Wa Py Va   ## 3inch Water///Water envelope ?  
       n             iseq         frac           a    a-b      [ns] label
    .                                151         1.00 


    In [4]: b.selmat = "LS Ac Wa Py Py Va"                                                                                                                                                          
    In [6]: b.seqhis_ana.table                                                                                                                                                                      
    Out[6]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                                 80         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000           7cccc2        0.662          53        [6 ] SI BT BT BT BT SD
    0001           8cccc2        0.338          27        [6 ] SI BT BT BT BT SA
       n             iseq         frac           a    a-b      [ns] label
    .                                 80         1.00 


    In [23]: for i in range(10): 
        ...:     for j in range(5): print(" i%2d : j%2d : %10.4f %10.4f %10.4f %10.4f  " % tuple([i,j] + list(map(float,b.dx[i,j+1,0] - b.dx[i,j,0])))) 
        ...:                                                                                                                                                                                        
     i 0 : j 0 : 16858.1159  -235.7075 -4949.7780    90.0824  
     i 0 : j 1 :   115.1360    -1.6122   -33.8120     0.6206  
     i 0 : j 2 :  1526.9241   -20.6078  -446.3348     7.3022  
     i 0 : j 3 :     0.0021    -0.0007    -0.0005     0.0000  
     i 0 : j 4 :    11.7091    -3.7085    -2.8257     0.0640  
     i 1 : j 0 :  -176.5541 -11884.9110 13270.3232    91.4891  
     i 1 : j 1 :    -1.1873   -80.0644    89.3881     0.6233  
     i 1 : j 2 :   -16.3651 -1059.0165  1185.2411     7.2901  
     i 1 : j 3 :    -0.0006    -0.0015     0.0016     0.0000  
     i 1 : j 4 :    -3.4381    -7.9642     8.7771     0.0628  
     i 2 : j 0 : -7631.3413 -10013.3366 -12208.0008    89.6300  
     i 2 : j 1 :   -52.2175   -68.5213   -83.5436     0.6134  
     i 2 : j 2 :  -697.5267  -913.7778 -1112.8038     7.3352  
     i 2 : j 3 :    -0.0003    -0.0017    -0.0016     0.0000  
     i 2 : j 4 :    -1.8518    -9.5419    -9.1234     0.0672  
     i 3 : j 0 :  4914.2048 -5987.1817 -15650.6330    89.2742  
     i 3 : j 1 :    33.7698   -41.1428  -107.5494     0.6162  
     i 3 : j 2 :   443.8378  -540.8515 -1413.4618     7.2474  
     i 3 : j 3 :     0.0007    -0.0011    -0.0016     0.0000  
     i 3 : j 4 :     3.9101    -5.8275    -8.7300     0.0568  
     i 4 : j 0 :  2770.0218 -11759.6495 -12639.9983    89.4047  
     i 4 : j 1 :    19.0118   -80.7049   -86.7510     0.6165  
     i 4 : j 2 :   251.4560 -1069.4992 -1148.2234     7.3111  
     i 4 : j 3 :     0.0001    -0.0019    -0.0012     0.0000  
     i 4 : j 4 :     0.2926   -10.3600    -6.7676     0.0627  
     i 5 : j 0 : -1495.4270 -16960.7678  4824.2390    90.3386  
     i 5 : j 1 :   -10.1391  -115.0213    32.7087     0.6138  
     i 5 : j 2 :  -133.9105 -1511.1629   432.0303     7.2762  
     i 5 : j 3 :    -0.0006    -0.0018     0.0007     0.0000  
     i 5 : j 4 :    -3.0705    -9.9140     3.9653     0.0562  
     i 6 : j 0 :  5560.8245 12877.5752 11060.9744    91.3700  
     i 6 : j 1 :    37.3613    86.5128    74.3035     0.6132  
     i 6 : j 2 :   497.0163  1153.0651   991.8883     7.4046  
     i 6 : j 3 :     0.0001     0.0020     0.0013     0.0000  
     i 6 : j 4 :     0.3035    11.1964     7.1022     0.0668  
     i 7 : j 0 :  5223.0564  5595.8546 16179.2256    92.3290  
     i 7 : j 1 :    35.0227    37.5177   108.4740     0.6271  
     i 7 : j 2 :   459.8151   493.9536  1428.3212     7.2454  
     i 7 : j 3 :     0.0002     0.0009     0.0019     0.0000  
     i 7 : j 4 :     1.0335     4.8504    10.1970     0.0579  
     i 8 : j 0 : -1465.4378   734.4137 17849.7344    91.5695  
     i 8 : j 1 :    -9.8092     4.9148   119.4985     0.6151  
     i 8 : j 2 :  -130.9457    65.9646  1589.4416     7.2958  
     i 8 : j 3 :    -0.0006     0.0007     0.0022     0.0000  
     i 8 : j 4 :    -3.1173     3.7515    11.9969     0.0656  
     i 9 : j 0 :  3547.6569 -15963.7957  6787.7508    90.8713  
     i 9 : j 1 :    24.0480  -108.2045    46.0005     0.6223  
     i 9 : j 2 :   320.3511 -1443.6873   616.1915     7.3481  
     i 9 : j 3 :    -0.0001    -0.0019     0.0014     0.0000  
     i 9 : j 4 :    -0.6177   -10.6526     7.7870     0.0672  

    In [24]:                                                          



All the G4 Py Py are microsteps with zero time difference::

    In [37]: b.dx[:,4,0]-b.dx[:,3,0]                                                                                                                                                                
    Out[37]: 
    A([[ 0.0021, -0.0007, -0.0005,  0.    ],
       [-0.0006, -0.0015,  0.0016,  0.    ],
       [-0.0003, -0.0017, -0.0016,  0.    ],
       [ 0.0007, -0.0011, -0.0016,  0.    ],
       [ 0.0001, -0.0019, -0.0012,  0.    ],
       ...
       [ 0.0006, -0.0019,  0.0005,  0.    ],
       [ 0.0019, -0.    , -0.0013,  0.    ],
       [-0.0001,  0.0001,  0.002 ,  0.    ],
       [ 0.0013, -0.0015, -0.001 ,  0.    ]])

    In [38]:                                 

::


    In [39]: mst = b.dx[:,4,0,:3]-b.dx[:,3,0,:3]                                                                                                                                                    

    In [40]: np.sqrt(np.sum(mst*mst, axis=1))                                                                                                                                                       
    Out[40]: 
    A([0.0023, 0.0023, 0.0024, 0.0021, 0.0023, 0.0021, 0.0024, 0.0021, 0.0023, 0.0024, 0.002 , 0.0023, 0.002 , 0.0024, 0.002 , 0.0023, 0.0022, 0.0023, 0.0022, 0.002 , 0.0021, 0.0022, 0.0024, 0.0021,
       0.0024, 0.0022, 0.002 , 0.002 , 0.0023, 0.0021, 0.0022, 0.0021, 0.002 , 0.0021, 0.0021, 0.0021, 0.002 , 0.0022, 0.0024, 0.0022, 0.0024, 0.0024, 0.0024, 0.0023, 0.002 , 0.002 , 0.002 , 0.0023,
       0.0021, 0.0021, 0.002 , 0.0022, 0.0021, 0.0024, 0.0021, 0.0022, 0.0022, 0.0021, 0.0022, 0.0022, 0.0022, 0.0021, 0.0022, 0.002 , 0.0023, 0.0021, 0.0022, 0.002 , 0.0021, 0.0021, 0.0021, 0.002 ,
       0.0023, 0.0023, 0.0024, 0.0023, 0.002 , 0.0023, 0.002 , 0.0022])

    In [41]:                                                                   

    In [42]: d_mst.min()                                                                                                                                                                            
    Out[42]: A(0.002)

    In [43]: d_mst.max()                                                                                                                                                                            
    Out[43]: A(0.0024)



seqmat zeros::

    In [5]: ab.mat[:40]                                                                                                                                                                             
    Out[5]: 
    ab.mat
    .       seqmat_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11142     11142       764.05/48 = 15.92  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000            defb1      1994      1881    113              3.30         1.060 +- 0.024        0.943 +- 0.022  [5 ] LS Ac Wa Py Va
    0001               11      1821      1781     40              0.44         1.022 +- 0.024        0.978 +- 0.023  [2 ] LS LS
    0002           defb11      1557      1312    245             20.92         1.187 +- 0.030        0.843 +- 0.023  [6 ] LS LS Ac Wa Py Va
    0003              111       832       863    -31              0.57         0.964 +- 0.033        1.037 +- 0.035  [3 ] LS LS LS
    0004          defb111       887       732    155             14.84         1.212 +- 0.041        0.825 +- 0.031  [7 ] LS LS LS Ac Wa Py Va
    0005             3fb1       571       481     90              7.70         1.187 +- 0.050        0.842 +- 0.038  [4 ] LS Ac Wa Ty
    0006             1111       423       442    -19              0.42         0.957 +- 0.047        1.045 +- 0.050  [4 ] LS LS LS LS
    0007         defb1111       403       413    -10              0.12         0.976 +- 0.049        1.025 +- 0.050  [8 ] LS LS LS LS Ac Wa Py Va
    0008            3fb11       340       300     40              2.50         1.133 +- 0.061        0.882 +- 0.051  [5 ] LS LS Ac Wa Ty
    0009            11111       208       217     -9              0.19         0.959 +- 0.066        1.043 +- 0.071  [5 ] LS LS LS LS LS
    0010        defb11111       209       167     42              4.69         1.251 +- 0.087        0.799 +- 0.062  [9 ] LS LS LS LS LS Ac Wa Py Va
    0011           3fb111       157       174    -17              0.87         0.902 +- 0.072        1.108 +- 0.084  [6 ] LS LS LS Ac Wa Ty
    0012             ffb1       121       100     21              2.00         1.210 +- 0.110        0.826 +- 0.083  [4 ] LS Ac Wa Wa
    0013           111111       107        88     19              1.85         1.216 +- 0.118        0.822 +- 0.088  [6 ] LS LS LS LS LS LS
    0014       defb111111        95        98     -3              0.05         0.969 +- 0.099        1.032 +- 0.104  [10] LS LS LS LS LS LS Ac Wa Py Va
    0015           deffb1        87        74     13              1.05         1.176 +- 0.126        0.851 +- 0.099  [6 ] LS Ac Wa Wa Py Va
    0016          3fb1111        81        79      2              0.03         1.025 +- 0.114        0.975 +- 0.110  [7 ] LS LS LS LS Ac Wa Ty
    0017            ffb11        83        65     18              2.19         1.277 +- 0.140        0.783 +- 0.097  [5 ] LS LS Ac Wa Wa
    0018          deffb11        62        56      6              0.31         1.107 +- 0.141        0.903 +- 0.121  [7 ] LS LS Ac Wa Wa Py Va

    0019            3fbb1         0       112   -112            112.00         0.000 +- 0.000        0.000 +- 0.000  [5 ] LS Ac Ac Wa Ty

    0020              bb1        58        46     12              1.38         1.261 +- 0.166        0.793 +- 0.117  [3 ] LS Ac Ac
    0021           ffb111        54        46      8              0.64         1.174 +- 0.160        0.852 +- 0.126  [6 ] LS LS LS Ac Wa Wa
    0022          1111111        41        43     -2              0.05         0.953 +- 0.149        1.049 +- 0.160  [7 ] LS LS LS LS LS LS LS
    0023             bb11        52        30     22              5.90         1.733 +- 0.240        0.577 +- 0.105  [4 ] LS LS Ac Ac

    0024           deefb1         0        80    -80             80.00         0.000 +- 0.000        0.000 +- 0.000  [6 ] LS Ac Wa Py Py Va            ###

    0025       efb1111111        41        38      3              0.11         1.079 +- 0.169        0.927 +- 0.150  [10] LS LS LS LS LS LS LS Ac Wa Py

    0026           defbb1         0        78    -78             78.00         0.000 +- 0.000        0.000 +- 0.000  [6 ] LS Ac Ac Wa Py Va

    0027          defbb11         1        75    -74             72.05         0.013 +- 0.013       75.000 +- 8.660  [7 ] LS LS Ac Ac Wa Py Va
    0028         3fb11111        42        34      8              0.84         1.235 +- 0.191        0.810 +- 0.139  [8 ] LS LS LS LS LS Ac Wa Ty
    0029         deffb111        36        30      6              0.55         1.200 +- 0.200        0.833 +- 0.152  [8 ] LS LS LS Ac Wa Wa Py Va
    0030       fb11111111        27        36     -9              1.29         0.750 +- 0.144        1.333 +- 0.222  [10] LS LS LS LS LS LS LS LS Ac Wa
    0031          ffb1111        27        26      1              0.02         1.038 +- 0.200        0.963 +- 0.189  [7 ] LS LS LS LS Ac Wa Wa

    0032          deefb11         0        52    -52             52.00         0.000 +- 0.000        0.000 +- 0.000  [7 ] LS LS Ac Wa Py Py Va         ###

    0033           3fbb11         0        52    -52             52.00         0.000 +- 0.000        0.000 +- 0.000  [6 ] LS LS Ac Ac Wa Ty
    0034         defbb111         0        49    -49             49.00         0.000 +- 0.000        0.000 +- 0.000  [8 ] LS LS LS Ac Ac Wa Py Va

    0035            22fb1        47         0     47             47.00         0.000 +- 0.000        0.000 +- 0.000  [5 ] LS Ac Wa St St
    0036         11111111        18        26     -8              1.45         0.692 +- 0.163        1.444 +- 0.283  [8 ] LS LS LS LS LS LS LS LS
    0037            bb111        23        21      2              0.09         1.095 +- 0.228        0.913 +- 0.199  [5 ] LS LS LS Ac Ac
    0038            2fbb1         0        38    -38             38.00         0.000 +- 0.000        0.000 +- 0.000  [5 ] LS Ac Ac Wa St
    .                              11142     11142       764.05/48 = 15.92  (pval:0.000 prob:1.000)  

    In [6]:                             


Increasing microstep suppression cut eliminates PyPy leaving AcAc::

    In [6]: ab.mat[:40]                                                                                                                                                                             
    Out[6]: 
    ab.mat
    .       seqmat_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11142     11142       623.28/46 = 13.55  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000            defb1      1994      1961     33              0.28         1.017 +- 0.023        0.983 +- 0.022  [5 ] LS Ac Wa Py Va
    0001               11      1821      1781     40              0.44         1.022 +- 0.024        0.978 +- 0.023  [2 ] LS LS
    0002           defb11      1557      1364    193             12.75         1.141 +- 0.029        0.876 +- 0.024  [6 ] LS LS Ac Wa Py Va
    0003              111       832       863    -31              0.57         0.964 +- 0.033        1.037 +- 0.035  [3 ] LS LS LS
    0004          defb111       887       759    128              9.95         1.169 +- 0.039        0.856 +- 0.031  [7 ] LS LS LS Ac Wa Py Va
    0005             3fb1       571       481     90              7.70         1.187 +- 0.050        0.842 +- 0.038  [4 ] LS Ac Wa Ty
    0006             1111       423       442    -19              0.42         0.957 +- 0.047        1.045 +- 0.050  [4 ] LS LS LS LS
    0007         defb1111       403       432    -29              1.01         0.933 +- 0.046        1.072 +- 0.052  [8 ] LS LS LS LS Ac Wa Py Va
    0008            3fb11       340       300     40              2.50         1.133 +- 0.061        0.882 +- 0.051  [5 ] LS LS Ac Wa Ty
    0009            11111       208       217     -9              0.19         0.959 +- 0.066        1.043 +- 0.071  [5 ] LS LS LS LS LS
    0010        defb11111       209       175     34              3.01         1.194 +- 0.083        0.837 +- 0.063  [9 ] LS LS LS LS LS Ac Wa Py Va
    0011           3fb111       157       174    -17              0.87         0.902 +- 0.072        1.108 +- 0.084  [6 ] LS LS LS Ac Wa Ty
    0012             ffb1       121       100     21              2.00         1.210 +- 0.110        0.826 +- 0.083  [4 ] LS Ac Wa Wa
    0013       defb111111        95       102     -7              0.25         0.931 +- 0.096        1.074 +- 0.106  [10] LS LS LS LS LS LS Ac Wa Py Va
    0014           111111       107        88     19              1.85         1.216 +- 0.118        0.822 +- 0.088  [6 ] LS LS LS LS LS LS
    0015           deffb1        87        74     13              1.05         1.176 +- 0.126        0.851 +- 0.099  [6 ] LS Ac Wa Wa Py Va
    0016          3fb1111        81        79      2              0.03         1.025 +- 0.114        0.975 +- 0.110  [7 ] LS LS LS LS Ac Wa Ty
    0017            ffb11        83        65     18              2.19         1.277 +- 0.140        0.783 +- 0.097  [5 ] LS LS Ac Wa Wa
    0018          deffb11        62        57      5              0.21         1.088 +- 0.138        0.919 +- 0.122  [7 ] LS LS Ac Wa Wa Py Va

    0019            3fbb1         0       112   -112            112.00         0.000 +- 0.000        0.000 +- 0.000  [5 ] LS Ac Ac Wa Ty

    0020              bb1        58        46     12              1.38         1.261 +- 0.166        0.793 +- 0.117  [3 ] LS Ac Ac

    0021           ffb111        54        46      8              0.64         1.174 +- 0.160        0.852 +- 0.126  [6 ] LS LS LS Ac Wa Wa
    0022          1111111        41        43     -2              0.05         0.953 +- 0.149        1.049 +- 0.160  [7 ] LS LS LS LS LS LS LS
    0023             bb11        52        30     22              5.90         1.733 +- 0.240        0.577 +- 0.105  [4 ] LS LS Ac Ac

    0024           defbb1         0        81    -81             81.00         0.000 +- 0.000        0.000 +- 0.000  [6 ] LS Ac Ac Wa Py Va

    0025          defbb11         1        79    -78             76.05         0.013 +- 0.013       79.000 +- 8.888  [7 ] LS LS Ac Ac Wa Py Va

    0026       efb1111111        41        38      3              0.11         1.079 +- 0.169        0.927 +- 0.150  [10] LS LS LS LS LS LS LS Ac Wa Py
    0027         3fb11111        42        34      8              0.84         1.235 +- 0.191        0.810 +- 0.139  [8 ] LS LS LS LS LS Ac Wa Ty
    0028         deffb111        36        30      6              0.55         1.200 +- 0.200        0.833 +- 0.152  [8 ] LS LS LS Ac Wa Wa Py Va
    0029       fb11111111        27        36     -9              1.29         0.750 +- 0.144        1.333 +- 0.222  [10] LS LS LS LS LS LS LS LS Ac Wa
    0030          ffb1111        27        26      1              0.02         1.038 +- 0.200        0.963 +- 0.189  [7 ] LS LS LS LS Ac Wa Wa

    0031           3fbb11         0        52    -52             52.00         0.000 +- 0.000        0.000 +- 0.000  [6 ] LS LS Ac Ac Wa Ty

    0032         defbb111         0        50    -50             50.00         0.000 +- 0.000        0.000 +- 0.000  [8 ] LS LS LS Ac Ac Wa Py Va
    0033            22fb1        47         0     47             47.00         0.000 +- 0.000        0.000 +- 0.000  [5 ] LS Ac Wa St St
    0034         11111111        18        26     -8              1.45         0.692 +- 0.163        1.444 +- 0.283  [8 ] LS LS LS LS LS LS LS LS
    0035            bb111        23        21      2              0.09         1.095 +- 0.228        0.913 +- 0.199  [5 ] LS LS LS Ac Ac

    0036            2fbb1         0        38    -38             38.00         0.000 +- 0.000        0.000 +- 0.000  [5 ] LS Ac Ac Wa St

    0037        deffb1111        23        13     10              2.78         1.769 +- 0.369        0.565 +- 0.157  [9 ] LS LS LS LS Ac Wa Wa Py Va
    0038        3fb111111        18        18      0              0.00         1.000 +- 0.236        1.000 +- 0.236  [9 ] LS LS LS LS LS LS Ac Wa Ty
    .                              11142     11142       623.28/46 = 13.55  (pval:0.000 prob:1.000)  








Increase suppression cut::


     83 CRecorder::CRecorder(CCtx& ctx)
     84     :
     85     m_ctx(ctx),
     86     m_ok(m_ctx.getOpticks()),
     87     m_microStep_mm(0.004),
     88     m_suppress_same_material_microStep(true),
     89     m_mode(m_ok->getManagerMode()),   // --managermode
     90     m_recpoi(m_ok->isRecPoi()),   // --recpoi
     91     m_reccf(m_ok->isRecCf()),     // --reccf





Look into AcAc
-----------------

There is one in OK from scatter in the acrylic::


    In [7]: a.selmat = "LS LS Ac Ac Wa Py Va"                                                                                                                                                       

    In [8]: a.seqhis_ana.table                                                                                                                                                                      
    Out[8]: 
    seqhis_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                                  1         1.00 
    n             iseq         frac           a    a-b      [ns] label
    0000          8cc6c62        1.000           1        [7 ] SI SC BT SC BT BT SA
    n             iseq         frac           a    a-b      [ns] label
    .                                  1         1.00 

    In [9]:                              



Ending on "Ac Ac" is all from absorption in Ac::

    In [9]: b.selmat = "LS Ac Ac"                                                                                                                                                                    

    In [11]: b.seqhis_ana.table                                                                                                                                                                     
    Out[11]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                                 46         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000              4c2        1.000          46        [3 ] SI BT AB
       n             iseq         frac           a    a-b      [ns] label
    .                                 46         1.00 



    In [12]: b.selmat = "LS Ac Ac Wa Ty"                                                                                                                                                            

    In [18]: a.selmat = "LS Ac Ac Wa Ty"                                                                                                                                                            
    [{_init_selection     :evt.py    :1312} WARNING  - _init_selection EMPTY nsel 0 len(psel) 11142 


    ## what is causing "Ac Ac" in G4 ? 
    ## really need equivalent of OK boundary sequence for G4 in order to understand

    In [13]: b.seqhis_ana.table                                                                                                                                                                     
    Out[13]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                                112         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000            8ccc2        1.000         112        [5 ] SI BT BT BT SA
       n             iseq         frac           a    a-b      [ns] label
    .                                112         1.00 




    In [14]: a.selmat = "LS Ac Wa Ty"                                                                                                                                                               

    In [15]: a.seqhis_ana.table                                                                                                                                                                     
    Out[15]: 
    seqhis_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                                571         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000             8cc2        0.998         570        [4 ] SI BT BT SA
    0001             8cc1        0.002           1        [4 ] CK BT BT SA
       n             iseq         frac           a    a-b      [ns] label
    .                                571         1.00 


    In [16]: b.selmat = "LS Ac Wa Ty"                                                                                                                                                               

    In [17]: b.seqhis_ana.table                                                                                                                                                                     
    Out[17]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3gun 
    .                                481         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000             8cc2        1.000         481        [4 ] SI BT BT SA
       n             iseq         frac           a    a-b      [ns] label
    .                                481         1.00 




G4 analog of OK boundary sequence ?
--------------------------------------

The boundary is formed between a volume and its parent using the materials and surfaces.


But first perhaps Ac Ac is from the GPU excluded mm8 ?  YES, all the few tens looked at were from the dreaded uni_acrylic3 
------------------------------------------------------------------------------------------------------------------------------

::

    ab.mat
    ...
    0019            3fbb1         0       112   -112            112.00         0.000 +- 0.000        0.000 +- 0.000  [5 ] LS Ac Ac Wa Ty

::

    --dbgseqmat 0x3fbb1

    export DBGSEQMAT=0x3fbb1


::

    2021-06-26 01:02:28.563 INFO  [231502] [CRec::dump@194] CDebug::dump record_id 9727  origin[ 114.294358.154-140.885 ; 401.480]   Ori[ 114.294358.154-140.885 ; 401.480] 
    2021-06-26 01:02:28.563 INFO  [231502] [CRec::dump@200]  nstp 4
    (0 )  SI/BT     FrT                       PRE_SAVE POST_SAVE STEP_START 
    [   0](Stp ;opticalphoton stepNum    4(tk ;opticalphoton tid 14818 pid 14617 nm 561.183 mm  ori[  114.294 358.154-140.885 ;  401.480]  pos[ -2753.693-5900.258-18946.313 ; 20033.938]  )
      pre pv                   pTarget lv                   lTarget so                   sTarget mlv                  lAcrylic mso                  sAcrylic
      pre              LS          noProc           Undefined pos[      0.000     0.000     0.000 ;      0.000]  dir[   -0.142  -0.290  -0.946 ;    1.000]  pol[   -0.838  -0.473   0.271 ;    1.000]  ns  3.303 nm 561.183 mm/ns 198.199
     post pv                  pAcrylic lv                  lAcrylic so                  sAcrylic mlv               lInnerWater mso               sInnerWater
     post         Acrylic  Transportation        GeomBoundary pos[  -2518.064 -5132.587-16732.664 ;  17682.368]  dir[   -0.142  -0.290  -0.946 ;    1.000]  pol[   -0.400   0.891  -0.213 ;    1.000]  ns 92.518 nm 561.183 mm/ns 196.949
     Cfsp dpos[ -2518.064-5132.587-16732.664 ; 17682.368]  ddir[    0.000   0.000  -0.000 ;    0.000]  dpol[    0.438   1.364  -0.484 ;    1.512]  dtim[   89.215]        epsilon 1e-06
     )
    (1 )  BT/BT     SAM                                           POST_SAVE 
    [   1](Stp ;opticalphoton stepNum    4(tk ;opticalphoton tid 14818 pid 14617 nm 561.183 mm  ori[  114.294 358.154-140.885 ;  401.480]  pos[ -2753.693-5900.258-18946.313 ; 20033.938]  )
      pre pv                  pAcrylic lv                  lAcrylic so                  sAcrylic mlv               lInnerWater mso               sInnerWater
      pre         Acrylic  Transportation        GeomBoundary pos[  -2518.064 -5132.587-16732.664 ;  17682.368]  dir[   -0.142  -0.290  -0.946 ;    1.000]  pol[   -0.400   0.891  -0.213 ;    1.000]  ns 92.518 nm 561.183 mm/ns 196.949
     post pv            lAddition_phys lv                 lAddition so              uni_acrylic3 mlv               lInnerWater mso               sInnerWater
     post         Acrylic  Transportation        GeomBoundary pos[  -2535.153 -5167.414-16846.253 ;  17802.399]  dir[   -0.142  -0.290  -0.946 ;    1.000]  pol[   -0.400   0.891  -0.213 ;    1.000]  ns 93.128 nm 561.183 mm/ns 196.949
     Cfsp dpos[  -17.089 -34.827-113.588 ;  120.030]  same_dir same_pol dtim[    0.609]        epsilon 1e-06
     )
    (2 )  BT/BT     FrT                                           POST_SAVE 
    [   2](Stp ;opticalphoton stepNum    4(tk ;opticalphoton tid 14818 pid 14617 nm 561.183 mm  ori[  114.294 358.154-140.885 ;  401.480]  pos[ -2753.693-5900.258-18946.313 ; 20033.938]  )
      pre pv            lAddition_phys lv                 lAddition so              uni_acrylic3 mlv               lInnerWater mso               sInnerWater
      pre         Acrylic  Transportation        GeomBoundary pos[  -2535.153 -5167.414-16846.253 ;  17802.399]  dir[   -0.142  -0.290  -0.946 ;    1.000]  pol[   -0.400   0.891  -0.213 ;    1.000]  ns 93.128 nm 561.183 mm/ns 196.949
     post pv               pInnerWater lv               lInnerWater so               sInnerWater mlv            lReflectorInCD mso            sReflectorInCD
     post           Water  Transportation        GeomBoundary pos[  -2550.093 -5197.863-16945.563 ;  17907.341]  dir[   -0.096  -0.330  -0.939 ;    1.000]  pol[   -0.408   0.874  -0.265 ;    1.000]  ns 93.661 nm 561.183 mm/ns 218.189
     Cfsp dpos[  -14.941 -30.449 -99.311 ;  104.943]  ddir[    0.047  -0.040   0.007 ;    0.062]  dpol[   -0.007  -0.017  -0.052 ;    0.056]  dtim[    0.533]        epsilon 1e-06
     )
    (3 )  BT/SA     NRI              POST_SAVE POST_DONE LAST_POST SURF_ABS 
    [   3](Stp ;opticalphoton stepNum    4(tk ;opticalphoton tid 14818 pid 14617 nm 561.183 mm  ori[  114.294 358.154-140.885 ;  401.480]  pos[ -2753.693-5900.258-18946.313 ; 20033.938]  )
      pre pv               pInnerWater lv               lInnerWater so               sInnerWater mlv            lReflectorInCD mso            sReflectorInCD
      pre           Water  Transportation        GeomBoundary pos[  -2550.093 -5197.863-16945.563 ;  17907.341]  dir[   -0.096  -0.330  -0.939 ;    1.000]  pol[   -0.408   0.874  -0.265 ;    1.000]  ns 93.661 nm 561.183 mm/ns 218.189
     post pv          pCentralDetector lv            lReflectorInCD so            sReflectorInCD mlv           lOuterWaterPool mso           sOuterWaterPool
     post           Tyvek  Transportation        GeomBoundary pos[  -2753.693 -5900.258-18946.313 ;  20033.938]  dir[   -0.096  -0.330  -0.939 ;    1.000]  pol[   -0.408   0.874  -0.265 ;    1.000]  ns 103.424 nm 561.183 mm/ns 218.189
     Cfsp dpos[ -203.599-702.395-2000.750 ; 2130.214]  same_dir same_pol dtim[    9.763]        epsilon 1e-06
     )
    2021-06-26 01:02:28.564 INFO  [231502] [CRec::dump@204]  npoi 0
    2021-06-26 01:02:28.564 INFO  [231502] [CDebug::dump_brief@204] CRecorder::dump_brief m_ctx._record_id     9727 m_



How to switch off lAddition in G4 ?
------------------------------------

* ~/j/issues/comment_setupCD_Sticks.rst

::

    svn diff Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstruction.cc

    +
    +#ifdef WITH_G4OPTICKS
    +  LOG(LEVEL) << " OPTICKS DEBUGGING : SKIP LSExpDetectorConstruction::setupCD_Sticks " ; 
    +#else
       setupCD_Sticks(cd_det);
    +#endif


Unrelated issue when compiling JUNO offline
------------------------------------------------

*  ~/j/issues/offline_deuteron_evar_crash.rst


After remove the sticks : the poppy becomes somewhat less OK reemission 
-------------------------------------------------------------------------------------

* move along to :doc:`ok_less_reemission`

::

    In [4]: ab.his[:50]                                                                                                                                                                             
    Out[4]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11684     11684       109.20/59 =  1.85  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               42      1741      1721     20              0.12         1.012 +- 0.024        0.989 +- 0.024  [2 ] SI AB
    0001            7ccc2      1480      1406     74              1.90         1.053 +- 0.027        0.950 +- 0.025  [5 ] SI BT BT BT SD
    0002           7ccc62       737       666     71              3.59         1.107 +- 0.041        0.904 +- 0.035  [6 ] SI SC BT BT BT SD
    0003            8ccc2       660       597     63              3.16         1.106 +- 0.043        0.905 +- 0.037  [5 ] SI BT BT BT SA
    0004             8cc2       629       615     14              0.16         1.023 +- 0.041        0.978 +- 0.039  [4 ] SI BT BT SA
    0005              452       436       536   -100             10.29         0.813 +- 0.039        1.229 +- 0.053  [3 ] SI RE AB               ## LESS OK_RE 
    0006           7ccc52       424       438    -14              0.23         0.968 +- 0.047        1.033 +- 0.049  [6 ] SI RE BT BT BT SD
    0007              462       425       405     20              0.48         1.049 +- 0.051        0.953 +- 0.047  [3 ] SI SC AB
    0008           8ccc62       283       262     21              0.81         1.080 +- 0.064        0.926 +- 0.057  [6 ] SI SC BT BT BT SA
    0009          7ccc662       266       222     44              3.97         1.198 +- 0.073        0.835 +- 0.056  [7 ] SI SC SC BT BT BT SD
    0010            8cc62       209       212     -3              0.02         0.986 +- 0.068        1.014 +- 0.070  [5 ] SI SC BT BT SA
    0011          7ccc652       187       205    -18              0.83         0.912 +- 0.067        1.096 +- 0.077  [7 ] SI RE SC BT BT BT SD
    0012           8ccc52       189       201    -12              0.37         0.940 +- 0.068        1.063 +- 0.075  [6 ] SI RE BT BT BT SA
    0013            8cc52       151       192    -41              4.90         0.786 +- 0.064        1.272 +- 0.092  [5 ] SI RE BT BT SA         ### LESS OK:RE 
    0014               41       162       145     17              0.94         1.117 +- 0.088        0.895 +- 0.074  [2 ] CK AB
    0015          7ccc552       133       160    -27              2.49         0.831 +- 0.072        1.203 +- 0.095  [7 ] SI RE RE BT BT BT SD
    0016             4552       124       165    -41              5.82         0.752 +- 0.067        1.331 +- 0.104  [4 ] SI RE RE AB            ### LESS OK:RE
    0017             4cc2       133       115     18              1.31         1.157 +- 0.100        0.865 +- 0.081  [4 ] SI BT BT AB
    0018             4662       136       110     26              2.75         1.236 +- 0.106        0.809 +- 0.077  [4 ] SI SC SC AB
    0019             4652       121       117      4              0.07         1.034 +- 0.094        0.967 +- 0.089  [4 ] SI RE SC AB
    0020          8ccc662        86       108    -22              2.49         0.796 +- 0.086        1.256 +- 0.121  [7 ] SI SC SC BT BT BT SA
    0021         7ccc6662        87        91     -4              0.09         0.956 +- 0.102        1.046 +- 0.110  [8 ] SI SC SC SC BT BT BT SD
    0022          8ccc652        77        79     -2              0.03         0.975 +- 0.111        1.026 +- 0.115  [7 ] SI RE SC BT BT BT SA
    0023         7ccc6652        59        86    -27              5.03         0.686 +- 0.089        1.458 +- 0.157  [8 ] SI RE SC SC BT BT BT SD
    0024          7ccc562        76        51     25              4.92         1.490 +- 0.171        0.671 +- 0.094  [7 ] SI SC RE BT BT BT SD
    0025           8cc662        57        69    -12              1.14         0.826 +- 0.109        1.211 +- 0.146  [6 ] SI SC SC BT BT SA
    0026          8ccc552        62        63     -1              0.01         0.984 +- 0.125        1.016 +- 0.128  [7 ] SI RE RE BT BT BT SA
    0027         7ccc6552        53        71    -18              2.61         0.746 +- 0.103        1.340 +- 0.159  [8 ] SI RE RE SC BT BT BT SD
    0028             4562        49        66    -17              2.51         0.742 +- 0.106        1.347 +- 0.166  [4 ] SI SC RE AB
    0029           8cc552        38        70    -32              9.48         0.543 +- 0.088        1.842 +- 0.220  [6 ] SI RE RE BT BT SA
    0030            4cc62        55        52      3              0.08         1.058 +- 0.143        0.945 +- 0.131  [5 ] SI SC BT BT AB
    0031           8cc652        56        50      6              0.34         1.120 +- 0.150        0.893 +- 0.126  [6 ] SI RE SC BT BT SA
    0032           7cccc2        53        51      2              0.04         1.039 +- 0.143        0.962 +- 0.135  [6 ] SI BT BT BT BT SD
    0033              4c2        57        35     22              5.26         1.629 +- 0.216        0.614 +- 0.104  [3 ] SI BT AB
    0034            45552        29        49    -20              5.13         0.592 +- 0.110        1.690 +- 0.241  [5 ] SI RE RE RE AB
    0035            46662        42        34      8              0.84         1.235 +- 0.191        0.810 +- 0.139  [5 ] SI SC SC SC AB
    0036            4cc52        34        40     -6              0.49         0.850 +- 0.146        1.176 +- 0.186  [5 ] SI RE BT BT AB
    0037         8ccc6662        36        36      0              0.00         1.000 +- 0.167        1.000 +- 0.167  [8 ] SI SC SC SC BT BT BT SA
    0038         7ccc5552        36        35      1              0.01         1.029 +- 0.171        0.972 +- 0.164  [8 ] SI RE RE RE BT BT BT SD
    0039            46552        37        31      6              0.53         1.194 +- 0.196        0.838 +- 0.150  [5 ] SI RE RE SC AB
    0040            46652        38        29      9              1.21         1.310 +- 0.213        0.763 +- 0.142  [5 ] SI RE SC SC AB
    0041          8ccc562        33        33      0              0.00         1.000 +- 0.174        1.000 +- 0.174  [7 ] SI SC RE BT BT BT SA



    Out[8]: 
    ab.mat
    .       seqmat_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11684     11684        42.10/35 =  1.20  (pval:0.191 prob:0.809)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000            cdeb1      2144      2014    130              4.06         1.065 +- 0.023        0.939 +- 0.021  [5 ] LS Ac Py Va PE
    0001               11      1902      1866     36              0.34         1.019 +- 0.023        0.981 +- 0.023  [2 ] LS LS
    0002           cdeb11      1662      1582     80              1.97         1.051 +- 0.026        0.952 +- 0.024  [6 ] LS LS Ac Py Va PE
    0003          cdeb111       938       931      7              0.03         1.008 +- 0.033        0.993 +- 0.033  [7 ] LS LS LS Ac Py Va PE
    0004              111       873       959    -86              4.04         0.910 +- 0.031        1.099 +- 0.035  [3 ] LS LS LS
    0005             3eb1       630       618     12              0.12         1.019 +- 0.041        0.981 +- 0.039  [4 ] LS Ac Py Ty
    0006         cdeb1111       433       488    -55              3.28         0.887 +- 0.043        1.127 +- 0.051  [8 ] LS LS LS LS Ac Py Va PE
    0007             1111       436       472    -36              1.43         0.924 +- 0.044        1.083 +- 0.050  [4 ] LS LS LS LS
    0008            3eb11       370       411    -41              2.15         0.900 +- 0.047        1.111 +- 0.055  [5 ] LS LS Ac Py Ty
    0009        cdeb11111       230       254    -24              1.19         0.906 +- 0.060        1.104 +- 0.069  [9 ] LS LS LS LS LS Ac Py Va PE
    0010            11111       218       213      5              0.06         1.023 +- 0.069        0.977 +- 0.067  [5 ] LS LS LS LS LS
    0011           3eb111       174       211    -37              3.56         0.825 +- 0.063        1.213 +- 0.083  [6 ] LS LS LS Ac Py Ty
    0012             eeb1       133       116     17              1.16         1.147 +- 0.099        0.872 +- 0.081  [4 ] LS Ac Py Py
    0013       cdeb111111       103       129    -26              2.91         0.798 +- 0.079        1.252 +- 0.110  [10] LS LS LS LS LS LS Ac Py Va PE
    0014           111111       111        96     15              1.09         1.156 +- 0.110        0.865 +- 0.088  [6 ] LS LS LS LS LS LS
    0015          3eb1111        94        95     -1              0.01         0.989 +- 0.102        1.011 +- 0.104  [7 ] LS LS LS LS Ac Py Ty
    0016            eeb11        90        94     -4              0.09         0.957 +- 0.101        1.044 +- 0.108  [5 ] LS LS Ac Py Py
    0017           cdeeb1        87        91     -4              0.09         0.956 +- 0.102        1.046 +- 0.110  [6 ] LS Ac Py Py Va PE
    0018          cdeeb11        66        64      2              0.03         1.031 +- 0.127        0.970 +- 0.121  [7 ] LS LS Ac Py Py Va PE
    0019           eeb111        57        67    -10              0.81         0.851 +- 0.113        1.175 +- 0.144  [6 ] LS LS LS Ac Py Py
    0020       deb1111111        48        61    -13              1.55         0.787 +- 0.114        1.271 +- 0.163  [10] LS LS LS LS LS LS LS Ac Py Va
    0021              bb1        58        35     23              5.69         1.657 +- 0.218        0.603 +- 0.102  [3 ] LS Ac Ac
    0022             bb11        50        42      8              0.70         1.190 +- 0.168        0.840 +- 0.130  [4 ] LS LS Ac Ac
    0023         3eb11111        45        43      2              0.05         1.047 +- 0.156        0.956 +- 0.146  [8 ] LS LS LS LS LS Ac Py Ty
    0024          1111111        43        45     -2              0.05         0.956 +- 0.146        1.047 +- 0.156  [7 ] LS LS LS LS LS LS LS
    0025          eeb1111        32        51    -19              4.35         0.627 +- 0.111        1.594 +- 0.223  [7 ] LS LS LS LS Ac Py Py
    0026         cdeeb111        36        37     -1              0.01         0.973 +- 0.162        1.028 +- 0.169  [8 ] LS LS LS Ac Py Py Va PE
    0027       eb11111111        30        27      3              0.16         1.111 +- 0.203        0.900 +- 0.173  [10] LS LS LS LS LS LS LS LS Ac Py
    0028        cdeeb1111        26        29     -3              0.16         0.897 +- 0.176        1.115 +- 0.207  [9 ] LS LS LS LS Ac Py Py Va PE
    0029            bb111        25        26     -1              0.02         0.962 +- 0.192        1.040 +- 0.204  [5 ] LS LS LS Ac Ac
    0030       1111111111        23        22      1              0.02         1.045 +- 0.218        0.957 +- 0.204  [10] LS LS LS LS LS LS LS LS LS LS
    0031       b111111111        22        20      2              0.10         1.100 +- 0.235        0.909 +- 0.203  [10] LS LS LS LS LS LS LS LS LS Ac
    0032         eeb11111        19        21     -2              0.10         0.905 +- 0.208        1.105 +- 0.241  [8 ] LS LS LS LS LS Ac Py Py
    0033           bb1111        21        17      4              0.42         1.235 +- 0.270        0.810 +- 0.196  [6 ] LS LS LS LS Ac Ac
    0034        3eb111111        17        16      1              0.03         1.062 +- 0.258        0.941 +- 0.235  [9 ] LS LS LS LS LS LS Ac Py Ty
    0035         11111111        17        14      3              0.29         1.214 +- 0.295        0.824 +- 0.220  [8 ] LS LS LS LS LS LS LS LS
    0036            3eeb1        14        15     -1              0.00         0.933 +- 0.249        1.071 +- 0.277  [5 ] LS Ac Py Py Ty
    0037        111111111         9        12     -3              0.00         0.750 +- 0.250        1.333 +- 0.385  [9 ] LS LS LS LS LS LS LS LS LS
    0038       cdeeb11111         9        12     -3              0.00         0.750 +- 0.250        1.333 +- 0.385  [10] LS LS LS LS LS Ac Py Py Va PE
    0039           3eeb11        10        10      0              0.00         1.000 +- 0.316        1.000 +- 0.316  [6 ] LS LS Ac Py Py Ty
    0040          bb11111         7        13     -6              0.00         0.538 +- 0.204        1.857 +- 0.515  [7 ] LS LS LS LS LS Ac Ac
    0041       bb1bb1bb11        10        10      0              0.00         1.000 +- 0.316        1.000 +- 0.316  [10] LS LS Ac Ac LS Ac Ac LS Ac Ac
    0042         3eddeb11        20         0     20              0.00         0.000 +- 0.000        0.000 +- 0.000  [8 ] LS LS Ac Py Va Va Py Ty
    0043        3edddeb11         0        19    -19              0.00         0.000 +- 0.000        0.000 +- 0.000  [9 ] LS LS Ac Py Va Va Va Py Ty
    0044       3edddeb111         0        17    -17              0.00         0.000 +- 0.000        0.000 +- 0.000  [10] LS LS LS Ac Py Va Va Va Py Ty
    0045        eeb111111        10         7      3              0.00         1.429 +- 0.452        0.700 +- 0.265  [9 ] LS LS LS LS LS LS Ac Py Py
    0046       3eddeb1111        16         0     16              0.00         0.000 +- 0.000        0.000 +- 0.000  [10] LS LS LS LS Ac Py Va Va Py Ty
    0047       deeb111111         8         8      0              0.00         1.000 +- 0.354        1.000 +- 0.354  [10] LS LS LS LS LS LS Ac Py Py Va
    0048            ddeb1         9         7      2              0.00         1.286 +- 0.429        0.778 +- 0.294  [5 ] LS Ac Py Va Va
    .                              11684     11684        42.10/35 =  1.20  (pval:0.191 prob:0.809)  


