Optical Photon Propagation Times need to use GROUPVEL
=======================================================

GROUPVEL material property is auto added by Geant4 (calulated from RINDEX) 
at the first *G4MaterialPropertiesTable::GetProperty*

Where/when/how to bring into GGeo ? 
----------------------------------------

Another material property added to the G4DAE export ? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just need to call *mpt->GetProperty("GROUPVEL")*  on all G4 materials
before (or as part of) the export for the property to be created 
and thus seen by the exporter.

Thus approach is simple but somewhat arduous, if the calulation can 
easily be reproduced can do so on RINDEX at later stage.

::

    ggv-;ggv-pmt-test --cdetector --export --exportconfig /tmp/test.dae

    delta:issues blyth$ grep GROUPVEL /tmp/test.dae
            <matrix coldim="2" name="GROUPVEL0x7f8f5147a3d0">1.512e-06 205.619 1.5498e-06 205.619 1.58954e-06 205.619 1.63137e-06 205.619 1.67546e-06 205.619 1.722e-06 205.619 1.7712e-06 205.619 1.8233e-06 205.619 1.87855e-06 205.619 1.93725e-06 205.619 1.99974e-06 205.619 2.0664e-06 205.619 2.13766e-06 205.619 2.214e-06 205.619 2.296e-06 205.619 2.38431e-06 205.619 2.47968e-06 205.619 2.583e-06 205.619 2.69531e-06 205.619 2.81782e-06 205.619 2.952e-06 205.619 3.0996e-06 205.619 3.26274e-06 205.619 3.44401e-06 205.619 3.64659e-06 205.619 3.87451e-06 205.619 4.13281e-06 205.619 4.42801e-06 205.619 4.76862e-06 205.619 5.16601e-06 205.619 5.63564e-06 205.619 6.19921e-06 205.619 6.88801e-06 205.619 7.74901e-06 205.619 8.85601e-06 205.619 1.0332e-05 205.619 1.23984e-05 205.619 1.5498e-05 205.619 2.0664e-05 205.619</matrix>
            <property name="GROUPVEL" ref="GROUPVEL0x7f8f5147a3d0"/>
    ...


*AssimpGGeo::convertMaterials*::

     430 
     431             //printf("AssimpGGeo::convertMaterials aiScene materialIndex %u (GMaterial) name %s \n", i, name);
     432             GMaterial* gmat = new GMaterial(name, index);
     433             gmat->setStandardDomain(standard_domain);
     434             addProperties(gmat, mat );
     435             gg->add(gmat);
     436 
     437             {
     438                 // without standard domain applied
     439                 GMaterial* gmat_raw = new GMaterial(name, index);
     440                 addProperties(gmat_raw, mat );
     441                 gg->addRaw(gmat_raw);
     442             }
     443 
      

     All properties get incorporated into the pmap by AssimpGGeo

     void AssimpGGeo::addPropertyVector(GPropertyMap<float>* pmap, const char* k, aiMaterialProperty* property )



GROUPVEL in Geant4
---------------------

* :google:`geant4 bug report 741`

* http://bugzilla-geant4.kek.jp/show_bug.cgi?id=741

Optical photons were incorrectly propagating at phase velocity, should be group velocity


G. Horton-Smith, 2005/04/14
-----------------------------

* http://neutrino.phys.ksu.edu/~gahs/G4_GROUPVEL_fix/
* http://neutrino.phys.ksu.edu/~gahs/G4_GROUPVEL_fix/G4Track.patch

As described in Geant4 bug report #741, optical photons in Geant4 release 7.0
propagate at the phase velocity c/n(E), where E is the energy of the photon.
This is a bug because photons in real life propagate at the group velocity 
vg = c/(n(E)+dn/d(log(E)).


Geant4 fix into 4.7.1
-----------------------

* http://geant4.web.cern.ch/geant4/support/ReleaseNotes4.7.1.html

Added SetGROUPVEL() to G4MaterialPropertiesTable. Addresses problem report #741.


Observe
--------

* https://www.physicsforums.com/threads/trying-to-derive-a-group-velocity-equation.441274/


Is that the same ? ::

   d(log(E))     1
  ---------  =  --  
     dE          E

* https://en.wikipedia.org/wiki/Dispersion_(optics)

    vg = c / ( n - w dn/dw )



G4MaterialPropertiesTable::SetGROUPVEL
------------------------------------------


::

    034 // File: G4MaterialPropertiesTable.cc 
     35 // Version:     1.0
     36 // Created:     1996-02-08
     37 // Author:      Juliet Armstrong
     38 // Updated:     2005-05-12 add SetGROUPVEL(), courtesy of
     39 //              Horton-Smith (bug report #741), by P. Gumplinger


    119 G4MaterialPropertyVector* G4MaterialPropertiesTable::SetGROUPVEL()
    120 {
    ...
    141   G4MaterialPropertyVector* groupvel = new G4MaterialPropertyVector();
    142 
    143   // fill GROUPVEL vector using RINDEX values
    144   // rindex built-in "iterator" was advanced to first entry above
    145   //
    146   G4double E0 = rindex->Energy(0);
    147   G4double n0 = (*rindex)[0];
    ...
    160     G4double E1 = rindex->Energy(1);
    161     G4double n1 = (*rindex)[1];
    168 
    169     G4double vg;
    170 
    171     // add entry at first photon energy
    172     //
    173     vg = c_light/(n0+(n1-n0)/std::log(E1/E0));
    174 
    175     // allow only for 'normal dispersion' -> dn/d(logE) > 0
    176     //
    177     if((vg<0) || (vg>c_light/n0))  { vg = c_light/n0; }
    178 
    179     groupvel->InsertValues( E0, vg );
    180 
    181     // add entries at midpoints between remaining photon energies
    182     //
    183 
    184     for (size_t i = 2; i < rindex->GetVectorLength(); i++)
    185     {
    186       vg = c_light/( 0.5*(n0+n1)+(n1-n0)/std::log(E1/E0));
    187 
    188       // allow only for 'normal dispersion' -> dn/d(logE) > 0
    189       //
    190       if((vg<0) || (vg>c_light/(0.5*(n0+n1))))  { vg = c_light/(0.5*(n0+n1)); }
    191       groupvel->InsertValues( 0.5*(E0+E1), vg );
    192 
    193       // get next energy/value pair, or exit loop
    194       //
    195       E0 = E1;
    196       n0 = n1;
    197       E1 = rindex->Energy(i);
    198       n1 = (*rindex)[i];
    199 
    200       if (E1 <= 0.)
    201       {
    202         G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    203                     FatalException, "Optical Photon Energy <= 0");
    204       }
    205     }
    206 
    207     // add entry at last photon energy
    208     //
    209     vg = c_light/(n1+(n1-n0)/std::log(E1/E0));
    210 
    211     // allow only for 'normal dispersion' -> dn/d(logE) > 0
    212     //
    213     if((vg<0) || (vg>c_light/n1))  { vg = c_light/n1; }
    214     groupvel->InsertValues( E1, vg );
    215   }
    216   else // only one entry in RINDEX -- weird!
    217   {
    218     groupvel->InsertValues( E0, c_light/n0 );
    219   }
    220 
    221   this->AddProperty( "GROUPVEL", groupvel );
    222 
    223   return groupvel;
    224 }



Recreate the calc ?
----------------------

::

    In [22]: np.dstack([w, n])
    Out[22]: 
    array([[[  60.   ,    1.434],
            [  79.737,    1.434],
            [  99.474,    1.434],
            [ 119.211,    1.434],
            [ 138.947,    1.642],


Negative is normal dispersion (n down and w up)::

    In [26]: 1000.*dn/dw
    Out[26]: 
    array([  0.   ,   0.   ,   0.   ,  10.542,   5.896, -12.743,   4.491,
            -0.933,  -0.933,  -0.933,  -0.933,  -0.933,  -0.264,  -0.264,
            -0.264,  -0.264,  -0.264,  -0.144,  -0.105,  -0.095,  -0.095,
            -0.072,  -0.062,  -0.062,  -0.06 ,  -0.059,  -0.048,  -0.039,
            -0.039,  -0.039,  -0.039,  -0.028,  -0.016,  -0.016,  -0.016,
            -0.016,  -0.016,   0.   ])

    In [27]: n
    Out[27]: 
    array([ 1.434,  1.434,  1.434,  1.434,  1.642,  1.758,  1.507,  1.596,
            1.577,  1.559,  1.54 ,  1.522,  1.503,  1.498,  1.493,  1.488,
            1.483,  1.477,  1.475,  1.473,  1.471,  1.469,  1.467,  1.466,
            1.465,  1.464,  1.463,  1.462,  1.461,  1.46 ,  1.459,  1.459,
            1.458,  1.458,  1.457,  1.457,  1.457,  1.456,  1.456], dtype=float32)




* https://en.wikipedia.org/wiki/Talk%3ADispersion_(optics)

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons/420/1.html

* https://indico.fnal.gov/contributionDisplay.py?sessionId=18&contribId=41&confId=4535




Bringing GROUPVEL into Opticks
--------------------------------

Properties are fed in via the boundary texture::


     705 void App::prepareOptiX()
     706 {
     ...
     723     m_olib = new OBndLib(context,m_ggeo->getBndLib());
     724     m_olib->convert();


     23 void OBndLib::makeBoundaryTexture(NPY<float>* buf)
     24 {
     25     //  eg (123, 4, 39, 4)   boundary, imat-omat-isur-osur, wavelength-samples, 4-props
     26 
     27     unsigned int ni = buf->getShape(0);
     28     unsigned int nj = buf->getShape(1);
     29     unsigned int nk = buf->getShape(2);
     30     unsigned int nl = buf->getShape(3);
     31 
     32     assert(ni == m_lib->getNumBnd()) ;
     33     assert(nj == GPropertyLib::NUM_QUAD && nk == Opticks::DOMAIN_LENGTH && nl == GPropertyLib::NUM_PROP );
     34 
     35     unsigned int nx = nk ;
     36     unsigned int ny = ni*nj ;   // not nl as using float4
     37 
     38     LOG(debug) << "OBndLib::makeBoundaryTexture buf "
     39               << buf->getShapeString()
     40               << " ---> "
     41               << " nx " << nx
     42               << " ny " << ny
     43               ;
     44 
     45     optix::TextureSampler tex = makeTexture(buf, RT_FORMAT_FLOAT4, nx, ny);
     46 
       

     Source Bnd buffer is a memcpy interweave of Material and Surface buffers

     393 NPY<float>* GBndLib::createBuffer()
     394 {
     395     NPY<float>* mat = m_mlib->getBuffer();
     396     NPY<float>* sur = m_slib->getBuffer();
     397 
     398     unsigned int ni = getNumBnd();
     399     unsigned int nj = NUM_QUAD ;       // im-om-is-os
     400     unsigned int nk = Opticks::DOMAIN_LENGTH ;
     401     unsigned int nl = NUM_PROP ;       // 4 interweaved props   
             // for materials the 4 props are refractive_index, scattering_length, absorption_length, reemission_probability 


     60 static __device__ __inline__ float4 wavelength_lookup(float nm, unsigned int line )
     61 {
     62     // x:low y:high z:step w:mid   tex coords are offset by 0.5 
     63     // texture lookups benefit from hardware interpolation 
     64     float nmi = (nm - boundary_domain.x)/boundary_domain.z + 0.5f ;
     65 
     66     if( line > boundary_bounds.w )
     67     {
     68         rtPrintf("wavelength_lookup OUT OF BOUNDS line %4d nmi %10.4f \n", line, nmi );
     69     }
     70 
     71     return line <= boundary_bounds.w ?
     72                   tex2D(boundary_texture, nmi, line + 0.5f ) :
     73                   make_float4(1.123456789f, 123456789.f, 123456789.f, 1.0f )    ;    // some obnoxious values for debug 
     74 
     75     // refractive_index, absorption_length, scattering_length, reemission_prob
     76     // DEBUG KLUDGE
     77 }

     27 __device__ void fill_state( State& s, int boundary, uint4 identity, float wavelength )
     28 {
     29     // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
     30     // >0 outward going photon
     31     // <0 inward going photon
     32 
     33     int line = boundary > 0 ? (boundary - 1)*BNUMQUAD : (-boundary - 1)*BNUMQUAD  ;
     34 
     35     // pick relevant lines depening on boundary sign, ie photon direction relative to normal
     36     // 
     37     int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;
     38     int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;
     39     int su_line = boundary > 0 ? line + ISUR : line + OSUR ;
     40 
     41     //  consider photons arriving at PMT cathode surface
     42     //  geometry normals are expected to be out of the PMT 
     43     //
     44     //  boundary sign will be -ve : so line+3 outer-surface is the relevant one
     45 
     46     s.material1 = wavelength_lookup( wavelength, m1_line );
     47     s.material2 = wavelength_lookup( wavelength, m2_line ) ;
     48     s.surface   = wavelength_lookup( wavelength, su_line );

     define.h:#define BNUMQUAD 4  // quads per boundary in wavelength texture


Would need to increase NUM_PROP(nl) and BNUMQUAD from 4 to 8  
Extra props could then be accessed at::

    int m1x_line = m1_line + 4 ;
    



