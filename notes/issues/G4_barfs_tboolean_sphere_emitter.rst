G4_barfs_tboolean_sphere_emitter
==================================


FIXED ISSUE
--------------

G4 barfs when emit photons from the sphere, and normal to it 

* fixed by doing the mom-dir/normal calc in double precision


Symptom
---------

::


    delta:issues blyth$ tboolean-;tboolean-sphere --okg4

    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  
    args = opticks_main(csgpath="/tmp/blyth/opticks/tboolean-sphere--")

    material = "Pyrex"

    CSG.kwa = dict(poly="IM", resolution="20" , emitconfig="photons=600000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1" )
    container = CSG("box",    param=[0,0,0,400.0], boundary="Rock//perfectAbsorbSurface/Vacuum", emit=0 )  
    sphere    = CSG("sphere", param=[0,0,0,200.0], boundary="Vacuum/perfectSpecularSurface//%s" % material, emit=1 ) 

    CSG.Serialize([container, sphere], args.csgpath )

    ...

    2017-11-11 12:38:04.986 INFO  [4368463] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
      G4ParticleChange::CheckIt  : the Momentum Change is not unit vector !!  Difference:  1.30658e-08
    opticalphoton E=3.26274e-06 pos=-0.0694692, 0.0930607, -0.162953
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :                  291
            Stepping Control      :                    0
        First Step In the voulme  : 
            Mass (GeV)   :                    0
            Charge (eplus)   :                    0
            MagneticMoment   :                    0
                    :  =                    0*[e hbar]/[2 m]
            Position - x (mm)   :                 -171
            Position - y (mm)   :                  228
            Position - z (mm)   :                 -400
            Time (ns)           :                0.971
            Proper Time (ns)    :                    0
            Momentum Direct - x :               -0.347
            Momentum Direct - y :                0.465
            Momentum Direct - z :               -0.814
            Kinetic Energy (MeV):             3.26e-06
            Velocity  (/c):                    1
            Polarization - x    :                   -0
            Polarization - y    :                0.868
            Polarization - z    :                0.496
            Touchable (pointer) :                  0x0
      G4ParticleChange::CheckIt  : the Momentum Change is not unit vector !!  Difference:  3.21264e-08



CPU Side photon flow to G4 
----------------------------

* NEmitPhotonsNPY::init forms the photons, writing them into an NPY<float> buffer

* CInputPhotonSource::convertPhoton gives them to G4

::

     g4-cls ThreeVector


::

     71 G4PrimaryVertex* CInputPhotonSource::convertPhoton(unsigned pho_index)
     72 {
     73     part_prop_t& pp = m_pp.Get();
     74 
     75     glm::vec4 post = m_pho->getPositionTime(pho_index) ;
     76     glm::vec4 dirw = m_pho->getDirectionWeight(pho_index) ;
     77     glm::vec4 polw = m_pho->getPolarizationWavelength(pho_index) ;
     78 
     79     pp.position.set(post.x, post.y, post.z);
     80     float time = post.w ;
     81 
     82     G4PrimaryVertex* vertex = new G4PrimaryVertex(pp.position, time );
     83 
     84     pp.momentum_direction.set(dirw.x, dirw.y ,dirw.z);
     85 





