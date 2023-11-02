blackbody_revival
===================

Review how to revive 
---------------------

Old way uses lots of code doing not much... 
Need a direct way. 

::

    epsilon:opticks blyth$ opticks-fl blackbody 
    ./ana/twhite.py
         informative comment : but why use 2D texture for the icdf ? 
         could just use 1D ? 

    ./cfg4/cfg4.bash
    ./ggeo/GAry.hh
    ./ggeo/GAry.cc
          planck_spectral_radiance, cie_weight, cie_X, cie_Y, cie_Z
          based on NPlanck.hpp NCIE.hpp

    ./ggeo/GProperty.hh
    ./ggeo/GProperty.cc

    ./ggeo/GSource.hh
    ./ggeo/GSource.cc

    ./ggeo/tests/GSourceTest.cc
          GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );

    ./ggeo/GSourceLib.hh
    ./ggeo/GSourceLib.cc

        095 void GSourceLib::generateBlackBodySample(unsigned int n)
         96 {
         97     GSource* bbs = GSource::make_blackbody_source("D65", 0, 6500.f );
         98     GProperty<double>* icdf = constructInvertedSourceCDF(bbs);
         99     GAry<double>* sample = icdf->lookupCDF(n);
        100     sample->save("$TMP/ggeo/GSourceLib/blackbody.npy");
        101 }
        102 

    ./ggeo/tests/GSourceLibTest.cc

     




    ./optickscore/okc.bash
    epsilon:opticks blyth$ 


Need real water RINDEX for rainbows too
------------------------------------------

::

    epsilon:opticks blyth$ opticks-f H2OHale
    ./ana/treflect.py:    boundary = Boundary("Vacuum///MainH2OHale")
    ./ana/trainbow.py:    boundary = Boundary("Vacuum///MainH2OHale")
    ./ana/bnd.py:    Out[13]: (38, 2, 39, 4)   # 38 materials, 2 groups, 39 wavelengths, 4 qwns in each group : 38 mats include 2 added ones: GlassSchottF2, MainH2OHale -> 36 standard ones
    ./ana/bnd.py:        ## huh m0.names shows that GlassSchottF2 appears in the middle, not at the tail ???  MainH2OHale is last 
    ./ana/bnd.py:    assert np.all( b0m.data == m0.data[~np.logical_or(m0.names == 'GlassSchottF2', m0.names == 'MainH2OHale')] )
    ./ana/droplet.py:    boundary = Boundary("Vacuum///MainH2OHale")
    ./ana/xrainbow.py:    boundary = Boundary("Vacuum///MainH2OHale") 
    ./ana/sphere.py:    boundary = Boundary("Vacuum///MainH2OHale")
    ./cfg4/CMaterialLib.cc:    if(strcmp(name,"MainH2OHale")==0)
    ./bin/ggv.bash:    local material=MainH2OHale
    ./integration/tests/tbox.bash:#tbox-m2(){ echo MainH2OHale ; }
    ./integration/tests/treflect.bash:treflect-material(){ echo MainH2OHale ; }
    ./integration/tests/tboolean.bash:#tboolean-material(){ echo MainH2OHale ; }
    ./integration/tests/tboolean.bash:#material = "MainH2OHale"
    ./integration/tests/trainbow.bash:    local material=MainH2OHale
    ./extg4/X4MaterialLib.cc:1. 2 extra OK materials (GlassSchottF2, MainH2OHale)  : the test glass comes after Air in the middle 
    ./ggeo/GMaterialLib.cc:    rix.push_back(SS("MainH2OHale",   "$OPTICKS_INSTALL_PREFIX/opticksaux/refractiveindex/tmp/main/H2O/Hale.npy"));
    epsilon:opticks blyth$ 




