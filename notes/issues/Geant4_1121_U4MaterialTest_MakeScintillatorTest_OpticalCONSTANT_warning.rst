Geant4_1121_U4MaterialTest_MakeScintillatorTest_OpticalCONSTANT_warning
=========================================================================


::

    [blyth@localhost tests]$ ./U4MaterialTest.sh 
    $Name: geant4-11-02-patch-01 [MT]$ (16-February-2024)
    TEST[MakeScintillator]

    -------- WWWW ------- G4Exception-START -------- WWWW -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat219
          issued by : G4MaterialPropertiesTable::AddProperty()
    AddProperty warning. A material property vector must have more than one value.
    Unless you will later add an entry, this is an error.
    Property name: OpticalCONSTANT
    *** This is just a warning message. ***
    -------- WWWW ------- G4Exception-END -------- WWWW -------



::




    [blyth@localhost LS]$ f
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
    f

    CMDLINE:/home/blyth/np/f.py
    f.base:.

      : f.NPFold_index                                     :                (19,) : 130 days, 0:10:20.320370 
      : f.RINDEX                                           :              (18, 2) : 130 days, 0:10:20.320370 
      : f.GROUPVEL                                         :              (18, 2) : 130 days, 0:10:20.320370 
      : f.RAYLEIGH                                         :              (11, 2) : 130 days, 0:10:20.320370 
      : f.ABSLENGTH                                        :             (497, 2) : 130 days, 0:10:20.320370 
      : f.FASTCOMPONENT                                    :             (275, 2) : 130 days, 0:10:20.320370 
      : f.SLOWCOMPONENT                                    :             (275, 2) : 130 days, 0:10:20.320370 
      : f.REEMISSIONPROB                                   :              (28, 2) : 130 days, 0:10:20.320370 
      : f.OpticalCONSTANT                                  :               (1, 2) : 130 days, 0:10:20.320370 
      : f.GammaCONSTANT                                    :               (4, 2) : 130 days, 0:10:20.319369 
      : f.AlphaCONSTANT                                    :               (4, 2) : 130 days, 0:10:20.319369 
      : f.NeutronCONSTANT                                  :               (4, 2) : 130 days, 0:10:20.319369 
      : f.PPOABSLENGTH                                     :             (770, 2) : 130 days, 0:10:20.319369 
      : f.PPOREEMISSIONPROB                                :              (15, 2) : 130 days, 0:10:20.319369 
      : f.PPOCOMPONENT                                     :             (200, 2) : 130 days, 0:10:20.319369 
      : f.PPOTIMECONSTANT                                  :               (2, 2) : 130 days, 0:10:20.319369 
      : f.bisMSBABSLENGTH                                  :             (375, 2) : 130 days, 0:10:20.319369 
      : f.bisMSBREEMISSIONPROB                             :              (23, 2) : 130 days, 0:10:20.319369 
      : f.bisMSBCOMPONENT                                  :             (275, 2) : 130 days, 0:10:20.319369 
      : f.bisMSBTIMECONSTANT                               :               (2, 2) : 130 days, 0:10:20.319369 
      : f.NPFold_names                                     :                 (0,) : 130 days, 0:10:20.319369 

     min_stamp : 2023-12-16 21:37:08.441959 
     max_stamp : 2023-12-16 21:37:08.442960 
     dif_stamp : 0:00:00.001001 
     age_stamp : 130 days, 0:10:20.319369 

    In [1]: f.OpticalCONSTANT
    Out[1]: array([[1.5, 1. ]])




Looks like this warning should be fixed following geom update
---------------------------------------------------------------


* https://code.ihep.ac.cn/JUNO/offline/junosw/-/merge_requests/393?commit_id=ca25a86b17fa1c2a987fe459ee19a16e73b1394e
* https://code.ihep.ac.cn/JUNO/offline/dbdata/-/merge_requests/20
* https://code.ihep.ac.cn/JUNO/offline/dbdata/-/blob/523ada6c3ddfd9c487821a7e28c9fe26ee4d3f47/offline-data/Simulation/DetSim/Material/LS/OpticalCONSTANT

::

    1.50  *ns  1.0
    1.50  *ns  0.0



::

    epsilon:~ blyth$ jgr OpticalCONSTANT
    ./Simulation/DetSimV2/PhysiSim/src/DsG4ScintSimple.cc:      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("OpticalCONSTANT");
    ./Simulation/DetSimV2/PhysiSim/src/DsG4Scintillation.cc:      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("OpticalCONSTANT");
    ./Simulation/DetSimV2/DetSimOptions/src/LSExpDetectorConstructionMaterial.icc:        helper_mpt(LSMPT, "OpticalCONSTANT",         mcgt.data(), "Material.LS.OpticalCONSTANT");
    ./Simulation/DetSimV2/AnalysisCode/src/OpticalParameterAnaMgr.cc:        get_matprop(tbl_LS, "OpticalCONSTANT", LS_OpticalCon_n, LS_OpticalCon_time, LS_OpticalCon_ratio);
    epsilon:junosw blyth$ 


