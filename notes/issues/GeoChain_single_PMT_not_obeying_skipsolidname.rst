GeoChain_single_PMT_not_obeying_skipsolidname
================================================

Issue : cxs render shows outer PMT solid only that appears to not have the horizontals
----------------------------------------------------------------------------------------

* assume cause is a failure to skip the degenerate body solid
  due to the highly abnormal GeoChain single PMT only geometry 

::

    # build PMTSim lib, providing standalone PV creation 
    jps    # cd ~/j/PMTSim
    om

    # build GeoChain which links with PMTSIm lib 
    gc     # cd ~/opticks/GeoChain
    om

    # run GeoChain with GeoChainVolumeTest creating the PV and running it through the conversions
    ./run.sh 

    # cxs 2d intersect render
    cx 
    ./b7
    ./cxs.sh 

    # grap intersects and display locally 
    laptop> cx ; ./grab.sh ; ./cxs.sh 



Possible nudge issue with body_phys
-------------------------------------

* looks like an equatorial sombrero 

::

   gc
   ./run.sh   # volume test with body_phys 

   cx
   om
   ./cxr_geochain.sh   # with body_phys    




Possible cause of why --skipsolidname not working
-----------------------------------------------------

* skip logic only in GInstancer::labelRepeats_r and not in GInstancer::labelGlobals_r


* moved skipping logic in GInstancer into GInstancer::visitNode so can 
  call from labelRepeat_r or labelGlobals_r however the notes in 
  why cannot do global level solid skips at such a late stage seem to 
  suggest its not worth pursuing. 

* BUT considering alternatives GInstancer seems like the natural place to skip
  because the volumes are already partitioned there 

* instead look at X4PhysicalVolume::convertStructure that grabs the 
  GMesh created in X4PhysicalVolume::convertSolid 

  * but the natural way to do things there is to set a flag on the GMesh 
    which is used from the GNode, which boils down to the same skip in 
    GInstancer... so need to face whats going wrong with global skips 
   





