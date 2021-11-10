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


Possible cause of why --skipsolidname not working
-----------------------------------------------------

* skip logic only in GInstancer::labelRepeats_r and not in GInstancer::labelGlobals_r

::

    /**
    GInstancer::labelGlobals_r
    -------------------------------

    Only recurses whilst in global territory with ridx == 0, as soon as hit a repeated 
    volume, with ridx > 0, stop recursing. 

    Skipping of an instanced LV is done here by setting a flag.

    Currently trying to skip a global lv at this rather late juncture 
    leads to inconsistencies manifesting in a corrupted color buffer 
    (i recall that all global volumes are retained for index consistency in the merge of GMergedMesh GGeoLib)
    so moved to simply editing the input GDML
    Presumably could also do this by moving the skip earlier to the Geant4 X4 traverse
    see notes/issues/torus_replacement_on_the_fly.rst


    **/

    void GInstancer::labelGlobals_r( GNode* node, unsigned depth )
    {
        unsigned ridx = node->getRepeatIndex() ; 
        if( ridx > 0 ) return ; 
        assert( ridx == 0 );  

        unsigned pidx = 0 ; 
        unsigned oidx = m_globals_count ; 
        unsigned triplet_identity = OpticksIdentity::Encode(ridx, pidx, oidx); 
        node->setTripletIdentity( triplet_identity );  
     
        m_globals_count += 1 ; 

        unsigned lvidx = node->getMeshIndex();  
        m_meshset[ridx].insert( lvidx ) ; 

    /*
        if( m_ok->isCSGSkipLV(lvidx) )   // --csgskiplv
        {
            assert(0 && "skipping of LV used globally, ie non-instanced, is not currently working "); 

            GVolume* vol = dynamic_cast<GVolume*>(node); 
            vol->setCSGSkip(true);      

            m_csgskiplv[lvidx].push_back( node->getIndex() ); 
            m_csgskiplv_count += 1 ; 
        }
    */  
     
        for(unsigned i = 0; i < node->getNumChildren(); i++) labelGlobals_r(node->getChild(i), depth + 1 );
    }





