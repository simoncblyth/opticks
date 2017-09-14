j1707 sensor hits
====================


ggeo-
-------

::

     315 GMaterial* GGeo::getCathode()
     316 {
     317     return m_cathode ;
     318 }
     319 void GGeo::setCathode(GMaterial* cathode)
     320 {
     321     m_cathode = cathode ;
     322 }
     323 
     324 void GGeo::addCathodeLV(const char* lv)
     325 {
     326    m_cathode_lv.insert(lv);
     327 }
     328 
     329 unsigned int GGeo::getNumCathodeLV()
     330 {
     331    return m_cathode_lv.size() ;
     332 }
     333 const char* GGeo::getCathodeLV(unsigned int index)
     334 {
     335     typedef std::unordered_set<std::string>::const_iterator UCI ;
     336     UCI it = m_cathode_lv.begin() ;
     337     std::advance( it, index );
     338     return it != m_cathode_lv.end() ? it->c_str() : NULL  ;
     339 }
     340 
     341 void GGeo::dumpCathodeLV(const char* msg)
     342 {
     343     printf("%s\n", msg);
     344     typedef std::unordered_set<std::string>::const_iterator UCI ;
     345     for(UCI it=m_cathode_lv.begin() ; it != m_cathode_lv.end() ; it++)
     346     {
     347         printf("GGeo::dumpCathodeLV %s \n", it->c_str() );
     348     }
     349 }


::

    delta:ggeo blyth$ opticks-f setCathode
    ./assimprap/AssimpGGeo.cc:                gg->setCathode(gmat) ;  
    ./ggeo/GGeo.cc:void GGeo::setCathode(GMaterial* cathode)
    ./ggeo/GGeo.hh:        void setCathode(GMaterial* cathode);
    delta:opticks blyth$ 


assimp
-------

::

    /usr/local/opticks/externals/assimp/assimp-fork/code

    delta:code blyth$ grep g4dae_ *.*
    ColladaLoader.cpp:            const char* prefix = "g4dae_" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_bordersurface_physvolume1 = "g4dae_bordersurface_physvolume1" ; 
    ColladaParser.cpp:const std::string ColladaParser::g4dae_bordersurface_physvolume2 = "g4dae_bordersurface_physvolume2" ; 
    ColladaParser.cpp:const std::string ColladaParser::g4dae_skinsurface_volume = "g4dae_skinsurface_volume" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_opticalsurface_name   = "g4dae_opticalsurface_name" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_opticalsurface_finish = "g4dae_opticalsurface_finish" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_opticalsurface_model  = "g4dae_opticalsurface_model" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_opticalsurface_type   = "g4dae_opticalsurface_type" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_opticalsurface_value  = "g4dae_opticalsurface_value" ;
    ColladaParser.cpp:    pProperties[g4dae_opticalsurface_name]   = pOpticalSurface.mName ; 
    ColladaParser.cpp:    pProperties[g4dae_opticalsurface_model]  = pOpticalSurface.mModel ; 
    ColladaParser.cpp:    pProperties[g4dae_opticalsurface_type]   = pOpticalSurface.mType ; 
    ColladaParser.cpp:    pProperties[g4dae_opticalsurface_finish] = pOpticalSurface.mFinish ; 
    ColladaParser.cpp:    pProperties[g4dae_opticalsurface_value]  = pOpticalSurface.mValue ; 
    ColladaParser.cpp:        pMaterial.mExtra->mProperties[g4dae_skinsurface_volume] = pSkinSurface.mVolume ; 
    ColladaParser.cpp:        pMaterial.mExtra->mProperties[g4dae_bordersurface_physvolume1] = pBorderSurface.mPhysVolume1 ; 
    ColladaParser.cpp:        pMaterial.mExtra->mProperties[g4dae_bordersurface_physvolume2] = pBorderSurface.mPhysVolume2 ; 
    ColladaParser.h:    static const std::string g4dae_bordersurface_physvolume1 ; 
    ColladaParser.h:    static const std::string g4dae_bordersurface_physvolume2 ;
    ColladaParser.h:    static const std::string g4dae_skinsurface_volume ;
    ColladaParser.h:    static const std::string g4dae_opticalsurface_name ;
    ColladaParser.h:    static const std::string g4dae_opticalsurface_finish ;
    ColladaParser.h:    static const std::string g4dae_opticalsurface_model ;
    ColladaParser.h:    static const std::string g4dae_opticalsurface_type ;
    ColladaParser.h:    static const std::string g4dae_opticalsurface_value ;
    delta:code blyth$ 



::

     24420     <material id="Steel0x14aa2a0">
     24421       <instance_effect url="#Steel_fx_0x14aa2a0"/>
     24422       <extra>
     24423         <matrix coldim="2" name="ABSLENGTH0x14aabd0">1.55e-06 0.001 6.2e-06 0.001 1.033e-05 0.001 1.55e-05 0.001</matrix>
     24424         <property name="ABSLENGTH" ref="ABSLENGTH0x14aabd0"/>
     24425       </extra>
     24426     </material>
     24427     <material id="Tyvek0x14a7890">
     24428       <instance_effect url="#Tyvek_fx_0x14a7890"/>
     24429       <extra>
     24430         <matrix coldim="2" name="ABSLENGTH0x14a7bf0">1.55e-06 10000 6.2e-06 10000 1.033e-05 10000 1.55e-05 10000</matrix>
     24431         <property name="ABSLENGTH" ref="ABSLENGTH0x14a7bf0"/>
     24432       </extra>
     24433     </material>


DYB .dae has EFFICIENCY property on the Bialkali material of the cathode::

    065902     <material id="__dd__Materials__Bialkali0xc2f2428">
     65903       <instance_effect url="#__dd__Materials__Bialkali_fx_0xc2f2428"/>
     65904       <extra>
     65905         <matrix coldim="2" name="ABSLENGTH0xc0b7a90">1.55e-06 0.0001 1.61e-06 500 2.07e-06 1000 2.48e-06 2000 3.56e-06 1000 4.13e-06 1000 6.2e-06 1000 1.033e-05 1000 1.55e-05 1000</matr       ix>
     65906         <property name="ABSLENGTH" ref="ABSLENGTH0xc0b7a90"/>
     65907         <matrix coldim="2" name="EFFICIENCY0xc2c6598">1.55e-06 0.0001 1.8e-06 0.002 1.9e-06 0.005 2e-06 0.01 2.05e-06 0.017 2.16e-06 0.03 2.19e-06 0.04 2.23e-06 0.05 2.27e-06 0.06 2.32e       -06 0.07 2.36e-06 0.08 2.41e-06 0.09 2.46e-06 0.1 2.5e-06 0.11 2.56e-06 0.13 2.61e-06 0.15 2.67e-06 0.16 2.72e-06 0.18 2.79e-06 0.19 2.85e-06 0.2 2.92e-06 0.21 2.99e-06 0.22 3.06e-06 0.       22 3.14e-06 0.23 3.22e-06 0.24 3.31e-06 0.24 3.4e-06 0.24 3.49e-06 0.23 3.59e-06 0.22 3.7e-06 0.21 3.81e-06 0.17 3.94e-06 0.14 4.07e-06 0.09 4.1e-06 0.035 4.4e-06 0.005 5e-06 0.001 6.2e       -06 0.0001 1.033e-05 0 1.55e-05 0</matrix>
     65908         <property name="EFFICIENCY" ref="EFFICIENCY0xc2c6598"/>
     65909         <matrix coldim="2" name="RINDEX0xc0fd260">1.55e-06 1.458 2.07e-06 1.458 4.13e-06 1.458 6.2e-06 1.458 1.033e-05 1.458 1.55e-05 1.458</matrix>
     65910         <property name="RINDEX" ref="RINDEX0xc0fd260"/>
     65911       </extra>
     65912     </material>

    152930       <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" type="0" value="1">
    152931         <matrix coldim="2" name="REFLECTIVITY0xc04f6a8">1.5e-06 0 6.5e-06 0</matrix>
    152932         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc04f6a8"/>
    152933         <matrix coldim="2" name="RINDEX0xc33da70">1.5e-06 0 6.5e-06 0</matrix>
    152934         <property name="RINDEX" ref="RINDEX0xc33da70"/>
    152935       </opticalsurface>
    152936       <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface" type="0" value="1">
    152937         <matrix coldim="2" name="BACKSCATTERCONSTANT0xc28d340">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    152938         <property name="BACKSCATTERCONSTANT" ref="BACKSCATTERCONSTANT0xc28d340"/>
    152939         <matrix coldim="2" name="REFLECTIVITY0xc563328">1.55e-06 0.0393 1.771e-06 0.0393 2.066e-06 0.0394 2.48e-06 0.03975 2.755e-06 0.04045 3.01e-06 0.04135 3.542e-06 0.0432 4.133e-06        0.04655 4.959e-06 0.0538 6.2e-06 0.067 1.033e-05 0.114 1.55e-05 0.173</matrix>
    152940         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc563328"/>
    152941         <matrix coldim="2" name="SPECULARLOBECONSTANT0xbfa85d0">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    152942         <property name="SPECULARLOBECONSTANT" ref="SPECULARLOBECONSTANT0xbfa85d0"/>
    152943         <matrix coldim="2" name="SPECULARSPIKECONSTANT0xc03fc20">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    152944         <property name="SPECULARSPIKECONSTANT" ref="SPECULARSPIKECONSTANT0xc03fc20"/>
    152945       </opticalsurface>
    152946       <opticalsurface finish="0" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" type="0" value="0">
    152947         <matrix coldim="2" name="REFLECTIVITY0xc359d00">1.55e-06 0.98505 1.63e-06 0.98406 1.68e-06 0.96723 1.72e-06 0.9702 1.77e-06 0.97119 1.82e-06 0.96624 1.88e-06 0.95139 1.94e-06 0.       98307 2e-06 0.9801 2.07e-06 0.98901 2.14e-06 0.98505 2.21e-06 0.96525 2.3e-06 0.97614 2.38e-06 0.97812 2.48e-06 0.97515 2.58e-06 0.96525 2.7e-06 0.96624 2.82e-06 0.96129 2.95e-06 0.9583       2 3.1e-06 0.95733 3.26e-06 0.73656 3.44e-06 0.11583 3.65e-06 0.10395 3.88e-06 0.11682 4.13e-06 0.14256 4.43e-06 0.1188 4.77e-06 0.18018 4.96e-06 0.21384 6.2e-06 0.0099 1.033e-05 0.0099        1.55e-05 0.0099</matrix>
    152948         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc359d00"/>
    152949       </opticalsurface>



JUNO .dae has EFFICIENCY on the opticalsurface of Photocathode_opsurf and Photocathode_opsurf_3inch

::

    854339     <extra>
    854340       <opticalsurface finish="3" model="1" name="UpperChimneyTyvekOpticalSurface" type="0" value="0.2">
    854341         <matrix coldim="2" name="REFLECTIVITY0x180de20">1.55e-06 0.1 6.2e-06 0.1 1.033e-05 0.1 1.55e-05 0.1</matrix>
    854342         <property name="REFLECTIVITY" ref="REFLECTIVITY0x180de20"/>
    854343       </opticalsurface>
    854344       <opticalsurface finish="0" model="0" name="Photocathode_opsurf" type="0" value="1">
    854345         <matrix coldim="2" name="EFFICIENCY0x14c0780">1.55e-06 0.002214 1.77143e-06 0.002214 1.7971e-06 0.003426 1.82353e-06 0.005284 1.85075e-06 0.007921 1.87879e-06 0.011425 1.90769e-       06 0.015808 1.9375e-06 0.021143 1.96825e-06 0.026877 2e-06 0.033344 2.03279e-06 0.040519 2.06667e-06 0.048834 2.10169e-06 0.057679 2.13793e-06 0.067843 2.17544e-06 0.079047 2.21429e-06        0.091286 2.25454e-06 0.104205 2.2963e-06 0.119611 2.33962e-06 0.135205 2.38462e-06 0.154528 2.43137e-06 0.17464 2.48e-06 0.194504 2.53061e-06 0.210267 2.58333e-06 0.223053 2.6383e-06 0.       234931 2.69565e-06 0.248108 2.75556e-06 0.26528 2.81818e-06 0.281478 2.88372e-06 0.293765 2.95238e-06 0.30198 3.02439e-06 0.302932 3.1e-06 0.303274 3.17949e-06 0.299854 3.26316e-06 0.28       5137 3.35135e-06 0.270132 3.44444e-06 0.252713 3.54286e-06 0.227767 3.64706e-06 0.192104 3.75758e-06 0.143197 3.875e-06 0.063755 4e-06 0.015229 4.13333e-06 0.007972 1.55e-05 1e-06</matr       ix>
    854346         <property name="EFFICIENCY" ref="EFFICIENCY0x14c0780"/>
    854347         <matrix coldim="2" name="KINDEX0x14c0520">1.55e-06 1.6 6.2e-06 1.6 1.033e-05 1.6 1.55e-05 1.6</matrix>
    854348         <property name="KINDEX" ref="KINDEX0x14c0520"/>
    854349         <matrix coldim="2" name="REFLECTIVITY0x14c0630">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    854350         <property name="REFLECTIVITY" ref="REFLECTIVITY0x14c0630"/>
    854351         <matrix coldim="2" name="RINDEX0x14c0470">1.55e-06 2.9 6.2e-06 2.9 1.033e-05 2.9 1.55e-05 2.9</matrix>
    854352         <property name="RINDEX" ref="RINDEX0x14c0470"/>
    854353         <matrix coldim="2" name="THICKNESS0x14c1000">0 2.6e-05 375 2.6e-05</matrix>
    854354         <property name="THICKNESS" ref="THICKNESS0x14c1000"/>
    854355       </opticalsurface>
    854356       <opticalsurface finish="1" model="0" name="Mirror_opsurf" type="0" value="0.999">
    854357         <matrix coldim="2" name="REFLECTIVITY0x17feaf0">1.55e-06 0.9999 1.55e-05 0.9999</matrix>
    854358         <property name="REFLECTIVITY" ref="REFLECTIVITY0x17feaf0"/>
    854359       </opticalsurface>
    854360       <opticalsurface finish="0" model="0" name="Photocathode_opsurf_3inch" type="0" value="1">
    854361         <matrix coldim="2" name="EFFICIENCY0x14c1b70">1.55e-06 1e-05 1.737e-06 0.00159 1.769e-06 0.00255 1.791e-06 0.00355 1.808e-06 0.00469 1.825e-06 0.00605 1.844e-06 0.00774 1.864e-0       6 0.01003 1.884e-06 0.01325 1.904e-06 0.01718 1.923e-06 0.02059 1.947e-06 0.02608 1.978e-06 0.03229 2.008e-06 0.0396 2.041e-06 0.0479 2.069e-06 0.0548 2.104e-06 0.06387 2.141e-06 0.0779       7 2.174e-06 0.09129 2.211e-06 0.10541 2.251e-06 0.12003 2.303e-06 0.13668 2.361e-06 0.15564 2.41e-06 0.17078 2.462e-06 0.19267 2.522e-06 0.21437 2.595e-06 0.23089 2.675e-06 0.24073 2.77       1e-06 0.24868 2.857e-06 0.24983 2.954e-06 0.24753 3.04e-06 0.24185 3.147e-06 0.23304 3.248e-06 0.22351 3.355e-06 0.20848 3.482e-06 0.19001 3.594e-06 0.1692 3.661e-06 0.14451 3.744e-06 0       .12059 3.78e-06 0.09924 3.831e-06 0.07906 3.868e-06 0.06154 3.912e-06 0.04971 3.956e-06 0.0396 4.002e-06 0.03126 4.043e-06 0.02525 4.09e-06 0.01894 4.122e-06 0.01516 4.161e-06 0.01185 4       .194e-06 0.00893 4.222e-06 0.0067 4.251e-06 0.00521 4.286e-06 0.004 4.315e-06 0.00307 4.363e-06 0.00229 4.394e-06 0.00181 4.437e-06 0.00137 6.2e-06 1e-05 1.033e-05 1e-05 1.55e-05 1e-05<       /matrix>
    854362         <property name="EFFICIENCY" ref="EFFICIENCY0x14c1b70"/>
    854363         <matrix coldim="2" name="KINDEX0x14c1910">1.55e-06 1.6 6.2e-06 1.6 1.033e-05 1.6 1.55e-05 1.6</matrix>
    854364         <property name="KINDEX" ref="KINDEX0x14c1910"/>
    854365         <matrix coldim="2" name="REFLECTIVITY0x14c1a20">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    854366         <property name="REFLECTIVITY" ref="REFLECTIVITY0x14c1a20"/>
    854367         <matrix coldim="2" name="RINDEX0x14c1860">1.55e-06 2.9 6.2e-06 2.9 1.033e-05 2.9 1.55e-05 2.9</matrix>
    854368         <property name="RINDEX" ref="RINDEX0x14c1860"/>
    854369         <matrix coldim="2" name="THICKNESS0x14c2610">0 2.6e-05 375 2.6e-05</matrix>
    854370         <property name="THICKNESS" ref="THICKNESS0x14c2610"/>
    854371       </opticalsurface>
    854372       <opticalsurface finish="0" model="0" name="Absorb_opsurf" type="0" value="1">
    854373         <matrix coldim="2" name="REFLECTIVITY0x18374e0">1.55e-06 0 1.55e-05 0</matrix>
    854374         <property name="REFLECTIVITY" ref="REFLECTIVITY0x18374e0"/>
    854375       </opticalsurface>



JUNO .gdml solids have associated opticalsurface but no properties

::

   243   <solids>
   ...
   358     <intersection name="PMT_20inch_inner1_solid0x1814a90">
   359       <first ref="PMT_20inch_inner_solid0x1814800"/>
   360       <second ref="Inner_Separator0x1814990"/>
   361       <position name="PMT_20inch_inner1_solid0x1814a90_pos" unit="mm" x="0" y="0" z="91.999999999"/>
   362     </intersection>
   363     <opticalsurface finish="0" model="0" name="Photocathode_opsurf" type="0" value="1"/>
   364     <subtraction name="PMT_20inch_inner2_solid0x1863010">
   365       <first ref="PMT_20inch_inner_solid0x1814800"/>
   366       <second ref="Inner_Separator0x1814990"/>
   367       <position name="PMT_20inch_inner2_solid0x1863010_pos" unit="mm" x="0" y="0" z="91.999999999"/>
   368     </subtraction>
   369     <opticalsurface finish="1" model="0" name="Mirror_opsurf" type="0" value="0.999"/>


   412     <ellipsoid ax="38" by="38" cz="22" lunit="mm" name="PMT_3inch_inner1_solid_ell_helper0x1c9e510" zcut1="7.04319871929267" zcut2="22"/>
   413     <opticalsurface finish="0" model="0" name="Photocathode_opsurf_3inch" type="0" value="1"/>
   414     <ellipsoid ax="38" by="38" cz="22" lunit="mm" name="PMT_3inch_inner2_solid_ell_helper0x1c9e5d0" zcut1="-15.8745078663875" zcut2="7.04319871929267"/>
   415     <opticalsurface finish="0" model="0" name="Absorb_opsurf" type="0" value="1"/>



::

    277146     <skinsurface name="Tube_surf" surfaceproperty="TubeSurface">
    277147       <volumeref ref="lSurftube0x254b8d0"/>
    277148     </skinsurface>
    277149     <bordersurface name="UpperChimneyTyvekSurface" surfaceproperty="UpperChimneyTyvekOpticalSurface">
    277150       <physvolref ref="pUpperChimneyLS0x2547680"/>
    277151       <physvolref ref="pUpperChimneyTyvek0x2547de0"/>
    277152     </bordersurface>
    277153     <bordersurface name="PMT_20inch_photocathode_logsurf1" surfaceproperty="Photocathode_opsurf">
    277154       <physvolref ref="PMT_20inch_inner1_phys0x18012e0"/>
    277155       <physvolref ref="PMT_20inch_body_phys0xe4d580"/>
    277156     </bordersurface>
    277157     <bordersurface name="PMT_20inch_mirror_logsurf1" surfaceproperty="Mirror_opsurf">
    277158       <physvolref ref="PMT_20inch_inner2_phys0x1821730"/>
    277159       <physvolref ref="PMT_20inch_body_phys0xe4d580"/>
    277160     </bordersurface>
    277161     <bordersurface name="PMT_20inch_photocathode_logsurf2" surfaceproperty="Photocathode_opsurf">
    277162       <physvolref ref="PMT_20inch_body_phys0xe4d580"/>
    277163       <physvolref ref="PMT_20inch_inner1_phys0x18012e0"/>
    277164     </bordersurface>
    277165     <bordersurface name="PMT_3inch_photocathode_logsurf1" surfaceproperty="Photocathode_opsurf_3inch">
    277166       <physvolref ref="PMT_3inch_inner1_phys0x1c9f370"/>
    277167       <physvolref ref="PMT_3inch_body_phys0x1c9f2c0"/>
    277168     </bordersurface>
    277169     <bordersurface name="PMT_3inch_absorb_logsurf1" surfaceproperty="Absorb_opsurf">
    277170       <physvolref ref="PMT_3inch_inner2_phys0x1c9f420"/>
    277171       <physvolref ref="PMT_3inch_body_phys0x1c9f2c0"/>
    277172     </bordersurface>
    277173     <bordersurface name="PMT_3inch_photocathode_logsurf2" surfaceproperty="Photocathode_opsurf_3inch">
    277174       <physvolref ref="PMT_3inch_body_phys0x1c9f2c0"/>
    277175       <physvolref ref="PMT_3inch_inner1_phys0x1c9f370"/>
    277176     </bordersurface>
    277177     <bordersurface name="PMT_3inch_absorb_logsurf3" surfaceproperty="Absorb_opsurf">
    277178       <physvolref ref="PMT_3inch_cntr_phys0x1c9f4d0"/>
    277179       <physvolref ref="PMT_3inch_body_phys0x1c9f2c0"/>
    277180     </bordersurface>
    277181     <bordersurface name="ChimneyTyvekSurface" surfaceproperty="ChimneyTyvekOpticalSurface">
    277182       <physvolref ref="pLowerChimneyAcrylic0x254afc0"/>
    277183       <physvolref ref="pLowerChimneyTyvek0x254af30"/>
    277184     </bordersurface>
    277185     <bordersurface name="CDTyvekSurface" surfaceproperty="CDTyvekOpticalSurface">
    277186       <physvolref ref="pOuterWaterPool0x14dba40"/>
    277187       <physvolref ref="pCentralDetector0x14ddb50"/>
    277188     </bordersurface>
    277189   </structure>








    2556 void ColladaParser::addCommonOpticalSurfaceProperties( Collada::ExtraProperties::ExtraPropertiesMap& pProperties, Collada::OpticalSurface& pOpticalSurface )
    2557 {
    2558     pProperties[g4dae_opticalsurface_name]   = pOpticalSurface.mName ;
    2559     pProperties[g4dae_opticalsurface_model]  = pOpticalSurface.mModel ;
    2560     pProperties[g4dae_opticalsurface_type]   = pOpticalSurface.mType ;
    2561     pProperties[g4dae_opticalsurface_finish] = pOpticalSurface.mFinish ;
    2562     pProperties[g4dae_opticalsurface_value]  = pOpticalSurface.mValue ;
    2563 }
    2564 
    2565 
    2566 void ColladaParser::FakeExtraSkinSurface(Collada::SkinSurface& pSkinSurface,  Collada::Material& pMaterial)
    2567 {
    2568     // hijack Assimp material infrastructure to hold skin surface properties
    2569     if(!pMaterial.mExtra )
    2570         pMaterial.mExtra = new Collada::ExtraProperties();
    2571 
    2572     if(pSkinSurface.mOpticalSurface)
    2573     {
    2574         std::map<std::string,std::string>& ssm = pSkinSurface.mOpticalSurface->mExtra->mProperties ;
    2575         pMaterial.mExtra->mProperties.insert( ssm.begin(), ssm.end() );
    2576         pMaterial.mExtra->mProperties[g4dae_skinsurface_volume] = pSkinSurface.mVolume ;
    2577 
    2578         addCommonOpticalSurfaceProperties( pMaterial.mExtra->mProperties , *pSkinSurface.mOpticalSurface);
    2579     }
    2580 }
    2581 
    ....
    2642 void ColladaParser::FakeExtraBorderSurface(Collada::BorderSurface& pBorderSurface, Collada::Material& pMaterial)
    2643 {
    2644     // hijack Assimp material infrastructure to hold skin surface properties
    2645     if(!pMaterial.mExtra )
    2646         pMaterial.mExtra = new Collada::ExtraProperties();
    2647 
    2648     if(pBorderSurface.mOpticalSurface)
    2649     {
    2650         std::map<std::string,std::string>& bsm = pBorderSurface.mOpticalSurface->mExtra->mProperties ;
    2651         pMaterial.mExtra->mProperties.insert( bsm.begin(), bsm.end() );
    2652         pMaterial.mExtra->mProperties[g4dae_bordersurface_physvolume1] = pBorderSurface.mPhysVolume1 ;
    2653         pMaterial.mExtra->mProperties[g4dae_bordersurface_physvolume2] = pBorderSurface.mPhysVolume2 ;
    2654 
    2655         addCommonOpticalSurfaceProperties( pMaterial.mExtra->mProperties, *pBorderSurface.mOpticalSurface);
    2656     }
    2657 }






assimprap- 
------------


Hmm below setCathode is DYB specific, assuming cathode material 


::

     368 void AssimpGGeo::convertMaterials(const aiScene* scene, GGeo* gg, const char* query )
     369 {
     370     LOG(info)<<"AssimpGGeo::convertMaterials "
     371              << " query " << query
     372              << " mNumMaterials " << scene->mNumMaterials
     373              ;
     374 
     ...
     379     for(unsigned int i = 0; i < scene->mNumMaterials; i++)
     380     {
     381         unsigned int index = i ;  // hmm, make 1-based later 
     382 
     383         aiMaterial* mat = scene->mMaterials[i] ;
     384 
     385         aiString name_;
     386         mat->Get(AI_MATKEY_NAME, name_);
     387 
     388         const char* name = name_.C_Str();
     389 
     390         //if(strncmp(query, name, strlen(query))!=0) continue ;  
     391 
     392         LOG(debug) << "AssimpGGeo::convertMaterials " << i << " " << name ;
     393 
     394         const char* bspv1 = getStringProperty(mat, g4dae_bordersurface_physvolume1 );
     395         const char* bspv2 = getStringProperty(mat, g4dae_bordersurface_physvolume2 );
     396 
     397         const char* sslv  = getStringProperty(mat, g4dae_skinsurface_volume );
     398 
     399         const char* osnam = getStringProperty(mat, g4dae_opticalsurface_name );
     400         const char* ostyp = getStringProperty(mat, g4dae_opticalsurface_type );
     401         const char* osmod = getStringProperty(mat, g4dae_opticalsurface_model );
     402         const char* osfin = getStringProperty(mat, g4dae_opticalsurface_finish );
     403         const char* osval = getStringProperty(mat, g4dae_opticalsurface_value );
     404 

     ...
     422         if( sslv )
     423         {
     424             assert(os && "all ss must have associated os");
     425 
     426             GSkinSurface* gss = new GSkinSurface(name, index, os);
     ... 
     449         }
     450         else if (bspv1 && bspv2 )
     451         {
     452             assert(os && "all bs must have associated os");
     453             GBorderSurface* gbs = new GBorderSurface(name, index, os);
     ...
     471         else
     472         {
     473             assert(os==NULL);
     474 
     475 
     476             //printf("AssimpGGeo::convertMaterials aiScene materialIndex %u (GMaterial) name %s \n", i, name);
     477             GMaterial* gmat = new GMaterial(name, index);
     478             gmat->setStandardDomain(standard_domain);
     479             addProperties(gmat, mat );
     480             gg->add(gmat);
     481 
     482             {
     483                 // without standard domain applied
     484                 GMaterial* gmat_raw = new GMaterial(name, index);
     485                 addProperties(gmat_raw, mat );
     486                 gg->addRaw(gmat_raw);
     487             }
     488 
     489             if(hasVectorProperty(mat, EFFICIENCY ))
     490             {
     491                 assert(gg->getCathode() == NULL && "only expecting one material with an EFFICIENCY property" );
     492                 gg->setCathode(gmat) ;
     493                 m_cathode = mat ;
     494             }
     495 
     496         }
        







::

     517 void AssimpGGeo::convertSensors(GGeo* gg)
     518 {
     519 /*
     520 Opticks is a surface based simulation, as opposed to 
     521 Geant4 which is CSG volume based. In Geant4 hits are formed 
     522 on stepping into volumes with associated SensDet.
     523 The Opticks equivalent is intersecting with a "SensorSurface", 
     524 which are fabricated by AssimpGGeo::convertSensors.
     525 */
     526     convertSensors( gg, m_tree->getRoot(), 0);
     527 
     528     //assert(m_cathode);
     529     if(!m_cathode)
     530     {
     531          LOG(warning) << "AssimpGGeo::convertSensors m_cathode NULL : no material with an efficiency property ?  " ;
     532          return ;
     533     }
     534 
     535     unsigned int nclv = gg->getNumCathodeLV();
     536 
     537 
     538     LOG(info) << "AssimpGGeo::convertSensors"
     539               << " nclv " << nclv
     540               ;
     541 
     542     GDomain<float>* standard_domain = gg->getBndLib()->getStandardDomain();
     543 
     544     // DYB: nclv=2 for hemi and headon PMTs 
     545     for(unsigned int i=0 ; i < nclv ; i++)
     546     {
     547         const char* sslv = gg->getCathodeLV(i);
     548         LOG(info) << "AssimpGGeo::convertSensors"
     549                   << " i " << i
     550                   << " sslv " << sslv
     551                   ;
     552 
     553         std::string name = BStr::trimPointerSuffixPrefix(sslv, NULL );
     554         name += GSurfaceLib::SENSOR_SURFACE ;


         36 
         37 const char* GSurfaceLib::REFLECTIVITY = "REFLECTIVITY" ;
         38 const char* GSurfaceLib::EFFICIENCY   = "EFFICIENCY" ;
         39 const char* GSurfaceLib::SENSOR_SURFACE = "SensorSurface" ;
         40 


     555 
     556         const char* osnam = name.c_str() ;
     557         const char* ostyp = "0" ;
     558         const char* osmod = "1" ;
     559         const char* osfin = "3" ;
     560         const char* osval = "1" ;
     561 
     562         // TODO: check effects of above adhoc choice of common type/model/finish/value 
     563         // TODO: add parse ctor that understands: "type=dielectric_dielectric;model=unified;finish=ground;value=1.0"
     564 
     565         GOpticalSurface* os = new GOpticalSurface(osnam, ostyp, osmod, osfin, osval) ;
     566 
     567         // standard materials/surfaces use the originating aiMaterial index, 
     568         // extend that for fake SensorSurface by toting up all 
     569 
     570         unsigned int index = gg->getNumMaterials() + gg->getNumSkinSurfaces() + gg->getNumBorderSurfaces() ;
     571 
     572 
     573         GSkinSurface* gss = new GSkinSurface(name.c_str(), index, os);
     574 
     575         gss->setStandardDomain(standard_domain);
     576       
     577         gss->setSkinSurface(sslv);
     578 
     579         gss->setSensor();
     580         // story continues in GBoundaryLib::standardizeSurfaceProperties
     581         // that no longer exists, now probably GSurfaceLib::getSensorSurface
     582        //
     583 
     584         addProperties(gss, m_cathode );
     585 
     586         LOG(info) << "AssimpGGeo::convertSensors gss " << gss->description();
     587 
     588         gg->add(gss);
     589 
     590         {
     591             // without standard domain applied
     592             GSkinSurface*  gss_raw = new GSkinSurface(name.c_str(), index, os);
     593             gss_raw->setSkinSurface(sslv);
     594             // not setting sensor, only the standardized need that
     595             addProperties(gss_raw, m_cathode );
     596             gg->addRaw(gss_raw);
     597         }  
     598     }
     599 }


     601 void AssimpGGeo::convertSensors(GGeo* gg, AssimpNode* node, unsigned int depth)
     602 {
     603     // addCathodeLV into gg
     604     convertSensorsVisit(gg, node, depth);
     605     for(unsigned int i = 0; i < node->getNumChildren(); i++) convertSensors(gg, node->getChild(i), depth + 1);
     606 }
     ...
     ...
     608 void AssimpGGeo::convertSensorsVisit(GGeo* gg, AssimpNode* node, unsigned int depth)
     609 {
     610     // collects lv of nodes of cathode material allowing construction 
     611     // of "fake" GSkinSurface
     612     //
     613     // NB border surface sensors at not handled, as there are non of those in DYB
     614     //
     615 
     616     unsigned int nodeIndex = node->getIndex();
     617 
     618     const char* lv   = node->getName(0);
     619 
     620     const char* pv   = node->getName(1);
     621 
     622     unsigned int mti = node->getMaterialIndex() ;
     623 
     624     GMaterial* mt = gg->getMaterial(mti);
     625    
     626     NSensorList* sens = gg->getSensorList();
     627     /*
     628     NSensor* sensor0 = sens->getSensor( nodeIndex ); 
     629     NSensor* sensor1 = sens->findSensorForNode( nodeIndex ); 
     630     assert(sensor0 == sensor1);
     631     // these do not match
     632     */
     633     NSensor* sensor = sens->findSensorForNode( nodeIndex );
     634 
     635     if(sensor && mt == gg->getCathode())
     636     {
     637          LOG(debug) << "AssimpGGeo::convertSensorsVisit "
     638                    << " depth " << depth
     639                    << " lv " << lv
     640                    << " pv " << pv
     641                    ;
     642          gg->addCathodeLV(lv) ;
     643     }
     644 }



