Material Review
==================


test materials
----------------

::

    746 void GMaterialLib::addTestMaterials()
    747 {
    748     typedef std::pair<std::string, std::string> SS ;
    749     typedef std::vector<SS> VSS ;
    750 
    751     VSS rix ;
    752 
    753     rix.push_back(SS("GlassSchottF2", "$OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/glass/schott/F2.npy"));
    754     rix.push_back(SS("MainH2OHale",   "$OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/main/H2O/Hale.npy"));
    755     
    756     // NB when adding test materials also need to set in prefs ~/.opticks/GMaterialLib
    757     //
    758     //    * priority order (for transparent materials arrange to be less than 16 for material sequence tracking)
    759     //    * color 
    760     //    * two letter abbreviation
    761     //
    762     // for these settings to be acted upon must rebuild the geocache with : "ggv -G"      
    763     //

::

    151 const G4Material* CMaterialLib::convertMaterial(const GMaterial* kmat)
    152 {
    159     const char* name = kmat->getShortName();
    160     const G4Material* prior = getG4Material(name) ;
    161     if(prior)
    162     {
    169         return prior ;
    170     }
    173     unsigned int materialIndex = m_mlib->getMaterialIndex(kmat);
    174 
    175     G4String sname = name ;
    182 
    183     G4Material* material(NULL);
    184     if(strcmp(name,"MainH2OHale")==0)
    185     {
    186         material = makeWater(name) ;
    187     }
    188     else
    189     {
    190         G4double z, a, density ;
    191         // presumably z, a and density are not relevant for optical photons 
    192         material = new G4Material(sname, z=1., a=1.01*g/mole, density=universe_mean_density );
    193     }
    198     G4MaterialPropertiesTable* mpt = makeMaterialPropertiesTable(kmat);
    199     material->SetMaterialPropertiesTable(mpt);
    200 
    201     m_ggtog4[kmat] = material ;
    202     m_g4mat[name] = material ;   // used by getG4Material(shortname) 
    203 


CMaterialLibTest : does conversions
---------------------------------------

::

    op --cmat




