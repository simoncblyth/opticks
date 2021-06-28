raw_scintillator_material_props FIXED
=======================================

ana/ScintillatorLib.py shows 20nm bin effect on generated wavelength samples
-------------------------------------------------------------------------------

* FIXED by avoiding raw material standardizing, also moved
  to double precision surface and material props only narrowed to 
  float immediately prior to texture creation

* BUT: subsequently looks like OK surfaces messed up

* next :doc:`OK_lacking_SD_SA_following_prop_shift`


Inputs to GScintillatorLib need to be raw materials without standardization
------------------------------------------------------------------------------

* suspect inadventant domain standardization somewhere

::

    1240 void GGeo::prepareScintillatorLib()
    1241 {
    1242     LOG(verbose) << "GGeo::prepareScintillatorLib " ;
    1243 
    1244     findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB");
    1245 
    1246     unsigned int nscint = getNumScintillatorMaterials() ;
    1247 
    1248     if(nscint == 0)
    1249     {
    1250         LOG(LEVEL) << " found no scintillator materials  " ;
    1251     }
    1252     else
    1253     {
    1254         LOG(LEVEL) << " found " << nscint << " scintillator materials  " ;
    1255 
    1256         GScintillatorLib* sclib = getScintillatorLib() ;
    1257 
    1258         for(unsigned int i=0 ; i < nscint ; i++)
    1259         {
    1260             GPropertyMap<double>* scint = dynamic_cast<GPropertyMap<double>*>(getScintillatorMaterial(i));
    1261             sclib->add(scint);
    1262         }
    1263 
    1264         sclib->close();
    1265     }
    1266 }
    1267 
    1268 void GGeo::findScintillatorMaterials(const char* props)
    1269 {
    1270     m_scintillators_raw = getRawMaterialsWithProperties(props, ',');
    1271     //assert(m_scintillators_raw.size() > 0 );
    1272 }

    1122 std::vector<GMaterial*> GGeo::getRawMaterialsWithProperties(const char* props, char delim) const
    1123 {
    1124     return m_materiallib->getRawMaterialsWithProperties(props, delim );
    1125 }
    1126 

    1075 std::vector<GMaterial*> GMaterialLib::getRawMaterialsWithProperties(const char* props, char delim) const
    1076 {
    1077     std::vector<std::string> elem ;
    1078     BStr::split(elem, props, delim);
    1079 
    1080     LOG(LEVEL)
    1081          << props
    1082          << " m_materials_raw.size()  " << m_materials_raw.size()
    1083          ;
    1084 
    1085     std::vector<GMaterial*>  selected ;
    1086     for(unsigned int i=0 ; i < m_materials_raw.size() ; i++)
    1087     {
    1088         GMaterial* mat = m_materials_raw[i];
    1089         unsigned int found(0);
    1090         for(unsigned int p=0 ; p < elem.size() ; p++)
    1091         {
    1092            if(mat->hasProperty(elem[p].c_str())) found+=1 ;
    1093         }
    1094         if(found == elem.size()) selected.push_back(mat);
    1095     }
    1096     return selected ;
    1097 }

    0328 void GMaterialLib::addRaw(GMaterial* mat)
     329 {
     330     m_materials_raw.push_back(mat);
     331 }


    104 void X4MaterialTable::init()
    105 {
    106     unsigned num_input_materials = m_input_materials.size() ;
    107 
    108     LOG(LEVEL) << ". G4 nmat " << num_input_materials ;
    109 
    110     for(unsigned i=0 ; i < num_input_materials ; i++)
    111     {
    112         G4Material* material = m_input_materials[i] ;
    113         G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
    114 
    115         if( mpt == NULL )
    116         {
    117             LOG(error) << "PROCEEDING TO convert material with no mpt " << material->GetName() ;
    118             // continue ;  
    119         }
    120         else
    121         {
    122             LOG(LEVEL) << " converting material with mpt " <<  material->GetName() ;
    123         }
    124 
    125 
    126         GMaterial* mat = X4Material::Convert( material );
    127         if(mat->hasProperty("EFFICIENCY"))
    128         {
    129              m_materials_with_efficiency.push_back(material);
    130         }
    131 
    132         //assert( mat->getIndex() == i ); // this is not the lib, no danger of triggering a close
    133 
    134         m_mlib->add(mat) ;    // creates standardized material
    135         m_mlib->addRaw(mat) ; // stores as-is
    136     }
    137 }


* X4Material::Convert almost certainly does the domain standardization with addPropertyStandardized




    1287 unsigned int GGeo::getNumScintillatorMaterials()
    1288 {
    1289     return m_scintillators_raw.size();
    1290 }
    1291 
    1292 GMaterial* GGeo::getScintillatorMaterial(unsigned int index)
    1293 {
    1294     return index < m_scintillators_raw.size() ? m_scintillators_raw[index] : NULL ;
    1295 }
    1296 





