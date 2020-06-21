genstep-workflow-review
==========================

g4ok/G4Opticks
    passes 6*4 parameters over to m_genstep_collector (CGenstepCollector)
    
cfg4/CGenstepCollector

    134 void CGenstepCollector::collectScintillationStep
    135 (
    136             G4int                /*id*/,
    137             G4int                parentId,
    138             G4int                materialId,
    139             G4int                numPhotons,
    140 
    ...
    165 )
    166 {
    167      m_scintillation_count += 1 ;   // 1-based index
    168      m_gs_photons.push_back(numPhotons);
    169 
    173      uif_t uifa[4] ;
    174      uifa[0].i = SCINTILLATION ;
    175 
    176     // id == 0 ? m_scintillation_count : id  ;   // use the 1-based index when id zero 
    177      uifa[1].i = parentId ;
    178      uifa[2].i = translate(materialId) ;   // raw G4 materialId translated into GBndLib material line for GPU usage 
    179      uifa[3].i = numPhotons ;
    180 



    226 void CGenstepCollector::collectCerenkovStep
    227 (
    228             G4int              /*id*/,
    229             G4int                parentId,
    230             G4int                materialId,
    231             G4int                numPhotons,
    ...
    257 )
    258 {
    259      m_cerenkov_count += 1 ;   // 1-based index
    260      m_gs_photons.push_back(numPhotons);
    261 
    265      uif_t uifa[4] ;
    266      uifa[0].i = CERENKOV ;
    267    // id == 0 ? -m_cerenkov_count : id  ;   // use the negated 1-based index when id zero 
    268      uifa[1].i = parentId ;
    269      uifa[2].i = translate(materialId) ;
    270      uifa[3].i = numPhotons ;
    271 




* note the *id* parameter is not used in either of the above, 
  that slot is currently set to SCINTILLATION or CERENKOV   (enum from optickscore/OpticksPhoton.h)

* obvious extension 

1) new enum OpticksGenstep.h 
2) id -> gentype for identification


::

     13 
     14 enum
     15 {   
     16     OpticksGenstep_Invalid                  = 0,
     17     OpticksGenstep_G4Cerenkov_1042          = 1,
     18     OpticksGenstep_G4Scintillation_1042     = 2, 
     19     OpticksGenstep_DsG4Cerenkov_r3971       = 3,
     20     OpticksGenstep_DsG4Scintillation_r3971  = 4,
     21     OpticksGenstep_NumType                  = 5
     22 };
     23   



3) make it available as an ini for python

::

    epsilon:optickscore blyth$ cat /usr/local/opticks/build/optickscore/OpticksGenstep_Enum.ini 
    OpticksGenstep_Invalid=0
    OpticksGenstep_G4Cerenkov_1042=1
    OpticksGenstep_G4Scintillation_1042=2
    OpticksGenstep_DsG4Cerenkov_r3971=3
    OpticksGenstep_DsG4Scintillation_r3971=4
    OpticksGenstep_NumType=5
    epsilon:optickscore blyth$ 
    epsilon:optickscore blyth$ 


