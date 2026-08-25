Geant4_11_G4MaterialPropertiesTable__AddProperty_G4Exception_mat207_for_non_standard_names
============================================================================================


FIX avoiding Geant 10 and 11 API difference
----------------------------------------------

Fixed by adding and using compat statics in the test::

     15 struct U4MaterialPropertiesTable
     16 {
     17     static void AddProperty(G4MaterialPropertiesTable* mpt, const char* name, G4MaterialPropertyVector* vec);
     18     static void AddConstProperty(G4MaterialPropertiesTable* mpt, const char* name, double val);


Issue
------

::

    [lo] A[blyth@localhost u4]$ echo $OPTICKS_PREFIX
    /data1/blyth/local/opticks_Debug_g411


    [lo] A[blyth@localhost u4]$ U4Material_MakePropertyFold_ConstTest

    -------- EEEE ------- G4Exception-START -------- EEEE -------

    *** ExceptionHandler is not defined ***
    *** G4Exception : mat207
          issued by : G4MaterialPropertiesTable::AddProperty()
    Attempting to create a new material constant property key LAB2PPO_PROB without setting
    createNewKey parameter of AddProperty to true.
    *** Fatal Exception ***
    -------- EEEE ------- G4Exception-END -------- EEEE -------


    *** G4Exception: Aborting execution ***
    Aborted (core dumped)


::

     41 G4Material* MakeMaterial()
     42 {
     43     G4Material* mat = new G4Material("ConstTestMaterial", 1.0, 1.0*g/mole, 1.0*g/cm3);
     44     G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
     45     mpt->AddProperty("RINDEX", MakeVectorProperty());
     46
     47     for(int i=0 ; i < NUM_CONST ; i++) mpt->AddConstProperty(CONSTS[i].name, CONSTS[i].value);
     48     mat->SetMaterialPropertiesTable(mpt);
     49     return mat;
     50 }



