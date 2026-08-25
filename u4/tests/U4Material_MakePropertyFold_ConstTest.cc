/**
U4Material_MakePropertyFold_ConstTest.cc
=========================================

cd $TMP/U4Material_MakePropertyFold_ConstTest
and use f to check files.
**/

#include "OPTICKS_LOG.hh"
#include "spath.h"
#include "U4Material.hh"
#include "NPFold.h"

#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialPropertyVector.hh"
#include "G4SystemOfUnits.hh"

namespace {

struct ConstKV {
    const char* name ;
    double value ;
};

const ConstKV CONSTS[] = {
    {"LAB2PPO_PROB", 0.68392399},
    {"LAB2BIS_PROB", 0.31607601},
    {"SCINTILLATIONYIELD", 10168.0},
};

const int NUM_CONST = sizeof(CONSTS)/sizeof(ConstKV);

G4MaterialPropertyVector* MakeVectorProperty()
{
    double energy[6] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    double value[6]  = { 1.3, 1.5, 1.7, 1.6, 1.4, 1.2 };
    return new G4MaterialPropertyVector(energy, value, 6);
}

G4Material* MakeMaterial()
{
    G4Material* mat = new G4Material("ConstTestMaterial", 1.0, 1.0*g/mole, 1.0*g/cm3);
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", MakeVectorProperty());

    for(int i=0 ; i < NUM_CONST ; i++) mpt->AddConstProperty(CONSTS[i].name, CONSTS[i].value);
    mat->SetMaterialPropertiesTable(mpt);
    return mat;
}

} // namespace

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    G4Material* mat = MakeMaterial();
    LOG(info) << " mat " << mat->GetName();

    NPFold* fold = U4Material::MakePropertyFold(mat);

    const char* base = "$TMP/U4Material_MakePropertyFold_ConstTest" ;
    const char* fdir = spath::Resolve(base) ;
    fold->save(fdir);

    return 0;
}
