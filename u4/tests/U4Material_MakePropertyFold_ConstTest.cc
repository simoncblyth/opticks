/**
U4Material_MakePropertyFold_ConstTest.cc
=========================================

Below script creates material folder and opens that into ipython for inspection::

    ~/o/u4/tests/U4Material_MakePropertyFold_ConstTest.sh

**/

#include "spath.h"
#include "U4Material.hh"
#include "U4MaterialPropertiesTable.h"
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

    for(int i=0 ; i < NUM_CONST ; i++) U4MaterialPropertiesTable::AddConstProperty(mpt, CONSTS[i].name, CONSTS[i].value);
    // need to use compat method to avoid API difference between 10.x and 11.x
    mat->SetMaterialPropertiesTable(mpt);
    return mat;
}

} // namespace

int main(int argc, char** argv)
{
    G4Material* mat = MakeMaterial();
    std::cout << " mat " << mat->GetName() << "\n" ;

    NPFold* fold = U4Material::MakePropertyFold(mat);

    const char* base = "$TMP/U4Material_MakePropertyFold_ConstTest" ;
    const char* fdir = spath::Resolve(base) ;
    fold->save(fdir);

    return 0;
}

/**
Note that adding the dummy RINDEX results in the creation of GROUPVEL.

In [1]: f.RINDEX
Out[1]:
array([[1. , 1.3],
       [2. , 1.5],
       [3. , 1.7],
       [4. , 1.6],
       [5. , 1.4],
       [6. , 1.2]])

In [2]: f.GROUPVEL
Out[2]:
array([[  1.   , 188.722],
       [  1.5  , 177.545],
       [  2.5  , 143.218],
       [  3.5  , 181.692],
       [  4.5  , 199.862],
       [  6.   , 249.827]])


**/


