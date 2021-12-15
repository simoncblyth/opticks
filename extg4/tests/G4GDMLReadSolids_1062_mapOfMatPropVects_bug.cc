// om-;TEST=G4GDMLReadSolids_1062_mapOfMatPropVects_bug om-t

const char* GDML = R"LITERAL(<?xml version="1.0" encoding="UTF-8" ?>
<gdml xmlns:gdml="http://cern.ch/2001/Schemas/GDML" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd" >

 <define>
  <position name="pos0" unit="mm" x="0" y="0" z="0" />
  <position name="pos1" unit="mm" x="2000" y="0" z="0" /> 
  <matrix name="EFF0" coldim="2" values="0.0 0.0 1000.0 0.0"/>
  <matrix name="EFF1" coldim="2" values="0.0 1.0 1000.0 1.0"/>
 </define>

 <materials>
  <element Z="7" formula="N" name="Nitrogen" >
   <atom value="14.01" />
  </element>
  <element Z="8" formula="O" name="Oxygen" >
   <atom value="16" />
  </element>

  <material formula=" " name="Air" >
   <D value="0.00129" />
   <fraction n="0.7" ref="Nitrogen" />
   <fraction n="0.3" ref="Oxygen" />
  </material>
 </materials>

 <solids>
  <box aunit="radian" lunit="mm" name="world" x="10000" y="10000" z="10000" />
  <box aunit="radian" lunit="mm" name="s0" x="200" y="200" z="200" />
  <box aunit="radian" lunit="mm" name="s1" x="200" y="200" z="200" />

  <opticalsurface name="surf0" model="glisur" finish="polished" type="dielectric_dielectric" value="1.0">
      <property name="EFFICIENCY" ref="EFF0" />
  </opticalsurface>
  <opticalsurface name="surf1" model="glisur" finish="polished" type="dielectric_dielectric" value="1.0">
      <property name="EFFICIENCY" ref="EFF1" />
  </opticalsurface>

 </solids>

 <structure>
  <volume name="lv0" >
   <materialref ref="Air" />
   <solidref ref="s0" />
  </volume>

  <volume name="lv1" >
   <materialref ref="Air" />
   <solidref ref="s1" />
  </volume>

  <volume name="World" >
   <materialref ref="Air" />
   <solidref ref="world" />

   <physvol name="pv0">
    <volumeref ref="lv0" />
    <positionref ref="pos0" />
   </physvol>

   <physvol name="pv1">
    <volumeref ref="lv1" />
    <positionref ref="pos1" />
   </physvol>
  </volume>

  <skinsurface name="skin0" surfaceproperty="surf0" >
    <volumeref ref="lv0"/>
  </skinsurface> 

  <skinsurface name="skin1" surfaceproperty="surf1" >
    <volumeref ref="lv1"/>
  </skinsurface> 

 </structure>

 <setup name="Default" version="1.0" >
  <world ref="World" />
 </setup>
</gdml>

)LITERAL";

#include <sstream>
#include <iostream>
#include <cstring>
#include <fstream>
#include <cstdlib>

#include <set>

#include "G4Version.hh"
#include "G4GDMLParser.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4MaterialPropertiesTable.hh"


void getRange(G4double& mn, G4double& mx, const G4MaterialPropertyVector* pvec)
{
    size_t plen = pvec ? pvec->GetVectorLength() : 0 ;
    mn = std::numeric_limits<double>::max();   
    mx = std::numeric_limits<double>::lowest();
            
    for (size_t j=0; j<plen; j++)
    {   
        double value = (*pvec)[j] ;
        if(value > mx) mx = value ;
        if(value < mn) mn = value ;
    }
}

/**
G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc
-----------------------------------------------

::

    G4GDMLReadSolids_1062_mapOfMatPropVects_bug  
       ## gdml schema validation no longer done by default due to network hangs

    VALIDATE=1 G4GDMLReadSolids_1062_mapOfMatPropVects_bug
        ## enable schema validation with envvar

**/


int main(int argc, char** argv)
{
    const char* tmp = getenv("TMP"); 
    if(!tmp) tmp = "/tmp" ; 
    std::stringstream ss ; 
    ss << tmp << "/" << "mapOfMatPropVects_BUG.gdml" ; 
    std::string spath = ss.str(); 

    const char* path = argc > 1 ? argv[1] : spath.c_str() ;   

    std::cout << "writing gdml to " << path << std::endl ; 
    std::ofstream fp(path, std::ios::out); 
    fp.write(GDML, strlen(GDML)) ; 
    fp.close();   

    std::cout << "parsing gdml from " << path << std::endl ; 
    std::cout << "G4VERSION_NUMBER " << G4VERSION_NUMBER << std::endl; 
    bool validate = getenv("VALIDATE") != NULL ; 
    std::cout << "VALIDATE " << validate << std::endl ; 
    G4GDMLParser parser ; 
    parser.Read(path, validate);  

    const G4LogicalSkinSurfaceTable* tab = G4LogicalSkinSurface::GetSurfaceTable();
    assert( tab->size() == 2 ); 

    typedef G4MaterialPropertyVector MPV ; 

    std::set<double> effval ; 

    for(size_t i=0 ; i < tab->size() ; i++)
    {
        G4LogicalSkinSurface* skin = (*tab)[i] ;
        const G4String& skinname = skin->GetName();

        G4OpticalSurface* opsurf = dynamic_cast<G4OpticalSurface*>(skin->GetSurfaceProperty());
        assert(opsurf);

        const G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable();
        std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
     
        for(unsigned i=0 ; i < pns.size() ; i++)
        {
            const std::string& pname = pns[i] ;
            G4int pidx = mpt->GetPropertyIndex(pname);
            MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx);
            if(pvec == NULL) continue ; 
            G4double mn, mx ;  
            getRange(mn, mx, pvec); 
            std::cout 
                << " skinname " << skinname 
                << " pname " << pname << " : " 
                << " mn " << std::setw(10) << std::fixed << std::setprecision(4) << mn 
                << " mx " << std::setw(10) << std::fixed << std::setprecision(4) << mx 
                << std::endl 
                ; 
            assert( mn == mx );  
            effval.insert(mn); 
        }
    }

    assert( effval.size() == 2 ); // expecting one skin to have value 0. and the other 1. 
    return 0 ; 
}

/**
With 1042 get expected output::

    epsilon:extg4 blyth$ G4GDMLReadSolids_1062_mapOfMatPropVects_bug
    writing gdml to /tmp/mapOfMatPropVects_BUG.gdml
    parsing gdml from /tmp/mapOfMatPropVects_BUG.gdml
    G4VERSION_NUMBER 1042
    G4GDML: Reading '/tmp/mapOfMatPropVects_BUG.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/mapOfMatPropVects_BUG.gdml' done!
    Stripping off GDML names of materials, solids and volumes ...
     skinname skin0 pname EFFICIENCY :  mn     0.0000 mx     0.0000
     skinname skin1 pname EFFICIENCY :  mn     1.0000 mx     1.0000
    epsilon:extg4 blyth$ 

With 1062 get both EFFICIENCY all zero and the assert is tripped::

    epsilon:extg4 charles$ G4GDMLReadSolids_1062_mapOfMatPropVects_bug
    writing gdml to /tmp/mapOfMatPropVects_BUG.gdml
    parsing gdml from /tmp/mapOfMatPropVects_BUG.gdml
    G4VERSION_NUMBER 1062
    G4GDML: Reading '/tmp/mapOfMatPropVects_BUG.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/mapOfMatPropVects_BUG.gdml' done!
    Stripping off GDML names of materials, solids and volumes ...
     skinname skin0 pname EFFICIENCY :  mn     0.0000 mx     0.0000
     skinname skin1 pname EFFICIENCY :  mn     0.0000 mx     0.0000
    Assertion failed: (effval.size() == 2), function main, file /Users/charles/opticks/extg4/tests/G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc, line 173.
    Abort trap: 6
    epsilon:extg4 charles$ vi G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc
    epsilon:extg4 charles$ 

**/

/**
CAUSE OF BUG bordersurface/skinsurface property bug in 1062 (also in 1070)
============================================================================

source/persistency/gdml/include/G4GDMLReadSolids.hh::

    112 
    113 private:
    114   std::map<G4String, G4MaterialPropertyVector*> mapOfMatPropVects;
    115 
    116 };

source/persistency/gdml/src/G4GDMLReadSolids.cc::

    2477 void G4GDMLReadSolids::
    2478 PropertyRead(const xercesc::DOMElement* const propertyElement,
    2479              G4OpticalSurface* opticalsurface)
    2480 {
    ...
    2523    if (matrix.GetRows() == 0) { return; }
    2524   
    2525    G4MaterialPropertiesTable* matprop=opticalsurface->GetMaterialPropertiesTable();
    2526    if (!matprop)
    2527    {
    2528      matprop = new G4MaterialPropertiesTable();
    2529      opticalsurface->SetMaterialPropertiesTable(matprop);
    2530    }
    2531    if (matrix.GetCols() == 1)  // constant property assumed
    2532    {
    2533      matprop->AddConstProperty(Strip(name), matrix.Get(0,0));
    2534    }
    2535    else  // build the material properties vector
    2536    {
    2537      G4MaterialPropertyVector* propvect;
    2538      // first check if it was already built
    2539      if ( mapOfMatPropVects.find(Strip(name)) == mapOfMatPropVects.end())
    2540        {
    2541      // if not create a new one
    2542      propvect = new G4MaterialPropertyVector();
    2543      for (size_t i=0; i<matrix.GetRows(); i++)
    2544        {
    2545          propvect->InsertValues(matrix.Get(i,0),matrix.Get(i,1));
    2546        }
    2547      // and add it to the list for potential future reuse
    2548      mapOfMatPropVects[Strip(name)] = propvect;
    2549        }
    2550      else
    2551        {
    2552      propvect = mapOfMatPropVects[Strip(name)];
    2553        }
    2554      
    2555      matprop->AddProperty(Strip(name),propvect);
    2556    }
    2557 }


In source/persistency/gdml/src/G4GDMLReadSolids.cc
the key of the mapOfMatPropVects is "Strip(name)" which means
that only the first occurrence of each stripped property name (eg EFFICIENCY) 
amongst all skinsurface and bordersurface elements in the entire geometry
gets set to the actual G4MaterialPropertyVector values.  
Subsequent uses of the same property name in other skinsurface 
or bordersurface incorrectly adopt the G4MaterialPropertyVector values from
the one that happens to be first.

Obserbe the issue to be present in 1070 and 1062, but it is not in 1042.


Quick fix is to comment the below line, effectively eliminating mapOfMatPropVects::

    2548      //mapOfMatPropVects[Strip(name)] = propvect;

Suggested fix is to remove mapOfMatPropVects returning to what 1042 does::

    2479 void G4GDMLReadSolids::
    2480 PropertyRead(const xercesc::DOMElement* const propertyElement,
    2481              G4OpticalSurface* opticalsurface)
    2482 {
    ....
    2525    if (matrix.GetRows() == 0) { return; }
    2526 
    2527    G4MaterialPropertiesTable* matprop=opticalsurface->GetMaterialPropertiesTable();
    2528    if (!matprop)
    2529    {
    2530      matprop = new G4MaterialPropertiesTable();
    2531      opticalsurface->SetMaterialPropertiesTable(matprop);
    2532    }
    2533    if (matrix.GetCols() == 1)  // constant property assumed
    2534    {
    2535      matprop->AddConstProperty(Strip(name), matrix.Get(0,0));
    2536    }
    2537    else  // build the material properties vector
    2538    {
    2539      G4MaterialPropertyVector* propvect = new G4MaterialPropertyVector();
    2540      for (size_t i=0; i<matrix.GetRows(); i++)
    2541      {
    2542        propvect->InsertValues(matrix.Get(i,0),matrix.Get(i,1));
    2543      }
    2544      matprop->AddProperty(Strip(name),propvect);
    2545    }
    2546 }

The mapOfMatPropVects adds complexity (and this bug) with extremely minimal benefit 
even if it were to use a key with an appropriate identity. For example the name 
of the matrix element or the address of the G4GDMLMatrix.


**/

