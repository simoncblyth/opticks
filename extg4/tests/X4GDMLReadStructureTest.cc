#include <fstream>

#include "OPTICKS_LOG.hh"
#include "BFile.hh"
#include "X4GDMLReadStructure.hh"

#include <xercesc/util/PlatformUtils.hpp>


void test_ReadSolidFromString()
{
    const char*  GDML = R"LITERAL(

<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="SchemaLocation">
  <solids>
     <box lunit="mm" name="WorldBox0xc15cf400x3ebf070" x="4800000" y="4800000" z="4800000"/>
  </solids>
</gdml>

)LITERAL";

    const G4VSolid* solid = X4GDMLReadStructure::ReadSolidFromString(GDML) ;  
    LOG(info) << " solid " << solid ; 

    G4cout << *solid ; 
}




void test_readString(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : NULL ; 
    const char*  GDML = R"LITERAL(

<gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="SchemaLocation">

   <userinfo>
     <auxiliary auxtype="opticks_note" auxvalue="string literal GDML from x4  tests/X4GDMLReadStructureTest.cc "/>
   </userinfo>

  <define>

  <matrix coldim="2" name="EFFICIENCY0x1d79780" values="1.512e-06 0.0001 1.5498e-06 0.0001 1.58954e-06 0.000440306 1.63137e-06 0.000782349 1.67546e-06 0.00112439 1.722e-06 0.00146644 1.7712e-06 0.00180848 1.8233e-06 0.00272834 1.87855e-06 0.00438339 1.93725e-06 0.00692303 1.99974e-06 0.00998793 2.0664e-06 0.0190265 2.13766e-06 0.027468 2.214e-06 0.0460445 2.296e-06 0.0652553 2.38431e-06 0.0849149 2.47968e-06 0.104962 2.583e-06 0.139298 2.69531e-06 0.170217 2.81782e-06 0.19469 2.952e-06 0.214631 3.0996e-06 0.225015 3.26274e-06 0.24 3.44401e-06 0.235045 3.64659e-06 0.21478 3.87451e-06 0.154862 4.13281e-06 0.031507 4.42801e-06 0.00478915 4.76862e-06 0.00242326 5.16601e-06 0.000850572 5.63564e-06 0.000475524 6.19921e-06 0.000100476 6.88801e-06 7.50165e-05 7.74901e-06 5.00012e-05 8.85601e-06 2.49859e-05 1.0332e-05 0 1.23984e-05 0 1.5498e-05 0 2.0664e-05 0"/>

  </define>

  <materials>

   <isotope N="1" Z="1" name="H10x1e114d0">
      <atom unit="g/mole" value="1.00782503081372"/>
    </isotope>
    <isotope N="2" Z="1" name="H20x1e11350">
      <atom unit="g/mole" value="2.01410199966617"/>
    </isotope>
    <element name="/dd/Materials/Hydrogen0x1e113f0">
      <fraction n="0.999885" ref="H10x1e114d0"/>
      <fraction n="0.000115" ref="H20x1e11350"/>
    </element>

   <material name="/dd/Materials/fakePyrex" state="solid">
      <P unit="pascal" value="101324.946686941"/>
      <MEE unit="eV" value="50.1287588602539"/>
      <D unit="g/cm3" value="0.919999515933733"/>
      <fraction n="1.0" ref="/dd/Materials/Hydrogen0x1e113f0"/>
    </material>

    <material name="/dd/Materials/fakeVacuum" state="gas">
      <T unit="K" value="0.1"/>
      <P unit="pascal" value="9.99999473841014e-20"/>
      <MEE unit="eV" value="19.2"/>
      <D unit="g/cm3" value="1.09999942122512e-25"/>
      <fraction n="1.0" ref="/dd/Materials/Hydrogen0x1e113f0"/>
    </material>

  </materials>

  <solids>
     <box lunit="mm" name="WorldBox0xc15cf400x3ebf070" x="4800000" y="4800000" z="4800000"/>

     <sphere name="pmt-hemi0xc0fed900x3e85f00"     aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" rmax="99" rmin="0" startphi="0" starttheta="0" />
     <sphere name="pmt-hemi-vac0xc21e2480x3e85290" aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" rmax="98" rmin="0" startphi="0" starttheta="0" />
     <sphere name="pmt-hemi-cathode0xc2f1ce80x3e842d0" aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" rmax="98" rmin="0" startphi="0" starttheta="0" />


    <opticalsurface finish="0" model="0" name="SCB_photocathode_opsurf" type="0" value="1">
         <property name="EFFICIENCY" ref="EFFICIENCY0x1d79780"/>   <!-- the non-zero efficiency-->
    </opticalsurface>

  </solids>

   <structure>

    <volume name="/dd/Geometry/PMT/lvPmtHemiVacuum0xc2c7cc80x3ee9760">
      <materialref ref="/dd/Materials/fakeVacuum"/>
      <solidref ref="pmt-hemi-vac0xc21e2480x3e85290"/>
   </volume>
  
   <volume name="/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca00x3ee9400">
      <materialref ref="/dd/Materials/fakeVacuum"/>
      <solidref ref="pmt-hemi-cathode0xc2f1ce80x3e842d0"/>
    </volume>


   <volume name="/dd/Geometry/PMT/lvPmtHemi0xc1337400x3ee9b20">
      <materialref ref="/dd/Materials/fakePyrex"/>
      <solidref ref="pmt-hemi0xc0fed900x3e85f00"/>
      <physvol name="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0">
        <volumeref ref="/dd/Geometry/PMT/lvPmtHemiVacuum0xc2c7cc80x3ee9760"/>
      </physvol>
    </volume>

 <volume name="/dd/Geometry/PMT/lvPmtHemiVacuum0xc2c7cc80x3ee9760">
      <materialref ref="/dd/Materials/fakeVacuum"/>
      <solidref ref="pmt-hemi-vac0xc21e2480x3e85290"/>
      <physvol name="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720">
        <volumeref ref="/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca00x3ee9400"/>
      </physvol>

    </volume>



    <bordersurface name="SCB_photocathode_logsurf1" surfaceproperty="SCB_photocathode_opsurf">
       <physvolref ref="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0" />
       <physvolref ref="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720" />
    </bordersurface>

    <bordersurface name="SCB_photocathode_logsurf2" surfaceproperty="SCB_photocathode_opsurf">
       <physvolref ref="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720" />
       <physvolref ref="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0" />
    </bordersurface>



   <volume name="World0xc15cfc00x40f7000">
      <materialref ref="/dd/Materials/fakeVacuum"/>
      <solidref ref="WorldBox0xc15cf400x3ebf070"/>
    </volume>



   </structure> 

  <setup name="Default" version="1.0">
    <world ref="World0xc15cfc00x40f7000"/>
  </setup>

</gdml>

)LITERAL";


     if(path)
     {
         LOG(info) << "writing GDMLString to path " << path ; 
         X4GDMLReadStructure::WriteGDMLString(GDML, path); 
         X4GDMLReadStructure reader ; 
         reader.readFile(path) ; 
     }
     else
     {
         X4GDMLReadStructure reader ; 
         reader.readString(GDML) ; 
     }
  

}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    xercesc::XMLPlatformUtils::Initialize();

    //test_ReadSolidFromString(); 
    test_readString(argc, argv); 

    return 0 ; 
}


