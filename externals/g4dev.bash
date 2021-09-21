##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##

g4dev-src(){      echo externals/g4dev.bash ; }
g4dev-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(g4dev-src)} ; }
g4dev-vi(){       vi $(g4dev-source) ; }
g4dev-usage(){ cat << \EOU

Geant4 Development
=====================

The g4dev- functions are for working with versions of Geant4 other
than the current Opticks standard one. 

For the standard version use the g4- functions.


* https://github.com/Geant4/geant4/releases?after=v9.5.1

EOU
}
g4dev-env(){      
   olocal-  
   xercesc-  
   opticks-
}


g4dev-edir(){ echo $(opticks-home)/g4 ; }

g4dev-prefix(){ 
    case $NODE_TAG in 
       MGB) echo $HOME/local/opticks/externals ;;
         D) echo /usr/local/opticks/externals ;;
         X) echo /opt/geant4 ;;
         *) echo ${LOCAL_BASE:-/usr/local}/opticks/externals ;;
    esac
 }

g4dev-libsuffix(){ 
    case $NODE_TAG in 
         X) echo 64  ;;
         *) echo -n ;;
    esac
}



g4dev-tag(){   echo g4 ; }


g4dev-nom-notes(){ cat << EON

The nom identifier needs to match the name of the folder created by exploding the zip or tarball, 
unfortunamely this is not simply connected with the basename of the url and also Geant4 continues to 
reposition URLs so these are liable to going stale.

EON
}

#g4dev-nom(){ echo Geant4-10.2.1 ; }
#g4dev-nom(){ echo geant4-9.5.0 ; }
#g4dev-nom(){ echo geant4_10_04_p01 ; }
g4dev-nom(){  echo geant4_10_04_p02 ; }
#g4dev-nom(){ echo geant4.10.05.b01 ; }


g4dev-url(){   
   case $(g4dev-nom) in
       Geant4-10.2.1) echo http://geant4.cern.ch/support/source/geant4_10_02_p01.zip ;;
        geant4-9.5.0) echo https://github.com/Geant4/geant4/archive/v9.5.0.zip ;;
        geant4_10_04_p01) echo http://geant4-data.web.cern.ch/geant4-data/releases/geant4_10_04_p01.zip  ;; 
        geant4_10_04_p02) echo http://geant4-data.web.cern.ch/geant4-data/releases/geant4.10.04.p02.tar.gz ;; 
        geant4.10.05.b01) echo http://geant4-data.web.cern.ch/geant4-data/releases/geant4.10.05.b01.tar.gz ;; 
   esac
}

g4dev-idir(){ echo $(g4dev-prefix) ; }
g4dev-dir(){   echo $(g4dev-prefix)/$(g4dev-tag)/$(g4dev-nom) ; } 

g4dev-dist(){ echo $(dirname $(g4dev-dir))/$(basename $(g4dev-url)) ; }
g4dev-filename(){  echo $(basename $(g4dev-url)) ; }
g4dev-name(){  
   local name=$(g4dev-filename) ; 
   name=${name/.tar.gz}
   name=${name/.zip}
   echo $name
}

  

g4dev-txt(){ vi $(g4dev-dir)/CMakeLists.txt ; }


g4dev-info(){  cat << EOI

    g4dev-nom  : $(g4dev-nom)
    g4dev-url  : $(g4dev-url)
    g4dev-dist : $(g4dev-dist)
    g4dev-filename : $(g4dev-filename)
    g4dev-name     : $(g4dev-name)


    g4dev-idir : $(g4dev-idir)
    g4dev-dir  : $(g4dev-dir)


EOI
}

g4dev-find(){ find $(g4dev-dir) -name ${1:-G4OpBoundaryProcess.cc} ; }


g4dev-bdir(){ echo $(g4dev-dir).build ; }

g4dev-cmake-dir(){     echo $(g4dev-prefix)/lib$(g4dev-libsuffix)/$(g4dev-nom) ; }
g4dev-examples-dir(){  echo $(g4dev-prefix)/share/$(g4dev-nom)/examples ; }


g4dev-ecd(){  cd $(g4dev-edir); }
g4dev-cd(){   cd $(g4dev-dir); }
g4dev-icd(){  cd $(g4dev-prefix); }
g4dev-bcd(){  cd $(g4dev-bdir); }
g4dev-ccd(){  cd $(g4dev-cmake-dir); }
g4dev-xcd(){  cd $(g4dev-examples-dir); }


g4dev-get-tgz(){
   local dir=$(dirname $(g4dev-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4dev-url)
   # replace zip to tar.gz
   url=${url/.zip/.tar.gz}
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}

   [ ! -f "$tgz" ] && curl -L -O $url 
   [ ! -d "$nam" ] && tar zxvf $tgz 
}

g4dev-get(){
   local dir=$(dirname $(g4dev-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(g4dev-url)
   local dst=$(basename $url)
   local nom=$(g4dev-nom)

   [ ! -f "$dst" ] && echo getting $url && curl -L -O $url 

   if [ "${dst/.zip}" != "${dst}" ]; then 
        [ ! -d "$nom" ] && unzip $dst 
   fi 
   if [ "${dst/.tar.gz}" != "${dst}" ]; then 
        [ ! -d "$nom" ] && tar zxvf $dst 
   fi 

}

g4dev-wipe(){
   local bdir=$(g4dev-bdir)
   rm -rf $bdir
}



################# below funcions for styduing G4 source ##################################

g4dev-ifind(){ find $(g4dev-idir) -name ${1:-G4VUserActionInitialization.hh} ; }
g4dev-sfind(){ find $(g4dev-dir)/source -name ${1:-G4VUserActionInitialization.hh} ; }

g4dev-hh(){ find $(g4dev-dir)/source -name '*.hh' -exec grep -H ${1:-G4GammaConversion} {} \; ; }
g4dev-icc(){ find $(g4dev-dir)/source -name '*.icc' -exec grep -H ${1:-G4GammaConversion} {} \; ; }
g4dev-cc(){ find $(g4dev-dir)/source -name '*.cc' -exec grep -H ${1:-G4GammaConversion} {} \; ; }

g4dev-cls-copy(){
   local iwd=$PWD
   local name=${1:-G4Scintillation}
   local lname=${name/G4}

   local sauce=$(g4dev-dir)/source
   local hh=$(find $sauce -name "$name.hh")
   local cc=$(find $sauce -name "$name.cc")
   local icc=$(find $sauce -name "$name.icc")

   [ "$hh" != "" ]  && echo cp $hh $iwd/$lname.hh
   [ "$cc" != "" ] && echo cp $cc $iwd/$lname.cc
   [ "$icc" != "" ] && echo cp $icc $iwd/$lname.icc
}

g4dev-cls(){
   local iwd=$PWD
   g4dev-cd
   local name=${1:-G4Scintillation}

   local h=$(find source -name "$name.h")
   local hh=$(find source -name "$name.hh")
   local cc=$(find source -name "$name.cc")
   local icc=$(find source -name "$name.icc")

   local vcmd="vi -R $h $hh $icc $cc"
   echo $vcmd
   eval $vcmd

   cd $iwd
}

g4dev-look(){ 
   local iwd=$PWD
   g4dev-cd
   local spec=${1:-G4RunManagerKernel.cc:707}

   local name=${spec%:*}
   local line=${spec##*:}
   [ "$line" == "$spec" ] && line=1

   local fcmd="find source -name $name"
   local path=$($fcmd)

   echo $spec $name $line $path 

   if [ "$path" == "" ]; then 
      echo $msg FAILED to find $name with : $fcmd
      return 
   fi 
   local vcmd="vi -R $path +$line"
   echo $vcmd
   eval $vcmd
     
   cd $iwd
}



g4dev-aux-notes(){ cat << EON


GDML auxiliary
---------------

* https://github.com/hanswenzel/G4OpticksTest/blob/master/gdml/G4Opticks.gdml

::


    166         <volume name="Obj">
    167             <materialref ref="LS0x4b61c70"/>
    168             <solidref ref="Obj"/>
    169             <colorref ref="blue"/>
    170             <auxiliary auxtype="StepLimit" auxvalue="0.4" auxunit="mm"/>
    171             <auxiliary auxtype="SensDet" auxvalue="lArTPC"/>
    172             <physvol name="Det">
    173                 <volumeref ref="Det"/>
    174                 <position name="Det" unit="mm" x="0" y="0" z="100"/>
    175             </physvol>
    176         </volume>




::

    epsilon:geant4.10.04.p02 blyth$ g4-hh G4GDMLAuxMapType
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:typedef std::map<G4LogicalVolume*,G4GDMLAuxListType> G4GDMLAuxMapType;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:   const G4GDMLAuxMapType* GetAuxMap() const {return &auxMap;}
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:   G4GDMLAuxMapType auxMap;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLParser.hh:   inline const G4GDMLAuxMapType* GetAuxMap() const;
    epsilon:geant4.10.04.p02 blyth$ 


    epsilon:geant4.10.04.p02 blyth$ g4-hh G4GDMLAuxMapType
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:typedef std::map<G4LogicalVolume*,G4GDMLAuxListType> G4GDMLAuxMapType;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:   const G4GDMLAuxMapType* GetAuxMap() const {return &auxMap;}
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:   G4GDMLAuxMapType auxMap;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLParser.hh:   inline const G4GDMLAuxMapType* GetAuxMap() const;
    epsilon:geant4.10.04.p02 blyth$ g4-hh G4GDMLAuxListType
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLRead.hh:   const G4GDMLAuxListType* GetAuxList() const;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLRead.hh:   G4GDMLAuxListType auxGlobalList;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLAuxStructType.hh:typedef std::vector<G4GDMLAuxStructType> G4GDMLAuxListType;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLWriteStructure.hh:   std::map<const G4LogicalVolume*, G4GDMLAuxListType> auxmap;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:typedef std::map<G4LogicalVolume*,G4GDMLAuxListType> G4GDMLAuxMapType;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLReadStructure.hh:   G4GDMLAuxListType GetVolumeAuxiliaryInformation(G4LogicalVolume*) const;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLParser.hh:   inline G4GDMLAuxListType GetVolumeAuxiliaryInformation(G4LogicalVolume* lvol) const;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLParser.hh:   inline const G4GDMLAuxListType* GetAuxList() const;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLParser.hh:   G4GDMLAuxListType *rlist, *ullist;
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLWrite.hh:    void AddAuxInfo(G4GDMLAuxListType* auxInfoList, xercesc::DOMElement* element);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/persistency/gdml/include/G4GDMLWrite.hh:    G4GDMLAuxListType auxList;
    epsilon:geant4.10.04.p02 blyth$ 




::

    epsilon:src blyth$ grep AuxiliaryRead *.*
    G4GDMLRead.cc:AuxiliaryRead(const xercesc::DOMElement* const auxiliaryElement)
    G4GDMLRead.cc:        G4Exception("G4GDMLRead::AuxiliaryRead()",
    G4GDMLRead.cc:         G4Exception("G4GDMLRead::AuxiliaryRead()",
    G4GDMLRead.cc:         auxList->push_back(AuxiliaryRead(child));

           

    G4GDMLRead.cc:        auxGlobalList.push_back(AuxiliaryRead(child));
    G4GDMLReadStructure.cc:        { auxList.push_back(AuxiliaryRead(child)); } else
           from volume elements 

    epsilon:src blyth$ 



    322 void G4GDMLRead::UserinfoRead(const xercesc::DOMElement* const userinfoElement)
    323 {
    324 #ifdef G4VERBOSE
    325    G4cout << "G4GDML: Reading userinfo..." << G4endl;
    326 #endif
    327    for (xercesc::DOMNode* iter = userinfoElement->getFirstChild();
    328         iter != 0; iter = iter->getNextSibling())
    329    {
    330       if (iter->getNodeType() != xercesc::DOMNode::ELEMENT_NODE)  { continue; }
    331 
    332       const xercesc::DOMElement* const child
    333             = dynamic_cast<xercesc::DOMElement*>(iter);
    334       if (!child)
    335       {
    336         G4Exception("G4GDMLRead::UserinfoRead()",
    337                     "InvalidRead", FatalException, "No child found!");
    338         return;
    339       }
    340       const G4String tag = Transcode(child->getTagName());
    341 
    342       if (tag=="auxiliary")
    343       {
    344         auxGlobalList.push_back(AuxiliaryRead(child));
    345       }

    474 const G4GDMLAuxListType* G4GDMLRead::GetAuxList() const
    475 {
    476    return &auxGlobalList;
    477 }


    111 void G4GDMLWrite::UserinfoWrite(xercesc::DOMElement* gdmlElement)
    112 {
    113   if(auxList.size()>0)
    114   {
    115 #ifdef G4VERBOSE
    116     G4cout << "G4GDML: Writing userinfo..." << G4endl;
    117 #endif
    118     userinfoElement = NewElement("userinfo");
    119     gdmlElement->appendChild(userinfoElement);
    120     AddAuxInfo(&auxList, userinfoElement);
    121   }
    122 }


    352 void G4GDMLWrite::AddAuxiliary(G4GDMLAuxStructType myaux)
    353 {
    354    auxList.push_back(myaux);
    355 }

    epsilon:src blyth$ grep AddAuxiliary *.*
    G4GDMLParser.cc:    AddAuxiliary(raux);
    G4GDMLWrite.cc:void G4GDMLWrite::AddAuxiliary(G4GDMLAuxStructType myaux)

    epsilon:src blyth$ g4-cc AddAuxiliary
    /usr/local/opticks_externals/g4.build/geant4.10.04.p02/source/persistency/gdml/src/G4GDMLParser.cc:    AddAuxiliary(raux);
    /usr/local/opticks_externals/g4.build/geant4.10.04.p02/source/persistency/gdml/src/G4GDMLWrite.cc:void G4GDMLWrite::AddAuxiliary(G4GDMLAuxStructType myaux)


Looks like can use GDML like below::  

      <gdml>
           <userinfo>
               <auxiliary auxtype="type" auxvalue="value" auxunit="" />
           </userinfo>
      </gdml>


::

    160 inline
    161 G4GDMLAuxListType
    162 G4GDMLParser::GetVolumeAuxiliaryInformation(G4LogicalVolume* logvol) const
    163 {
    164   return reader->GetVolumeAuxiliaryInformation(logvol);
    165 }
    166 
    167 inline
    168 const G4GDMLAuxMapType* G4GDMLParser::GetAuxMap() const
    169 {
    170   return reader->GetAuxMap();
    171 }
    172 
    173 inline
    174 const G4GDMLAuxListType* G4GDMLParser::GetAuxList() const
    175 {
    176   return reader->GetAuxList();
    177 }
    178 
    179 inline
    180 void G4GDMLParser::AddAuxiliary(G4GDMLAuxStructType myaux)
    181 {
    182   return writer->AddAuxiliary(myaux);
    183 }





g4-cls G4GDMLWriteStructure::

    95    std::map<const G4LogicalVolume*, G4GDMLAuxListType> auxmap;

    580 void
    581 G4GDMLWriteStructure::AddVolumeAuxiliary(G4GDMLAuxStructType myaux,
    582                                          const G4LogicalVolume* const lvol)
    583 {
    584   std::map<const G4LogicalVolume*,
    585            G4GDMLAuxListType>::iterator pos = auxmap.find(lvol);
    586 
    587   if (pos == auxmap.end())  { auxmap[lvol] = G4GDMLAuxListType(); }
    588 
    589   auxmap[lvol].push_back(myaux);
    590 }

g4-cls G4GDMLAuxStructType::

    042 struct G4GDMLAuxStructType
     43 {
     44    G4String type;
     45    G4String value;
     46    G4String unit;
     47    std::vector<G4GDMLAuxStructType>* auxList;
     48 };
     49 
     50 typedef std::vector<G4GDMLAuxStructType> G4GDMLAuxListType;

g4-cls G4GDMLParser::

    119    inline G4VPhysicalVolume* GetWorldVolume(const G4String& setupName="Default") const;
    120    inline G4GDMLAuxListType GetVolumeAuxiliaryInformation(G4LogicalVolume* lvol) const;
    121    inline const G4GDMLAuxMapType* GetAuxMap() const;
    122    inline const G4GDMLAuxListType* GetAuxList() const;
    123    inline void AddAuxiliary(G4GDMLAuxStructType myaux);





EON

}


g4dev-notes-misc(){ cat << EON


Not finding xercesc
--------------------

::

    In file included from /usr/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/gdml/include/G4GDMLReadDefine.hh:45:
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/gdml/include/G4GDMLRead.hh:42:10: fatal error: 'xercesc/parsers/XercesDOMParser.hpp' file not found
    #include <xercesc/parsers/XercesDOMParser.hpp>
             ^
    1 error generated.

::

	/opticks/externals/g4/geant4_10_02_p01/source/persistency/gdml/include -I/home/blyth/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/mctruth/include  -W -Wall -pedantic -Wno-non-virtual-dtor -Wno-long-long -Wwrite-strings -Wpointer-arith -Woverloaded-virtual -Wno-variadic-macros -Wshadow -pipe -DG4USE_STD11 -O2 -g -fPIC   -std=c++11 -o CMakeFiles/G4persistency.dir/mctruth/src/G4VPHitsCollectionIO.cc.o -c /home/blyth/local/opticks/externals/g4/geant4_10_02_p01/source/persistency/mctruth/src/G4VPHitsCollectionIO.cc
	gmake[2]: *** No rule to make target `/home/blyth/local/opticks/externals/lib/libxerces-c-3-1.so', needed by `BuildProducts/lib64/libG4persistency.so'.  Stop.
	gmake[2]: Leaving directory `/home/blyth/local/opticks/externals/g4/geant4_10_02_p01.Debug.build'
	gmake[1]: *** [source/persistency/CMakeFiles/G4persistency.dir/all] Error 2
	gmake[1]: Leaving directory `/home/blyth/local/opticks/externals/g4/geant4_10_02_p01.Debug.build'
	gmake: *** [all] Error 2
	-bash: /home/blyth/local/opticks/externals/bin/geant4.sh: No such file or directory
	=== g4-export-ini : writing G4 environment to /home/blyth/local/opticks/externals/config/geant4.ini
	[blyth@localhost geant4_10_02_p01.Debug.build]$ 




Expat
-------

::

    yum install expat-devel





EON
}
