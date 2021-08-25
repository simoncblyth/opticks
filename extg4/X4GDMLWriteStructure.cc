/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "X4GDMLWriteStructure.hh"
#include "X4GDMLParser.hh"
#include "BFile.hh"

#include <cstring>

#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/framework/MemBufFormatTarget.hpp>

#include "PLOG.hh"

const plog::Severity X4GDMLWriteStructure::LEVEL = PLOG::EnvLevel("X4GDMLWriteStructure", "DEBUG" ); 

X4GDMLWriteStructure::X4GDMLWriteStructure(bool refs)
{
    init(refs); 
}

// /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/persistency/gdml/src/G4GDMLWrite.cc


void X4GDMLWriteStructure::write(const G4VSolid* solid, const char* path )
{
    add(solid); 
    write(path); 
}

std::string X4GDMLWriteStructure::to_string( const G4VSolid* solid )
{
    add(solid); 
    return write("MEMBUF") ; 
}


/**
X4GDMLWriteStructure::init
---------------------------

At some point after 1070 geant4 removes fixed size tempStr member variable XMLCh tempStr[10000].
To avoid having to change code with Geant4 versions the below uses its own local_tempStr member variable.

**/

void X4GDMLWriteStructure::init(bool refs)
{

   SchemaLocation = "SchemaLocation";
   addPointerToName = refs ;
   xercesc::XMLString::transcode("LS", local_tempStr, 9999);
   xercesc::DOMImplementationRegistry::getDOMImplementation(local_tempStr);
   xercesc::XMLString::transcode("Range", local_tempStr, 9999);

   impl = xercesc::DOMImplementationRegistry::getDOMImplementation(local_tempStr);

   xercesc::XMLString::transcode("gdml", local_tempStr, 9999);
   doc = impl->createDocument(0,local_tempStr,0);

   gdml = doc->getDocumentElement();

   gdml->setAttributeNode(NewAttribute("xmlns:xsi",
                          "http://www.w3.org/2001/XMLSchema-instance"));
   gdml->setAttributeNode(NewAttribute("xsi:noNamespaceSchemaLocation",
                          SchemaLocation));

}



/**
X4GDMLWriteStructure::add
---------------------------


**/

void X4GDMLWriteStructure::add(const G4VSolid* solid )
{
   SolidsWrite(gdml);

   G4String type = solid->GetEntityType() ; 

   LOG(LEVEL) << type ;  

   if( type != "G4DisplacedSolid" )
   AddSolid( solid );
}

/**
X4GDMLWriteStructure::write
-----------------------------

Action depends on the value of the path argument:

NULL
    write gdml to stdout
MEMBUF 
    return string with the gdml
some/path/to/write.gdml 
    writes gdml to the given path 

**/

std::string X4GDMLWriteStructure::write(const char* path)
{

#if XERCES_VERSION_MAJOR >= 3
                                             // DOM L3 as per Xerces 3.0 API
    xercesc::DOMLSSerializer* writer = ((xercesc::DOMImplementationLS*)impl)->createLSSerializer();

    xercesc::DOMConfiguration *dc = writer->getDomConfig();
    dc->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);

#else

   xercesc::DOMWriter* writer = ((xercesc::DOMImplementationLS*)impl)->createDOMWriter();

   if (writer->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
       writer->setFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);

#endif


   xercesc::XMLFormatTarget* target = NULL ; 

   bool buf = false ; 

   if( path == NULL ) 
   {
       target = new xercesc::StdOutFormatTarget() ;
   }
   else if( strcmp(path,"MEMBUF") == 0 )
   {
       target = new xercesc::MemBufFormatTarget() ; 
       buf = true ; 
   }  
   else
   {
       std::string xpath = BFile::preparePath(path); 
      // NB logging output from here gets swallowed by stream redirection in X4GDMLParser   
       LOG(LEVEL) 
           << " path " << path
           << " xpath " << xpath
           ;

       target = new xercesc::LocalFileFormatTarget(xpath.c_str()) ;
   }

   try
   {
#if XERCES_VERSION_MAJOR >= 3
                                            // DOM L3 as per Xerces 3.0 API
      xercesc::DOMLSOutput *theOutput = ((xercesc::DOMImplementationLS*)impl)->createLSOutput();
      theOutput->setByteStream(target);
      writer->write(doc, theOutput);
#else
      writer->writeNode(target, *doc);
#endif
   }
   catch (const xercesc::XMLException& toCatch)
   {
      char* message = xercesc::XMLString::transcode(toCatch.getMessage());
      G4cout << "G4GDML: Exception message is: " << message << G4endl;
      xercesc::XMLString::release(&message);
      //return G4Transform3D::Identity;
   }
   catch (const xercesc::DOMException& toCatch)
   {
      char* message = xercesc::XMLString::transcode(toCatch.msg);
      G4cout << "G4GDML: Exception message is: " << message << G4endl;
      xercesc::XMLString::release(&message);
   }
   catch (...)
   {   
      G4cout << "G4GDML: Unexpected Exception!" << G4endl;
   }        


   std::string ret ; 
   if(buf)
   {
       ret = (char*)((xercesc::MemBufFormatTarget*)target)->getRawBuffer() ; 
   }

   delete target;
   writer->release();

   return ret ; 
}








