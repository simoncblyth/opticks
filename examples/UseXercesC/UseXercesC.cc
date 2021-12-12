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

#include <iostream>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

int main(int argc, char** argv )
{
    xercesc::XMLPlatformUtils::Initialize();
    xercesc::DOMDocument* doc; 
    XMLCh tempStr[10000];
     
    // G4GDMLWrite::Write   g4-;g4-cls G4GDMLWrite
    xercesc::XMLString::transcode("LS", tempStr, 9999);
    xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    xercesc::XMLString::transcode("Range", tempStr, 9999);
    xercesc::DOMImplementation* impl =
    xercesc::DOMImplementationRegistry::getDOMImplementation(tempStr);
    xercesc::XMLString::transcode("gdml", tempStr, 9999);
    doc = impl->createDocument(0,tempStr,0);
    xercesc::DOMElement* gdml = doc->getDocumentElement();
    assert(gdml);

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

    std::cout << "XERCES_VERSION_MAJOR " << XERCES_VERSION_MAJOR << std::endl ; 
    std::cout << "XERCES_VERSION_MINOR " << XERCES_VERSION_MINOR << std::endl ; 
    std::cout << "XERCES_VERSION_REVISION " << XERCES_VERSION_REVISION << std::endl ; 
    std::cout << "XERCES_FULLVERSIONDOT " << XERCES_FULLVERSIONDOT << std::endl ;

    return 0 ; 
}

