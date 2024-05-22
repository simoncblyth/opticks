GDXML_XercesC_linking_issue
===========================


This issue has history, it typically resurfaces for each new Geant4 version::

* :doc:`gdxml-xercesc-linux-linking`


This time doing clean build of gdxml fixed it. 

Effectively what was doing was changing geant4 version. Such a major
change can only succeed with a clean build. Due to this 
and other issues changed opticks-full-make to use om-cleaninstall


::

    Solution
    ----------

    Make gdxml depend in G4 at CMake level only, just so have access to G4Persistency 
    target from which to grab the consistent XercesC version. 


    Symptoms
    ----------

    1. config fails to find the G4persistency target 



    -- Looking for pthread_create in pthread - found
    -- Found Threads: TRUE  

    -- FindOpticksXercesC.cmake. Did not find G4persistency target : so look for system XercesC or one provided by cmake arguments 
    -- looking for XercescC using XERCESC_INCLUDE_DIR or system paths 

    ## THIS USUALLY A PROBLEM 

    -- find_path looking for SAXParser.hpp yields OpticksXercesC_INCLUDE_DIR /usr/include
    -- Configuring GDXMLTest
 


::

    [ 71%] Built target GDXML
    [ 85%] Building CXX object tests/CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o
    [100%] Linking CXX executable GDXMLTest
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o: in function `xercesc_3_2::DTDEntityDecl::~DTDEntityDecl()':
    /home/blyth/junotop/ExternalLibs/Xercesc/3.2.4/include/xercesc/validators/DTD/DTDEntityDecl.hpp:162: undefined reference to `xercesc_3_2::XMLEntityDecl::~XMLEntityDecl()'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o: in function `xercesc_3_2::DTDEntityDecl::~DTDEntityDecl()':
    /home/blyth/junotop/ExternalLibs/Xercesc/3.2.4/include/xercesc/validators/DTD/DTDEntityDecl.hpp:162: undefined reference to `xercesc_3_2::XMemory::operator delete(void*)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o: in function `xercesc_3_2::HandlerBase::fatalError(xercesc_3_2::SAXParseException const&)':
    /home/blyth/junotop/ExternalLibs/Xercesc/3.2.4/include/xercesc/sax/HandlerBase.hpp:398: undefined reference to `xercesc_3_2::SAXParseException::SAXParseException(xercesc_3_2::SAXParseException const&)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /home/blyth/junotop/ExternalLibs/Xercesc/3.2.4/include/xercesc/sax/HandlerBase.hpp:398: undefined reference to `xercesc_3_2::SAXParseException::~SAXParseException()'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: /home/blyth/junotop/ExternalLibs/Xercesc/3.2.4/include/xercesc/sax/HandlerBase.hpp:398: undefined reference to `typeinfo for xercesc_3_2::SAXParseException'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213DTDEntityDeclE[_ZTVN11xercesc_3_213DTDEntityDeclE]+0x20): undefined reference to `xercesc_3_2::DTDEntityDecl::isSerializable() const'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213DTDEntityDeclE[_ZTVN11xercesc_3_213DTDEntityDeclE]+0x28): undefined reference to `xercesc_3_2::DTDEntityDecl::serialize(xercesc_3_2::XSerializeEngine&)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213DTDEntityDeclE[_ZTVN11xercesc_3_213DTDEntityDeclE]+0x30): undefined reference to `xercesc_3_2::DTDEntityDecl::getProtoType() const'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213XMLAttDefListE[_ZTVN11xercesc_3_213XMLAttDefListE]+0x20): undefined reference to `xercesc_3_2::XMLAttDefList::isSerializable() const'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213XMLAttDefListE[_ZTVN11xercesc_3_213XMLAttDefListE]+0x28): undefined reference to `xercesc_3_2::XMLAttDefList::serialize(xercesc_3_2::XSerializeEngine&)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTVN11xercesc_3_213XMLAttDefListE[_ZTVN11xercesc_3_213XMLAttDefListE]+0x30): undefined reference to `xercesc_3_2::XMLAttDefList::getProtoType() const'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: CMakeFiles/GDXMLTest.dir/GDXMLTest.cc.o:(.rodata._ZTIN11xercesc_3_213DTDEntityDeclE[_ZTIN11xercesc_3_213DTDEntityDeclE]+0x10): undefined reference to `typeinfo for xercesc_3_2::XMLEntityDecl'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XMLUni::fgXercescDefaultLocale'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::SAXParseException::getLineNumber() const'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::setDoSchema(bool)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `typeinfo for xercesc_3_2::XMLException'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XercesDOMParser::setErrorHandler(xercesc_3_2::ErrorHandler*)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::setDoNamespaces(bool)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::setValidationScheme(xercesc_3_2::AbstractDOMParser::ValSchemes)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::LocalFileFormatTarget::LocalFileFormatTarget(char const*, xercesc_3_2::MemoryManager*)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XMLString::release(char**, xercesc_3_2::MemoryManager*)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::getDocument()'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::parse(char const*)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XMemory::operator new(unsigned long)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XMLPlatformUtils::fgMemoryManager'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `typeinfo for xercesc_3_2::DOMException'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XMLPlatformUtils::Initialize(char const*, char const*, xercesc_3_2::PanicHandler*, xercesc_3_2::MemoryManager*)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XMLString::transcode(char const*, char16_t*, unsigned long, xercesc_3_2::MemoryManager*)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::AbstractDOMParser::setValidationSchemaFullChecking(bool)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XercesDOMParser::XercesDOMParser(xercesc_3_2::XMLValidator*, xercesc_3_2::MemoryManager*, xercesc_3_2::XMLGrammarPool*)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XMLString::transcode(char16_t const*, xercesc_3_2::MemoryManager*)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::XMLUni::fgDOMWRTFormatPrettyPrint'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libGDXML.so: undefined reference to `xercesc_3_2::DOMImplementationRegistry::getDOMImplementation(char16_t const*)'
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/GDXMLTest] Error 1
    make[1]: *** [tests/CMakeFiles/GDXMLTest.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all install : non-zero rc 2
    === om-all om-install : ERROR bdir /data/blyth/opticks_Debug/build/gdxml : non-zero rc 2
    === om-one-or-all install : non-zero rc 2
    === opticks-full : ERR from opticks-full-make
    [blyth@localhost opticks]$ 
