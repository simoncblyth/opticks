xercesc_dynamic_cast_error_visibility
========================================

::

    gxt
    ./G4CXOpticks_setGeometry_Test.sh
    ...

    G4GDML: Writing setup...
    G4GDML: Writing surfaces...
    G4GDML: Writing '/Users/blyth/.opticks/GEOM/FewPMT/origin_raw.gdml' done !
    2023-06-14 13:12:08.943493+0100 G4CXOpticks_setGeometry_Test[52472:14579023] dynamic_cast error 2: One or more of the following type_info's  has hidden visibility.  They should all have public visibility.   N11xercesc_3_17DOMNodeE, N11xercesc_3_116DOMElementNSImplE, N11xercesc_3_110DOMElementE.
    2023-06-14 13:12:08.943530+0100 G4CXOpticks_setGeometry_Test[52472:14579023] dynamic_cast error 2: One or more of the following type_info's  has hidden visibility.  They should all have public visibility.   N11xercesc_3_17DOMNodeE, N11xercesc_3_116DOMElementNSImplE, N11xercesc_3_110DOMElementE.
    2023-06-14 13:12:08.943542+0100 G4CXOpticks_setGeometry_Test[52472:14579023] dynamic_cast error 2: One or more of the following type_info's  has hidden visibility.  They should all have public visibility.   N11xercesc_3_17DOMNodeE, N11xercesc_3_113DOMAttrNSImplE, N11xercesc_3_17DOMAttrE.
    2023-06-14 13:12:08.943561+0100 G4CXOpticks_setGeometry_Test[52472:14579023] dynamic_cast error 2: One or more of the following type_info's  has hidden visibility.  They should all have public visibility.   N11xercesc_3_17DOMNodeE, N11xercesc_3_113DOMAttrNSImplE, N11xercesc_3_17DOMAttrE.
    2023-06-14 13:12:08.943637+0100 G4CXOpticks_setGeometry_Test[52472:14579023] dynamic_cast error 2: One or more of the following type_info's  has hidden visibility.  They should all have public visibility.   N11xercesc_3_17DOMNodeE, N11xercesc_3_113DOMAttrNSImplE, N11xercesc_3_17DOMAttrE.
    2023-06-14 13:12:08.943678+0100 G4CXOpticks_setGeometry_Test[52472:14579023] dynamic_cast error 2: One or more of the following type_info's  has hidden visibility.  They should all have public visibility.   N11xercesc_3_17DOMNodeE, N11xercesc_3_113DOMAttrNSImplE, N11xercesc_3_17DOMAttrE.
    2023-06-14 13:12:08.943749+0100 G4CXOpticks_setGeometry_Test[52472:14579023] dynamic_cast error 2: One or more of the following type_info's  has hidden visibility.  They should all have public visibility.   N11xercesc_3_17DOMNodeE, N11xercesc_3_113DOMAttrNSImplE, N11xercesc_3_17DOMAttrE.
    2023-06-14 13:12:08.947600+0100 G4CXOpticks_setGeometry_Test[52472:14579023] dynamic_cast error 2: One or more of the following type_info's  has hidden visibility.  They should all have public visibility.   N11xercesc_3_17DOMNodeE, N11xercesc_3_113DOMAttrNSImplE, N11xercesc_3_17DOMAttrE.
    2023-06-14 13:12:08.951 INFO  [14579023] [U4GDML::write@197]  Apply GDXML::Fix  rawpath /Users/blyth/.opticks/GEOM/FewPMT/origin_raw.gdml dstpath /Users/blyth/.opticks/GEOM/FewPMT/origin.gdml
    2023-06-14 13:12:08.951 INFO  [14579023] [CSGFoundry::save_@2195] /Users/blyth/.opticks/GEOM/FewPMT/CSGFoundry
    2023-06-14 13:12:08.953 INFO  [14579023] [CSGFoundry::save_@2217]  SSim::save /Users/blyth/.opticks/GEOM/FewPMT/CSGFoundry
    2023-06-14 13:12:08.968 ERROR [14579023] [G4CXOpticks::saveGeometry@572] skipped saving gg 
    2023-06-14 13:12:08.968 INFO  [14579023] [G4CXOpticks::saveGeometry@577] ] /Users/blyth/.opticks/GEOM/FewPMT
    2023-06-14 13:12:08.968 INFO  [14579023] [G4CXOpticks::setGeometry_@300] ] G4CXOpticks__setGeometry_saveGeometry 
    2023-06-14 13:12:08.968 INFO  [14579023] [G4CXOpticks::setGeometry_@303] [ fd 0x10ca82480




::

    epsilon:issues blyth$ c++filt N11xercesc_3_17DOMNodeE
    xercesc_3_1::DOMNode
    epsilon:issues blyth$ c++filt N11xercesc_3_116DOMElementNSImplE
    xercesc_3_1::DOMElementNSImpl
    epsilon:issues blyth$ c++filt N11xercesc_3_110DOMElementE
    xercesc_3_1::DOMElement
    epsilon:issues blyth$ c++filt N11xercesc_3_17DOMAttrE
    xercesc_3_1::DOMAttr


Maybe Darwin only issue ?

* https://stackoverflow.com/questions/51297638/how-to-fix-type-infos-has-hidden-visibility-they-should-all-have-public-visib




