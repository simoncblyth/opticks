investigate-sams-x375-a-solid-that-breaks-balancing
====================================================


Issue
-------

* https://github.com/63990/Opticks_install_guide
* https://groups.io/g/opticks/topic/geometry_balancing/32989642


Investigation
----------------

* Added GDML snippet reading capabaility to X4GDMLParser

::

   [blyth@localhost extg4]$ X4GDMLParserTest > /tmp/out

    ## tree is very big, so using nowrap in vim is handy 


::

    Hi Sam,

    I had a look at x375  with X4GDMLParserTest. x375 is a height 253 tree with 507 nodes. 
    I hope this is not an important piece of geometry for your photons because even if I
    succeed to convert it to a balanced GPU tree I expect the performance will be terrible.
    I expect your Geant4 performance with this will be really terrible also. 

       2019-09-26 21:30:38.255 INFO  [182477] [X4SolidStore::Dump@49] NTreeAnalyse height 253 count 507

    di : differernce
    cy : cylinder

                                                                                                                di
     599
     600                                                                                               di          cy
     601
     602                                                                                       di          cy
     603
     604                                                                               di          cy
     605
     606                                                                       di          cy
     607
     608                                                               di          cy
     609
     610                                                       di          cy
     611
     612                                               di          cy
     613
     614                                       di          cy

    ~ 500 lines like this

    1038                                                                               di          cy
    1039
    1040                                                                       di          cy
    1041
    1042                                                               di          cy
    1043
    1044                                                       di          cy
    1045
    1046                                               di          cy
    1047
    1048                                       di          cy
    1049
    1050                               di          cy
    1051
    1052                       di          cy
    1053
    1054               di          cy
    1055
    1056       di          cy
    1057
    1058   cy      cy
    1059


    Regards the number of solids, I mean how many that are used in logical volumes.
    Most of the many thousands of solids are just constituents of booleans, its the number
    of roots of the trees that matters. 

    Simon




