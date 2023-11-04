opticks-t-qudarap-fails
=========================

::

    60% tests passed, 8 tests failed out of 20
    Total Test time (real) =  13.81 sec
    The following tests FAILED:
          3 - QUDARapTest.QScintTest (INTERRUPT)
          4 - QUDARapTest.QCerenkovIntegralTest (INTERRUPT)
          5 - QUDARapTest.QCerenkovTest (Child aborted)
          7 - QUDARapTest.QSimTest (Child aborted)
          8 - QUDARapTest.QBndTest (Child aborted)
         10 - QUDARapTest.QPropTest (SEGFAULT)
         18 - QUDARapTest.QMultiFilmTest (INTERRUPT)
         20 - QUDARapTest.QPMTTest (INTERRUPT)
    Errors while running CTest


    The following tests FAILED:
          4 - QUDARapTest.QCerenkovIntegralTest (INTERRUPT)
          5 - QUDARapTest.QCerenkovTest (Child aborted)
          8 - QUDARapTest.QBndTest (Child aborted)
         10 - QUDARapTest.QPropTest (SEGFAULT)
         18 - QUDARapTest.QMultiFilmTest (INTERRUPT)
         20 - QUDARapTest.QPMTTest (SEGFAULT)

    The following tests FAILED:
          5 - QUDARapTest.QCerenkovTest (Child aborted)   ## failing for lack of bnd
          7 - QUDARapTest.QSimTest (Child aborted)
          8 - QUDARapTest.QBndTest (Child aborted)        ## failing for lack of optical 
         10 - QUDARapTest.QPropTest (SEGFAULT)         ## FIXED BY PATH UPDATES
         18 - QUDARapTest.QMultiFilmTest (INTERRUPT)
         20 - QUDARapTest.QPMTTest (INTERRUPT)


With full GEOM::

    The following tests FAILED:
          5 - QUDARapTest.QCerenkovTest (Child aborted)
          8 - QUDARapTest.QBndTest (Child aborted)
    Errors while running CTest

With minimal GEOM::

    The following tests FAILED:
          5 - QUDARapTest.QCerenkovTest (Child aborted)
          7 - QUDARapTest.QSimTest (Failed)
          8 - QUDARapTest.QBndTest (Child aborted)
    Errors while running CTest




