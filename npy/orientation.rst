NPY Orientation : array creation, updating and persisting
==========================================================

* :doc:`../docs/orientation`

* https://bitbucket.org/simoncblyth/opticks/src/master/npy/
* https://bitbucket.org/simoncblyth/opticks/src/master/npy/NPY.cpp


NPY
   holds simple arrays with metadata, used for both geometry and event data. Provides **.npy** format load/save 
   allowing loading the arrays into python::

       python -c "import sys, numpy as np ; print(np.load(sys.argv[1]))" /path/to/array.npy

NNode
   basis of the CSG implementation 

NCSG 
   primitive or composite shape holding a tree of NNode, held by ggeo/GParts :doc:`../ggeo/orientation`

NCSGData
   mechanics of persisting shapes into NPY arrays







  

