G4OpticksTest_fork_memory_leak
=================================

Added simple per G4Opticks::propagateOptical call profile collection.

* VM steadily increasing at around 4MB per call to propagate. 


Plotting the leak
------------------- 

::

    epsilon:~ blyth$ cd /tmp/blyth/opticks/G4Opticks/tests/

    epsilon:tests blyth$ scp P:/tmp/simon/opticks/G4Opticks/tests/G4OpticksProfilePlot.npy .

    epsilon:tests blyth$ np.py G4OpticksProfilePlot.npy
    a :                                     G4OpticksProfilePlot.npy :            (1000, 4) : 1c0676926c9acdb982556aa220b126fe : 20210215-1225 

    epsilon:tests blyth$ ipython -i ~/opticks/g4ok/tests/G4OpticksProfilePlot.py
    [[71888.7   21316.36      0.        0.   ]
     [71889.41  21279.992     0.        0.   ]
     [71890.07  21247.656     0.        0.   ]
     ...
     [72650.75  24906.477     0.        0.   ]
     [72651.55  24900.863     0.        0.   ]
     [72652.22  24830.473     0.        0.   ]]
     delta:   3514.11 slope0:      3.51 
    line fit:  slope       3.62    intercept   21284.92 

    In [1]:  
     

