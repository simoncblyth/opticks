Opticks Integration Tests
=============================

The below high level bash function tests all use the **op.sh** script :doc:`../bin/op` 
to simulate optical photons and save event files.  The functions then use 
python analysis scripts :doc:`../ana/index`   to compare events with each other 
and analytic expectations.

Note to see the plots produced by the tests during development you will 
need to use ipython and invoke them with **run** as shown below.  

.. code-block:: py

    delta:ana blyth$ ipython
    Python 2.7.11 (default, Dec  5 2015, 23:51:51) 
    Type "copyright", "credits" or "license" for more information.

    IPython 1.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    IPython profile: g4opticks

    In [1]: run tprism.py --tag 1
    tprism.py --tag 1
    INFO:__main__:sel prism/torch/  1 : TO BT BT SA 20160716-1941 /tmp/blyth/opticks/evt/prism/fdomtorch/1.npy 
    INFO:__main__:prism Prism(array([  60.,  300.,  300.,    0.]),Boundary Vacuum///GlassSchottF2 ) alpha 60.0  
    ...


.. toctree::

    tpmt 
    trainbow
    tnewton
    tprism



    tbox
