MPI-OptiX
===========

Context
----------

:doc:`multi-gpu-optix`


OptiX with MPI ?
-------------------

* have observed OptiX scaling to be limited to 4 GPUs, 
  forum messages confirm this :doc:`multi-gpu-optix`

* given 8 GPUs per cluster node perhaps using MPI could 
  harness two sets of four horses on one node, or more when 
  using multiple nodes.


MPI how difficult ? Can it work with slurm ?
--------------------------------------------------

* https://en.wikipedia.org/wiki/Message_Passing_Interface

* https://slurm.schedmd.com/mpi_guide.html#intel_mpi

* Rank 0 : interface to Geant4, collecting gensteps and distributing them to workers, integrating hits into collections  
* Rank 1,2,... :  worker for each set of 4 GPUs want to use


SLURM with MPI examples
--------------------------

* https://docs.computecanada.ca/wiki/Using_GPUs_with_Slurm


Tensorflow Distributed Training : tf.distribute.Strategy
-------------------------------------------------------------

* https://www.tensorflow.org/alpha/guide/distribute_strategy


:google:`OptiX MPI`
-----------------------

VTK : Visualization toolkit, uses both OptiX and MPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://vtk.org/


VMD : Visual Molecular Dynamics (John Stone) used MPI (also now VCA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.ks.uiuc.edu/Research/vmd/doxygen/vmdmpi.html
* http://on-demand.gputechconf.com/gtc/2015/presentation/S5246-David-McAllister.pdf


MPI vs Message Queue
----------------------

* https://stackoverflow.com/questions/6756630/python-openmpi-vs-rabbitmq


Hmm will need to compress NPY to do this
------------------------------------------

* https://msgpack.org/


