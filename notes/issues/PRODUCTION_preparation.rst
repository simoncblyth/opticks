PRODUCTION_preparation
========================


TODO : make changes to the standard cvmfs release scripts adding opticks
--------------------------------------------------------------------------

* consult with Tao on this : is that the right approch ? 


TODO : Test with newer : driver + CUDA + OptiX + compiler
------------------------------------------------------------

N ~/.local.bash::

     31 # follow lint example to use JUNO gcc830 /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v2r0-Pre0/quick-deploy-J21v2r0-Pre0.sh
     32 source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/bashrc
     33 source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/binutils/2.28/bashrc
     34 
     35 #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/bashrc
     36 #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bashrc


I dont recall exactly where the gcc 830 restriction came from (probably CUDA 10.1 ?)

* TODO: find notes on this


TODO : Preprocessor macros
------------------------------

Arrange that when the PRODUCTION macro 
is defined all the DEBUG macro blocks 
are turned off. 

For example::

    DEBUG_PIDX    ## qsim.h and all over 



