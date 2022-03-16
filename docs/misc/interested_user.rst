Interested Users
==================



JUNO : How to prepare for using Opticks
-----------------------------------------

Using Opticks on the IHEP GPU cluster implies you want to 
use a centralized install of Opticks. Thus you have to wait 
until development has advanced to the stage of having 
such a production release. 

Prior to then you need to have a GPU workstation and software 
expertise if you want to contribute to the development of Opticks.

As for how else to prepare for using Opticks.  I would suggest 
you start using the NumPy package and other python data analysis 
tools such as matplotlib and pyvista 

Opticks makes heavy use of NPY serialization and such python
tools so validating Opticks really benefits from skills 
with these tools. (Skills with using ROOT are almost totally useless
for working with Opticks).
  
NumPy may seem like a simple package, but it really benefits from 
long experience.  So for a task that you would normally use ROOT for 
try using python analysis tools. 
To move data from C++ into .npy files using my NP.hh header. See: 

    https://github.com/simoncblyth/np/



External
----------


Thanks for your interest in Opticks. I encourage you to subscribe to the mailing list 
I will announce any major developments there. So you will know when to try again 
to get Opticks working for you.

opticks+subscribe@groups.io 
https://groups.io/g/opticks





