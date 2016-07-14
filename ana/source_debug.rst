Wavelength Distribution Debugging
====================================

Compares simulated photon wavelengths against blackbody expectation.


* still a hint of "ringing steps" from 200:400nm, but seems acceptable 
  (TODO: try increasing icdf bins from 1024 to identify) 


[ISSUE] wp last bin elevated
----------------------------- 

::

    In [69]: plt.close();plt.hist(wp, bins=200)

    ,  2215.,  2158.,  2046.,  2017.,  2052.,  2111.,  2565.]),


[FIXED] Bug with w0 sel.recwavelength(0)  
-----------------------------------------

Without selection sel.recwavelength(0) from ggv-newton:

* length of 500000

* three bin spike at lower bound around 60nm, comprising about 7000 photons
  (not present in the uncompressed wp)

  **FIXED WHEN AVOID WAVELENGTH DOMAIN DISCREPANCY BETWEEN SOURCES AND COMPRESSION**  

* plateau from 60~190 nm

  **MADE MUCH LESS OBJECTIONABLE BY INCREASING ICDF BINS FROM 256 TO 1024** 

* normal service resumes above 190nm with good
  match to Planck black body curve

* 256 unique linspaced values, a result of the compression:: 

    In [36]: np.allclose(np.linspace(60,820,256),np.unique(w))  # upper changed 810 to 820 by the fix
    Out[36]: True


