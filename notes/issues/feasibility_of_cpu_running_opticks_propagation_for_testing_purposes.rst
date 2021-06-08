feasibility_of_cpu_running_opticks_propagation_for_testing_purposes
======================================================================


1. GPU style curand access easily mocked on CPU via preprocessor macros 
  
   * its easy because there are very few calls to mock 

2. mocking rtTrace/optixTrace is the stumbling block

   * would need a CPU BVH imp, eg Embree : using that is far to involved 
   * perhaps a simple noddy implementation would be ok, the aim 
     is to work with few photons anyhow for debug-ability    



