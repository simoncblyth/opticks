Hi Hans, 

> hope you are doing fine. The last few days it's been like spring here in
> Chicago which is very unusual. 
> 
> I was bench-marking the new opticks and was disappointed that the observed
> speed ups were much lower than what we observed previously. So I decided to
> check with the profiler to see what's going on. 
>
> It looks like between calls to
> opticks there are long periods where both CPU and GPU are idle. 
> 
> Any idea what could be going on?
> 
> thanks Hans 

To investigate I suggest you configure logging with eg "export G4CXOpticks=INFO"
or most other Opticks class/struct names.
Then using something like bin/log.sh bin/log.py to parse the logs you can 
identify where the time is being spent. 
You can also use Opticks plog logging external to Opticks by following 
the example of eg G4CX_LOG.{hh,cc} or any Opticks package. 
For the logging from your package to appear you will need to do the equivalent 
of what syslog/OPTICKS_LOG.hh does in the main to initialize the logger for 
your package.     
Alternatively you can arrange your logfile parsing to parse timestamps 
from some other logging system. 

In my experience parsing logfile time stamps is the most effective way of 
pinpointing time sinks, because after logging setup it is not invasive and 
you can easily add more logging wherever needed. Also when using Opticks 
logging it is easy to configure simply by setting envvars with the names 
of the classes/structs.
  
The more manual approach of using something like sysrap/schrono.h to make 
duration measurements is also an option, but that approach entails adding 
so much code that its only appropriate for monitoring slowdown prone bits
of code. 

> I have attached a screenshot and the profiler file which you can visualize with: 
> nvvp CaTS.nvpp

To make sense of the nvvp outputs I would start by establishing the 
correspondence between what that shows and the results of logfile parsing. 

My laptop CUDA+nvvp is too old to visualize your profile (it segments), 
but from what I can glean from the image and what you say it suggests the issue
is not with the ray tracing ?  

Logfile parsing will be able to confirm/deny that by measuring the duration 
of each launch call and the time between each launch. 

If the time were being spent within the launches then the prime suspect 
would be the geometry. Actually its the acceleration structure AS created from 
the geometry that is taking the time, but as changing the geometry 
and its modelling is the only way to change the AS they come to the same thing.

The way Opticks factorizes geometry into repeated instances can have a 
huge impact on performance.  While it depends on your geometry generally 
more instancing is better for performance.
The output from U4TreeCreateTest.sh is included below.
That test starts from GDML and does the same factorization
that G4CXOpticks::SetGeometry does.
The output shows that the JUNO geometry gets factorized into 10 compound solids 
with repeat counts of : 25600, 12615, 4997, 2400, ... corresponding to each type of PMT. 
Parts of the geometry that are not repeated 
enough to pass instancing cuts such as the world box end up in the global remainder. 
If your geometry is not being factorized everything will end up in the global remainder
compound solid and your ray tracing performance will be poor. 
  

Opticks has capabilities to create partial geometries on-the-fly with 
specific LV solids included/excluded. Comparing rendering times for these 
partial geometries allows specific LV solids that result in slow downs 
to be identified.     

See also some general notes on performance in:

https://bitbucket.org/simoncblyth/opticks/src/master/notes/performance.rst

Simon



Output from u4/tests/U4TreeCreateTest.sh 


epsilon:~ blyth$ u4
/Users/blyth/opticks/u4
epsilon:u4 blyth$ cd tests
epsilon:tests blyth$ ./U4TreeCreateTest.sh 
G4GDML: Reading '/Users/blyth/.opticks/GEOM/J007/origin.gdml'...
G4GDML: Reading definitions...
G4GDML: Reading materials...
G4GDML: Reading solids...
G4GDML: Reading structure...
G4GDML: Reading setup...
G4GDML: Reading '/Users/blyth/.opticks/GEOM/J007/origin.gdml' done!
G4GDMLParser::read             yielded chars :  cout      0 cerr      0 : set VERBOSE to see them 
[ U4Tree::Create 
[ stree::factorize 
[ stree::classifySubtrees 
] stree::classifySubtrees 
[ stree::disqualifyContainedRepeats 
] stree::disqualifyContainedRepeats  disqualify.size 35
[ stree::sortSubtrees 
] stree::sortSubtrees 
[ stree::enumerateFactors 
] stree::enumerateFactors 
[ stree::labelFactorSubtrees num_factor 9
stree::labelFactorSubtreessfactor index   0 freq  25600 sensors      0 subtree      5 olvid    133 freq*subtree  128000 sub [c0db131677b9e8d008ccbc042096e1af] outer_node.size 25600
stree::labelFactorSubtreessfactor index   1 freq  12615 sensors      0 subtree     11 olvid    128 freq*subtree  138765 sub [de92884ef346a7c0abd0956084dd516c] outer_node.size 12615
stree::labelFactorSubtreessfactor index   2 freq   4997 sensors      0 subtree     14 olvid    117 freq*subtree   69958 sub [b82ec07ef5c7e190f9cb8ae3724b14a8] outer_node.size 4997
stree::labelFactorSubtreessfactor index   3 freq   2400 sensors      0 subtree      6 olvid    145 freq*subtree   14400 sub [a3d2a83512e433e417246371953857a3] outer_node.size 2400
stree::labelFactorSubtreessfactor index   4 freq    590 sensors      0 subtree      1 olvid     98 freq*subtree     590 sub [c051c1bb98b71ccb15b0cf9c67d143ee] outer_node.size 590
stree::labelFactorSubtreessfactor index   5 freq    590 sensors      0 subtree      1 olvid     99 freq*subtree     590 sub [5e01938acb3e0df0543697fc023bffb1] outer_node.size 590
stree::labelFactorSubtreessfactor index   6 freq    590 sensors      0 subtree      1 olvid    100 freq*subtree     590 sub [cdc824bf721df654130ed7447fb878ac] outer_node.size 590
stree::labelFactorSubtreessfactor index   7 freq    590 sensors      0 subtree      1 olvid    101 freq*subtree     590 sub [3fd85f9ee7ca8882c8caa747d0eef0b3] outer_node.size 590
stree::labelFactorSubtreessfactor index   8 freq    504 sensors      0 subtree    130 olvid     11 freq*subtree   65520 sub [7d9a644fae10bdc1899c0765077e7a33] outer_node.size 504
] stree::labelFactorSubtrees 
stree::collectRemainderNodes rem.size 3089
stree::desc_factor
sfactor::Desc num_factor 9
sfactor index   0 freq  25600 sensors      0 subtree      5 olvid    133 freq*subtree  128000 sub [c0db131677b9e8d008ccbc042096e1af]
sfactor index   1 freq  12615 sensors      0 subtree     11 olvid    128 freq*subtree  138765 sub [de92884ef346a7c0abd0956084dd516c]
sfactor index   2 freq   4997 sensors      0 subtree     14 olvid    117 freq*subtree   69958 sub [b82ec07ef5c7e190f9cb8ae3724b14a8]
sfactor index   3 freq   2400 sensors      0 subtree      6 olvid    145 freq*subtree   14400 sub [a3d2a83512e433e417246371953857a3]
sfactor index   4 freq    590 sensors      0 subtree      1 olvid     98 freq*subtree     590 sub [c051c1bb98b71ccb15b0cf9c67d143ee]
sfactor index   5 freq    590 sensors      0 subtree      1 olvid     99 freq*subtree     590 sub [5e01938acb3e0df0543697fc023bffb1]
sfactor index   6 freq    590 sensors      0 subtree      1 olvid    100 freq*subtree     590 sub [cdc824bf721df654130ed7447fb878ac]
sfactor index   7 freq    590 sensors      0 subtree      1 olvid    101 freq*subtree     590 sub [3fd85f9ee7ca8882c8caa747d0eef0b3]
sfactor index   8 freq    504 sensors      0 subtree    130 olvid     11 freq*subtree   65520 sub [7d9a644fae10bdc1899c0765077e7a33]
 tot_freq_subtree  419003

] stree::factorize 
[ U4Tree::identifySensitive 
[ U4Tree::identifySensitiveInstances num_factor 9 st.sensor_count 0
U4Tree::identifySensitiveInstances factor 0 fac.sensors 0
U4Tree::identifySensitiveInstances factor 1 fac.sensors 0
U4Tree::identifySensitiveInstances factor 2 fac.sensors 0
U4Tree::identifySensitiveInstances factor 3 fac.sensors 0
U4Tree::identifySensitiveInstances factor 4 fac.sensors 0
U4Tree::identifySensitiveInstances factor 5 fac.sensors 0
U4Tree::identifySensitiveInstances factor 6 fac.sensors 0
U4Tree::identifySensitiveInstances factor 7 fac.sensors 0
U4Tree::identifySensitiveInstances factor 8 fac.sensors 0
] U4Tree::identifySensitiveInstances num_factor 9 st.sensor_count 0
[ U4Tree::identifySensitiveGlobals st.sensor_count 0 remainder.size 3089
] U4Tree::identifySensitiveGlobals  st.sensor_count 0 remainder.size 3089
[ stree::reorderSensors
] stree::reorderSensors sensor_count 0
] U4Tree::identifySensitive st.sensor_count 0
stree::add_inst i   0 gas_idx   1 nodes.size   25600
stree::add_inst i   1 gas_idx   2 nodes.size   12615
stree::add_inst i   2 gas_idx   3 nodes.size    4997
stree::add_inst i   3 gas_idx   4 nodes.size    2400
stree::add_inst i   4 gas_idx   5 nodes.size     590
stree::add_inst i   5 gas_idx   6 nodes.size     590
stree::add_inst i   6 gas_idx   7 nodes.size     590
stree::add_inst i   7 gas_idx   8 nodes.size     590
stree::add_inst i   8 gas_idx   9 nodes.size     504
] U4Tree::Create 
main@34:  save stree to FOLD /tmp/blyth/opticks/U4TreeCreateTest
[ stree::save_ /tmp/blyth/opticks/U4TreeCreateTest/stree
] stree::save_ /tmp/blyth/opticks/U4TreeCreateTest/stree

[stree::desc level 1 sensor_count 0 nds 422092 rem 3089 m2w 422092 w2m 422092 gtd 422092 digs 422092 subs 422092 soname 150 factor 9
 stree.desc.subs_freq 



