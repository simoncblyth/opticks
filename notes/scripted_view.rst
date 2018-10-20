scripted_view
===============

ScriptedView with some simple script syntax to do things like:

* jump to a position (frame index, position in frame)  Bookmark ? (but those are somewhat ephemeral)
* run a parametric view animation for a certain time
* switch view modes during the animation (eg raytrace, rasterized)
* run another view animation ...
* adjust some view parameter, eg changing near, fov, scale 


Overview
----------

Everything scripting-wise (especially async) looks so heavyweight, nothing than can pull together in a few days.
So de-scope to just a sequence of animations strung together and controlled via simple txt lingo.

Hmm even that looks too heavy and inconvenient to use, maybe can just generalize TrackView into a FlightView where the
path is designed via a numpy array of (n,4,4) way points with : position, times, view directions, frame indices. 

Being able to design a flight path with NumPy outside the app seems a much more controlled approach, 
than bookmark interpolation : can then use matplotlib 3d plotting to design the flight path, 
but will need to export mm0 global geometry dimensions to know where to aim.
Dump the CSG of the large extent volumes in mm0, for 2d drafting.



Invent a language OR use python/lua/... : boost-python is the obvious path, but what about async execution ?
-------------------------------------------------------------------------------------------------------------

* :google:`embed script interface inside C++ application`

* https://www.codeguru.com/csharp/.net/net_asp/scripting/article.php/c10919/Add-Embedded-Scripting-to-Your-C-Application.htm

* https://articles.emptycrate.com/2016/03/26/so-you-want-to-embed-a-scripting-language.html

* http://chaiscript.com

* http://jx9.symisc.net

* https://docs.python.org/2/extending/embedding.html

* https://docs.python.org/2/extending/embedding.html

* https://docs.python.org/3/extending/embedding.html

* https://www.boost.org/doc/libs/1_68_0/libs/python/doc/html/index.html

* https://www.boost.org/doc/libs/1_68_0/libs/python/doc/html/article.html


BUT : what about async interface
-----------------------------------


* :google:`async embedded scripting language c++`


Coroutine
~~~~~~~~~~

* https://en.wikipedia.org/wiki/Coroutine


Python coroutines : asyncio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://docs.python.org/3/library/asyncio-task.html
* https://www.geeksforgeeks.org/coroutine-in-python/


Jinx
~~~~~~~

* https://www.jinx-lang.org/Tutorial.htm

Jinx also has a fundamentally different approach to script execution.  Every
script has its own private execution stack by default, meaning each script runs
as a co-routine.  This makes it incredibly simple to write a script that
executes over a specified period of time, an important feature for real-time
applications.  Naturally, you can use Lua this way too, but it requires a
significant amount of non-trivial boiler-plate code.





