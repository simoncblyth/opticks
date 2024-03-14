opengl_visualization_resources
================================


OpenGL introductions
---------------------

Note that mathematics of view, projection etc.. is 
common to all graphics libraries. 
So studying OpenGL learning resources can help with any rendering 
not just OpenGL. 

* https://learnopengl.com/Introduction
* http://www.opengl-tutorial.org/



For arranging consistent ray trace and rasterized renders
-----------------------------------------------------------

* https://antongerdelan.net/opengl/raycasting.html


Debugging
----------

* https://learnopengl.com/In-Practice/Debugging

OpenGL expert on Stackoverflow
---------------------------------

* https://nicolbolas.github.io/oldtut/index.html

* https://nicolbolas.github.io/oldtut/Basics/Intro%20What%20is%20OpenGL.html

* https://nicolbolas.github.io/oldtut/Basics/Tut01%20Dissecting%20Display.html



Rendering Multiple Meshes
---------------------------

* https://www.reddit.com/r/opengl/comments/n9eagu/how_to_render_multiple_meshes_in_opengl/

Depending on the size of your models, you may be able to package all of them
into a single VBO and a single IBO. Then you can format a draw buffer and use
glMultiDrawElementsIndirect to render every single object with a single draw
call. That would probably be the most efficient way to do this.

But simply instancing all objects of the same model together, thus using 3-8
draw calls will probably be close to the same real world performance and
requires less work to set up :



* https://www.spacesimulator.net/tutorials/OpenGL_matrices_tutorial_3_3.html



