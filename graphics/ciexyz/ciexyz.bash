# === func-gen- : graphics/ciexyz/ciexyz fgp graphics/ciexyz/ciexyz.bash fgn ciexyz fgh graphics/ciexyz
ciexyz-src(){      echo graphics/ciexyz/ciexyz.bash ; }
ciexyz-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ciexyz-src)} ; }
ciexyz-vi(){       vi $(ciexyz-source) ; }
ciexyz-env(){      elocal- ; }
ciexyz-usage(){ cat << EOU


Simple Analytic Approximations to the CIE XYZ Color Matching Functions

* http://www.ppsloan.org/publications/XYZJCGT.pdf


* http://mathematica.stackexchange.com/questions/57389/convert-spectral-distribution-to-rgb-color

* http://www.cvrl.org


Related

* https://www.fourmilab.ch/documents/specrend/




The CIE XYZ and xyY Color Spaces, Douglas A. Kerr

* http://dougkerr.net/Pumpkin/articles/CIE_XYZ.pdf

* color matching funcs chosen to make Y the luminance



RGB Color Space Conversion

* http://www.ryanjuckett.com/programming/rgb-color-space-conversion/


Gary M Meyer, Tutorial on Color Science

* http://zach.in.tu-clausthal.de/teaching/cg_literatur/tutorial_on_color_science.pdf



Relative Luminance
~~~~~~~~~~~~~~~~~~~~~

* https://en.m.wikipedia.org/wiki/Luminance_(relative)

For RGB color spaces that use the ITU-R BT.709 primaries (or sRGB, which
defines the same primaries), relative luminance can be calculated from linear
RGB components:

Y = 0.2126 R + 0.7152 G + 0.0722 B 

The formula reflects the luminosity function: green light 
contributes the most to the intensity perceived by humans, and blue light the least.

Observation
~~~~~~~~~~~~~

The above just plucks the Y row from the matrix::

    [sRGB/D65] RGB -> XYZ
    [[ 0.41245643  0.35757608  0.18043748]
     [ 0.21267285  0.71515217  0.072175  ]
     [ 0.0193339   0.11919203  0.95030407]]



Spectral Rendering
~~~~~~~~~~~~~~~~~~~~~~

* http://graphicsinterface.org/wp-content/uploads/gi1999-7.pdf

Stratified Wavelength Clusters for Efficient Spectral Monte Carlo Rendering
Glenn F. Evans Michael D. McCool


Color Space Refs
~~~~~~~~~~~~~~~~~

* https://en.m.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport
* https://en.m.wikipedia.org/wiki/Standard_illuminant
* https://en.m.wikipedia.org/wiki/SRGB

Colour Space Conversions
* http://www.poynton.com/PDFs/coloureq.pdf



Linear RGB to device RGB (gamma scaling)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In what circumstance is a manual gamma correction/uncorrection needed ? 


Chapter 24. The Importance of Being Linear
* http://http.developer.nvidia.com/GPUGems3/gpugems3_ch24.html


* http://stackoverflow.com/questions/10347995/srgb-textures-is-this-correct


sRGB and OpenGL

* http://web.archive.org/web/20140209181347/http://www.arcsynthesis.org/gltut/


Planck 
~~~~~~~~

In-band Radiance: Integrating the Planck Equation Over a Finite Range
* http://www.spectralcalc.com/blackbody/inband_radiance.html

A better presentation of Planck’s radiation law
* http://arxiv.org/pdf/1109.3822.pdf






EOU
}
ciexyz-dir(){ echo $(env-home)/graphics/ciexyz ; }
ciexyz-cd(){  cd $(ciexyz-dir); }

ciexyz-pdf(){
   local path=$(ciexyz-pdfpath)
   mkdir -p $(dirname $path)
   [ ! -f "$path" ] && curl -L http://www.ppsloan.org/publications/XYZJCGT.pdf -o $path
   open $path
}

ciexyz-pdfpath(){
   echo $LOCAL_BASE/env/graphics/ciexyz/XYZJCGT.pdf
}
ciexyz-lib(){
   echo $LOCAL_BASE/env/graphics/ciexyz/ciexyz.dylib
}

ciexyz-make-ctypes()
{
   ciexyz-cd
   local lib=$(ciexyz-lib)
   mkdir -p $(dirname $lib) 
   clang ciexyz.c -shared -o $lib
}

ciexyz-make-ufunc()
{
   type $FUNCNAME
   ciexyz-cd
   sudo python setup.py install
   sudo rm -rf build


}

ciexyz-make()
{
   ciexyz-make-ufunc
   ciexyz-test-ufunc
}

ciexyz-test-ufunc()
{
   ciexyz-cd
   python ciexyz_ufunc_test.py 
}


