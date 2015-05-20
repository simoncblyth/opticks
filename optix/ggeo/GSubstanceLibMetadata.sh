#!/bin/bash -l

ggeoview-
idpath=$(ggeoview-idpath)
python ${0/.sh/.py} $idpath $*


