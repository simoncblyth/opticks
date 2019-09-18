# this gets sourced by opticks-env from opticks-

integration-source(){ echo $BASH_SOURCE ; }
integration-dir(){    echo $(dirname $BASH_SOURCE) ; }
integration-vi(){     vi $BASH_SOURCE ; }

tviz-(){         . $(integration-dir)/tests/tviz.bash      && tviz-env $* ; }
tpmt-(){         . $(integration-dir)/tests/tpmt.bash      && tpmt-env $* ; }
trainbow-(){     . $(integration-dir)/tests/trainbow.bash  && trainbow-env $* ; }
tnewton-(){      . $(integration-dir)/tests/tnewton.bash   && tnewton-env $* ; }
tprism-(){       . $(integration-dir)/tests/tprism.bash    && tprism-env $* ; }
tbox-(){         . $(integration-dir)/tests/tbox.bash      && tbox-env $* ; }
treflect-(){     . $(integration-dir)/tests/treflect.bash  && treflect-env $* ; }
twhite-(){       . $(integration-dir)/tests/twhite.bash    && twhite-env $* ; }
tlens-(){        . $(integration-dir)/tests/tlens.bash     && tlens-env $* ; }
tg4gun-(){       . $(integration-dir)/tests/tg4gun.bash    && tg4gun-env $* ; }
tlaser-(){       . $(integration-dir)/tests/tlaser.bash    && tlaser-env $* ; }
tboxlaser-(){    . $(integration-dir)/tests/tboxlaser.bash && tboxlaser-env $* ; }
tdefault-(){     . $(integration-dir)/tests/tdefault.bash  && tdefault-env $* ; }
tconcentric-(){  . $(integration-dir)/tests/tconcentric.bash  && tconcentric-env $* ; }
tboolean-(){     . $(integration-dir)/tests/tboolean.bash  && tboolean-env $* ; }
t-(){            . $(integration-dir)/tests/t.bash         && t-env $* ; }

tboolean-bib-(){ . $(integration-dir)/tests/tboolean-bib.bash  && tboolean-bib-env $* ; }
tjuno-(){        . $(integration-dir)/tests/tjuno.bash  && tjuno-env $* ; }
tgltf-(){        . $(integration-dir)/tests/tgltf.bash  && tgltf-env $* ; }





