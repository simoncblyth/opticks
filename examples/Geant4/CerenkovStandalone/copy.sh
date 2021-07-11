#!/bin/bash 

srcs(){ cat << EOS
/tmp/G4Cerenkov_modifiedTest/BetaInverse_1.500_step_length_100000.000_ASIS/pngs/BetaInverse_1p5.png
/tmp/G4Cerenkov_modifiedTest/BetaInverse_1.500_step_length_100000.000_SKIP_CONTINUE/pngs/BetaInverse_1p5.png
EOS
}


dst_fold=/Users/blyth/simoncblyth.bitbucket.io
dst_rel=env/presentation

printf "# copy into dst_fold ${dst_fold} \n\n" 

for src in $(srcs) ; do 
   nam=${src/\/tmp\//}
   dst=${dst_fold}/${dst_rel}/${nam}
   echo mkdir -p $(dirname $dst)
   echo cp $src $dst 
done 

printf "\n\n\n"

printf "cat << EOP\n\n"
for src in $(srcs) ; do 
   nam=${src/\/tmp\//}
   printf "    Slide Name\n"
   printf "    /${dst_rel}/${nam} 1280px_720px \n\n"
done 

printf "EOP\n\n"



