#!/bin/bash

index=${1}/_index.php
details=${1}/details.txt

echo "<html>" > $index
echo "<body>" >> $index
echo "<center><br>" >> $index

for pic in $1/*.pdf; do
    echo $pic
    convert -density 100 -trim ${pic} ${pic%%.pdf}.png
    basepic=$(basename $pic)
    echo "<div style='position:relative;z-index:1;display:inline'>" >> $index
    echo "<a href='${basepic}'><img src='${basepic%%.pdf}.png' /></a>" >> $index

    # if we have details file, cat and grep for detail
    if [ -f $details ]; then
        echo "<div style='position:absolute;z-index:2;right:50px;bottom:390px;width:250px;'>" >> $index
        echo $(cat $details | grep $basepic | cut -d ':' -f2) >> $index
        echo "</div>" >> $index
    fi

    echo "</div>" >> $index
    echo "" >> $index
done

echo "<center>" >> $index
echo "<body>" >> $index
echo "<html>" >> $index

# wait
chmod -R a+r $1
scp -rp $1 namin@web.physics.ucsb.edu:~/public_html/dump/
echo "web.physics.ucsb.edu/~namin/dump/${index}"
