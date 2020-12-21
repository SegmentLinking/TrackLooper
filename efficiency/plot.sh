mkdir -p plots/mtv_eff/
mkdir -p plots/dump/
cp index.php plots/mtv_eff/

if [ -z $1 ];
then
    echo "Please provide sample type name"
    exit
fi

if [ -z $2 ];
then
    echo "Please provide github tag"
    exit
fi

python plot_efficiency.py $1 $2
cp index.html plots/
