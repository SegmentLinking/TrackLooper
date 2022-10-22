DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p plots/mtv/
cp $DIR/../misc/index.php plots/mtv/

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

python3 $DIR/../python/plot_performance.py $1 $2
python3 $DIR/../python/plot_tc_te_compare.py $1 $2
cp $DIR/../misc/index.html plots/
