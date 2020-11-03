mkdir -p plots/mtv_eff/
mkdir -p plots/dump/
cp index.php plots/mtv_eff/

python plot_efficiency.py $1 $2
