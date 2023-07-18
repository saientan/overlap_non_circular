for i in {1..10}
do
#mkdir $i
cd $i
cp ../calculate_nematic_order_parameter.py ./
#sbatch sub.sh
python calculate_nematic_order_parameter.py
#python simple_MC_extended_object_NORMAL_compress.py
cd ..
done
