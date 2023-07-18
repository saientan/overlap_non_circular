for i in {1..10}
do
#mkdir $i
cd $i
cp ../from_x_y_theta_to_vmd_readable.py ./
#sbatch sub.sh
python from_x_y_theta_to_vmd_readable.py
#python simple_MC_extended_object_NORMAL_compress.py
cd ..
done
