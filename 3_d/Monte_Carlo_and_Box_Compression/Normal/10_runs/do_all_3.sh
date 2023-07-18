for i in {1..10}
do
#mkdir $i
cd $i
cp ../extend_simple_MC_extended_object_NORMAL_compress.py ./
cp ../sub.sh ./
sbatch sub.sh
#python extend_simple_MC_extended_object_NORMAL_compress.py
#python simple_MC_extended_object_NORMAL_compress.py
cd ..
done
