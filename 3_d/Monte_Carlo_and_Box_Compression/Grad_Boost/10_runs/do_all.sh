for i in {1..10}
do
mkdir $i
cd $i
cp ../* ./
#sbatch sub.sh
python simple_MC_extended_object_ML_compress.py &
cd ..
done
