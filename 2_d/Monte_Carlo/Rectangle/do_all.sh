for i in {1..10}
do
#python simple_MC_extended_object_ML_compress.py
python simple_MC_extended_object_NORMAL_compress.py
#cp time_taken_Decision_Tree time_taken_Decision_Tree_${i}
#cp time_taken_QDA time_taken_QDA_${i}
#cp time_taken_Naive_Bayes time_taken_Naive_Bayes_${i}
#cp time_taken_Grad_Boost time_taken_Grad_Boost_${i}
cp time_taken_NORMAL time_taken_NORMAL_${i}
done

