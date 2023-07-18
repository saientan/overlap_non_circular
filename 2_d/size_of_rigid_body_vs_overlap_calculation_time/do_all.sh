for abcd in 1 2 3 4 5 6 7 8 9
do
for N in 2 3 4 5 6 7 8 9 10 11 12 13 14    
do
cat>write_arb_coord.py<<EOF
f=open('coord.dat','w')
N=$N
for i in range(0,$N):
    f.write('0.0   '+str(-i*0.11)+'\n')
f.write('0.0 1.1102230246251565e-16'+'\n')
for i in range(0,$N):
    f.write('0.0   '+str(i*0.11)+'\n')

EOF

python write_arb_coord.py
python simple_translate_rotate_train.py
python compare_time.py >> time_taken_${abcd}.dat
done
done
#python plot_time.py

