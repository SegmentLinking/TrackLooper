g++ -I $PWD $(root-config --cflags --libs) -fopenmp -O3 -lm -lgomp -o duprm DupRM.cc -g
