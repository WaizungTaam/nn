CC=g++
FLAG=-std=c++11 -fopenmp -msse4
LAYERS=include/layer/layer.h include/layer/input_layer.h include/layer/linear_layer.h include/layer/activ_layer.h include/layer/output_layer.h

# mnist_test.o: include/data/mnist.h test/mnist_test.cc
	# $(CC) $(FLAG) include/data/mnist.h test/mnist_test.cc -o mnist_test.o

layer_test.o: include/layer.h test/layer_test.cc
	$(CC) $(FLAG)	include/layer.h test/layer_test.cc -o layer_test.o

sequential_test.o: include/sequential.h test/sequential_test.cc
	$(CC) $(FLAG) -g include/sequential.h test/sequential_test.cc -o sequential_test.o