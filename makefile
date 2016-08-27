CC=g++
FLAG=-std=c++11 -fopenmp -msse4
LAYERS=include/layer/layer.h include/layer/input_layer.h include/layer/linear_layer.h include/layer/activ_layer.h include/layer/output_layer.h

data_test.o: include/data.h test/data_test.cc
	$(CC) $(FLAG) include/data.h test/data_test.cc -o data_test.o

linear_layer_test.o: include/data.h include/node.h include/layer/layer.h include/layer/linear_layer.h test/linear_layer_test.cc
	$(CC) $(FLAG) include/data.h include/node.h include/layer/layer.h include/layer/linear_layer.h test/linear_layer_test.cc -o linear_layer_test.o

input_layer_test.o: include/data.h include/node.h include/layer/layer.h include/layer/input_layer.h test/input_layer_test.cc
	$(CC) $(FLAG) include/data.h include/node.h include/layer/layer.h include/layer/input_layer.h test/input_layer_test.cc -o input_layer_test.o

layers_test.o: include/data.h include/node.h include/activ.h include/loss.h include/optimizer.h $(LAYERS) test/layers_test.cc
	$(CC) $(FLAG) include/data.h include/node.h include/activ.h include/loss.h include/optimizer.h $(LAYERS) test/layers_test.cc -o layers_test.o

mlp_test.o: include/model/mlp.h test/mlp_test.cc
	$(CC) $(FLAG) include/model/mlp.h test/mlp_test.cc -o mlp_test.o

mnist_test.o: include/data/mnist.h test/mnist_test.cc
	$(CC) $(FLAG) include/data/mnist.h test/mnist_test.cc -o mnist_test.o