CC = gcc
CFLAGS = -g -std=c99 -Wall -O3 -I ./include
LDFLAGS = -flto -lm
	

run_main_accur : main_accur
	./$^
	

main_accur: main_stochastic_grad_accur.o kernel.o convolu.o nn.o CNN_backprop.o FC_backprop.o train.o backprop_runner.o mnist.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


%.o: ./src/%.c
	$(CC) $(CFLAGS) -c $^

clean:
	find . -type f -executable -delete
	rm -Rf *.o 


