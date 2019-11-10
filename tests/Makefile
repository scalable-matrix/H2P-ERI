CC  = icc
AR  = xiar rcs

SIMINT_DIR = /home/mkurisu/Workspace/simint/build-avx/install
SIMINT_LIB = $(SIMINT_DIR)/lib/libsimint.a
H2PERI_LIB = ../src/libH2PERI.a

INC    = -I../src/include/ -I$(SIMINT_DIR)/include/
CFLAGS = -std=gnu99 -Wall -g -qopenmp -O3 -xHost -mkl -DUSE_AVX

test_H2ERI = test_H2ERI.x

$(test_H2ERI): $(SIMINT_LIB) $(H2PERI_LIB) test_H2ERI.o
	$(CC) $(CFLAGS) -o $(test_H2ERI) test_H2ERI.o $(H2PERI_LIB) $(SIMINT_LIB) 

test_H2ERI.o: test_H2ERI.c
	$(CC) $(INC) $(CFLAGS) test_H2ERI.c -c

clean:
	rm *.o *.x