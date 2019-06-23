CC  = icc
AR  = xiar rcs

SIMINT_DIR = /home/mkurisu/Workspace/simint/build-avx/install
SIMINT_LIB = $(SIMINT_DIR)/lib/libsimint.a

INC     = -I. -I$(SIMINT_DIR)/include
CFLAGS  = -std=gnu99 -Wall -g -qopenmp -O3 -xHost -mkl

CMSLIB  = CMS.a
CMSTEST = test_CMS.x

all: $(CMSLIB) $(CMSTEST)

$(CMSLIB): CMS.o
	$(AR) $@ $^

$(CMSTEST): $(CMSLIB) test_CMS.o H2ERI_shell_operations.o
	$(CC) $(CFLAGS) -o $(CMSTEST) test_CMS.o H2ERI_shell_operations.o $(CMSLIB) $(SIMINT_LIB) 

CMS.o: CMS.h CMS.c
	$(CC) $(INC) $(CFLAGS) CMS.c -c

H2ERI_shell_operations.o: CMS.h H2ERI_shell_operations.c
	$(CC) $(INC) $(CFLAGS) H2ERI_shell_operations.c -c

test_CMS.o: CMS.h test_CMS.c
	$(CC) $(INC) $(CFLAGS) test_CMS.c -c

clean:
	rm *.o *.a