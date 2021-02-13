LIB   = libH2PERI.a
OBJS  = CMS.o H2ERI_typedef.o H2ERI_shell_operations.o H2ERI_partition.o \
        H2ERI_build_H2.o H2ERI_build_Coulomb.o

USE_MKL      = 1
USE_OPENBLAS = 0

SIMINT_INSTALL_DIR   = /home/mkurisu/Workspace/simint/build-avx/install
H2PACK_INSTALL_DIR   = /home/mkurisu/Workspace/H2Pack
OPENBLAS_INSTALL_DIR = /home/mkurisu/Workspace/OpenBLAS-0.3.8/install

CC      = icc
DEF     = 
INC     = -I./include -I$(SIMINT_INSTALL_DIR)/include -I$(H2PACK_INSTALL_DIR)/include
CFLAGS  = $(INC) -Wall -g -std=gnu99 -O3 $(DEF)

ifeq ($(CC), icc)
AR      = xiar rcs
CFLAGS  += -qopenmp -xHost
endif

ifeq ($(CC), gcc)
AR      = ar rcs
CFLAGS  += -fopenmp -march=native -Wno-unused-result -Wno-unused-function
endif

ifeq ($(USE_MKL), 1)
DEF     += -DUSE_MKL
CFLAGS  += -mkl
endif

ifeq ($(USE_OPENBLAS), 1)
DEF     += -DUSE_OPENBLAS
INC     += -I$(OPENBLAS_INSTALL_DIR)/include
endif

$(LIB): $(OBJS) 
	$(AR) $@ $^

CMS.o: include/*.h CMS.c
	$(CC) $(CFLAGS) CMS.c -c 

H2ERI_typedef.o: include/*.h H2ERI_typedef.c
	$(CC) $(CFLAGS) H2ERI_typedef.c -c 
    
H2ERI_shell_operations.o: include/*.h H2ERI_shell_operations.c
	$(CC) $(CFLAGS) H2ERI_shell_operations.c -c 

H2ERI_partition.o: include/*.h H2ERI_partition.c
	$(CC) $(CFLAGS) H2ERI_partition.c -c 

H2ERI_build_H2.o: include/*.h H2ERI_build_H2.c
	$(CC) $(CFLAGS) H2ERI_build_H2.c -c 

H2ERI_build_Coulomb.o: include/*.h H2ERI_build_Coulomb.c
	$(CC) $(CFLAGS) H2ERI_build_Coulomb.c -c 

clean:
	rm $(OBJS) $(LIB)
