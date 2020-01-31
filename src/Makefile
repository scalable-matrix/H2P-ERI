LIB   = libH2PERI.a
OBJS  = CMS.o H2ERI_typedef.o H2ERI_shell_operations.o H2ERI_partition.o \
        H2ERI_build_H2.o H2ERI_build_Coulomb.o

SIMINT_DIR = /home/mkurisu/Workspace/simint/build-avx/install
H2PACK_DIR = /home/mkurisu/Workspace/H2Pack

HOSTCC  = icc
AR      = xiar rcs
DEF     = -DUSE_MKL -mkl
INC     = -I./include -I$(SIMINT_DIR)/include -I$(H2PACK_DIR)/src/include
CFLAGS  = $(INC) -Wall -g -std=gnu99 -O3 -qopenmp -xHost $(DEF)
LDFLAGS = -O3 -xHost -qopenmp 

$(LIB): $(OBJS) 
	$(AR) $@ $^

CMS.o: include/*.h CMS.c
	$(HOSTCC) $(CFLAGS) CMS.c -c 

H2ERI_typedef.o: include/*.h H2ERI_typedef.c
	$(HOSTCC) $(CFLAGS) H2ERI_typedef.c -c 
    
H2ERI_shell_operations.o: include/*.h H2ERI_shell_operations.c
	$(HOSTCC) $(CFLAGS) H2ERI_shell_operations.c -c 

H2ERI_partition.o: include/*.h H2ERI_partition.c
	$(HOSTCC) $(CFLAGS) H2ERI_partition.c -c 

H2ERI_build_H2.o: include/*.h H2ERI_build_H2.c
	$(HOSTCC) $(CFLAGS) H2ERI_build_H2.c -c 

H2ERI_build_Coulomb.o: include/*.h H2ERI_build_Coulomb.c
	$(HOSTCC) $(CFLAGS) H2ERI_build_Coulomb.c -c 

clean:
	rm $(OBJS) $(LIB)
