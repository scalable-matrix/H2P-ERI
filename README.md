H2P-ERI is a package for compressing the 4D electron repulsion integral (ERI) with the $\mathcal{H}^2$ matrix format for computing the Coulomb matrix. H2P-ERI is developed based on the [H2Pack](https://github.com/scalable-matrix/H2Pack) library.

## Compiling H2P-ERI

Set `WORKDIR` to your working directory. For example, `export WORKDIR=$HOME/workspace`. Clone H2P-ERI:

```bash
cd $WORKDIR
git clone https://github.com/scalable-matrix/H2P-ERI.git
```

### Compiling SIMINT

H2P-ERI relies on the [SIMINT](https://github.com/simint-chem/simint-generator) library.

Download and compile the SIMINT code generator:

```bash
cd $WORKDIR
git clone https://github.com/simint-chem/simint-generator.git
cd simint-generator
mkdir build && cd build
CC=gcc CXX=g++ cmake ../
make -j4
cd ../
```

Generate the SIMINT source code:

```bash
python3 ./create.py -g build/generator/ostei -l 6 -p 4 -ve 4 -he 4 -vg 5 -hg 5 simint-2022Apr14
mv simint-2022Apr14 $WORKDIR
```

In this command:
* `-l` specifies the maximum angular momentum (6 == s, p, d, f, g, h shells) SIMINT needs to support
* `-p 4 -ve 4 -he 4 -vg 5 -hg 5` are the recommended performance-related parameters
* `simint-2022Apr14` is the name of the output directory (corresponding to commit `a497ec7`)

Compile the SIMINT library. We recommend using Intel C Compiler (ICC) versions 17, 18, 19, 20 (tested). You may also use GCC / Clang.

```bash
cd $WORKDIR/simint-2022Apr14
mkdir build && cd build
CC=icc CXX=icpc cmake -DSIMINT_VECTOR=avx2 -DCMAKE_INSTALL_PREFIX=./install
make -j8 install
```

In the commands above, `-DSIMINT_VECTOR=avx2` enables the SIMINT AVX2 support, you can replace it with `-DSIMINT_VECTOR=avx` or `-DSIMINT_VECTOR=commonavx512`, see `$WORKDIR/simint-2022Apr14/README` for more options.

### Compiling H2P-ERI

H2P-ERI requires a BLAS library and a LAPACK library. We recommend using Intel MKL or OpenBLAS. If you would like to use other BLAS + LAPACK implementations, please adjust the configuration file according to your system.

```bash
cd $WORKDIR/H2P-ERI/src
# Then edit common.make
```

Edit line 7, set variable `SIMINT_INSTALL_DIR` to `$WORKDIR/simint-2022Apr14/build/install`.

If you are using OpenBLAS, set variable `OPENBLAS_INSTALL_DIR` to the directory you install OpenBLAS. For example, on my computer, `OPENBLAS_INSTALL_DIR = /home/keqing/workspace/OpenBLAS-git/install`, and this directory should look like:
```bash
$ tree /home/keqing/workspace/OpenBLAS-git/install
/home/keqing/workspace/OpenBLAS-git/install
├── bin
├── include
│   ├── cblas.h
│   ├── f77blas.h
│   ├── lapacke_config.h
│   ├── lapacke.h
│   ├── lapacke_mangling.h
│   ├── lapacke_utils.h
│   ├── lapack.h
│   └── openblas_config.h
└── lib
    ├── cmake
    │   └── openblas
    │       ├── OpenBLASConfig.cmake
    │       └── OpenBLASConfigVersion.cmake
    ├── libopenblas.a -> libopenblas_sandybridgep-r0.3.23.dev.a
    ├── libopenblas_sandybridgep-r0.3.23.dev.a
    ├── libopenblas_sandybridgep-r0.3.23.dev.so
    ├── libopenblas.so -> libopenblas_sandybridgep-r0.3.23.dev.so
    ├── libopenblas.so.0 -> libopenblas_sandybridgep-r0.3.23.dev.so
    └── pkgconfig
        └── openblas.pc

6 directories, 16 files
```

If you are using ICC + MKL, run `make -f ICC-MKL.make` to compile H2P-ERI. If you are using GCC + OpenBLAS, run `make -f GCC-OpenBLAS.make` to compile H2P-ERI.

### Compiling and Running Test Programs

H2P-ERI has some test programs. The test programs reply on another small library [YATDFT](https://github.com/huanghua1994/YATDFT). Follow instructions in the YATDFT library to compile it before compiling H2P-ERI test programs.

Enter directory `$WORKDIR/H2P-ERI/tests` and edit the variables in the top 5 rows of `common.make`, then use `make -f ICC-MKL.make` or `make -f GCC-OpenBLAS.make` to compile.

## References

Please cite the following two papers if you use H2P-ERI in your work:

```bibtex
@article{xing2020jcp,
    title = {A Linear Scaling Hierarchical Block Low-rank Representation of the Electron Repulsion Integral Tensor},
    author = {Xing, Xin and Huang, Hua and Chow, Edmond},
    journal = {Journal of Chemical Physics},
    volume = {153},
    number = {8},
    pages = {084119},
    year = {2020},
    doi = {10.1063/5.0010732},
},

@article{xing2020sisc,
    title = {Fast {Coulomb} Matrix Construction via Compressing the Interactions Between Continuous Charge Distributions},
    author = {Xing, Xin and Chow, Edmond},
    journal = {SIAM Journal on Scientific Computing},
    volume = {42},
    number = {1},
    pages = {A162-A186},
    year = {2020},
    doi = {10.1137/19M1252855},
}
```

