NVCC ?= nvcc
# "native" needs CUDA 11.5+; otherwise pass e.g. ARCH=sm_75
ARCH ?= native
SRCS := $(wildcard kernels/*.cu)

.PHONY: lib bench test readme clean

lib: libkernels.so

libkernels.so: $(SRCS)
	$(NVCC) -O3 -arch=$(ARCH) -shared -Xcompiler -fPIC $(SRCS) -o $@

bench: lib
	python bench.py

test: lib
	python -m pytest -q tests

# Rebuild the bandwidth plots and the results table in README.md from results/*.json
readme:
	python tools/plot.py
	python tools/update_readme.py

clean:
	rm -f libkernels.so
