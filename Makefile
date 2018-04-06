# this Makefile is just for the Sphinx based documentation

.PHONY: docs default

default: docs
	@echo done

docs:
	@echo docs
	./bin/opticks_docs_make.sh



