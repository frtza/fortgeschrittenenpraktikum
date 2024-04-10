# constructing array of all subdirectories
SUBDIRS := $(wildcard */.)

# setting up subdirectories as dependency for 'all' and 'clean' targets
all: $(SUBDIRS)
clean: $(SUBDIRS)

# iterating through subdirectories, replace using parallelization
$(SUBDIRS):
	for dir in $(SUBDIRS); do \
		$(MAKE) -k -C $$dir $(MAKECMDGOALS); \
	done

# defining rules that are not output files to be built every time
.PHONY: $(SUBDIRS)
