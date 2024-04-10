all:
	make -k -C template
	make -k -C V46

template:
	make -C template

V46:
	make -C V46

clean:
	make -k -C template clean
	make -k -C V46 clean

.PHONY: clean template V46
