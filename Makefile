all:
	make -k -C template
	make -k -C V18
	make -k -C V46

template:
	make -C template

V18:
	make -C V18

V46:
	make -C V46

clean:
	make -k -C template clean
	make -k -C V46 clean
	make -k -C

.PHONY: clean template V18 V46
