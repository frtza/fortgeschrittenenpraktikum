all:
	make -k -C template
	make -k -C V46
	make -k -C V18

template:
	make -C template

V46:
	make -C V46
V18:
	make -C V18

clean:
	make -k -C template clean
	make -k -C V46 clean
	make -k -C

.PHONY: clean template V46
		clean template V18
