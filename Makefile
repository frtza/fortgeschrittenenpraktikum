all:
	make -k -C template
	make -k -C V18
	make -k -C V21
	make -k -C V46
	make -k -C V60

template:
	make -C template

V18:
	make -C V18

V21:
	make -C V21

V46:
	make -C V46

V60:
	make -C V60

clean:
	make -k -C template clean
	make -k -C V18 clean
	make -k -C V21 clean
	make -k -C V46 clean
	make -k -C V60 clean

.PHONY: clean template V18 V21 V46 V60
