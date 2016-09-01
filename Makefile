TARGET = Sort6
PROF_TARGET = Prof6

SOURCE = sort6.c

CFLAGS = -mavx2 -mavx -march=native -Wall -std=gnu99 -D_GNU_SOURCE
DFLAGS = -ggdb
PFLAGS = -DPROFILE -O3 -DNDEBUG
#PFLAGS += -pg
PFLAGS += -ggdb

$(TARGET): $(SOURCE)
	gcc $(CFLAGS) $(DFLAGS) $< -o ./$@

prof: $(SOURCE)
	gcc $(CFLAGS) $(PFLAGS) $< -o $(PROF_TARGET)

obj: $(SOURCE)
	gcc -c $(CFLAGS) $(PFLAGS) $<

clean:
	rm -f $(TARGET)

clean_prof:
	rm -f $(PROF_TARGET)
