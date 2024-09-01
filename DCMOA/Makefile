WARTHOG_SRC =  $(wildcard domains/*.cpp) \
				$(wildcard util/*.cpp) \
			  $(wildcard search/*.cpp) \
			  			  
			  

WARTHOG_OBJ = $(WARTHOG_SRC:.cpp=.o)

D_WARTHOG_INCLUDES = -I./ -I./sys -I./heuristics -I./util -I./search -I./domains\
					#  -I./heuristics  -I./domains \
					#  -I./memory \
					 -I./sys

D_INCLUDES = $(D_WARTHOG_INCLUDES) -I/usr/include -I/usr/local/include 
D_LIBS = -L./lib -L/usr/local/lib 

CFLAGS = -std=c++11 -pedantic -pthread -fopenmp -Wall -Werror -Wno-conversion -Wno-unused-result -Wno-unused-but-set-variable
FAST_CFLAGS = -O3 -DNDEBUG 

ifeq ("$(findstring Darwin, "$(shell uname -s)")", "Darwin")
  CC = g++
  CFLAGS += -DOS_MAC
else
  CC = g++
  ifeq ("$(findstring Linux, "$(shell uname -s)")", "Linux")
    D_LIBS += -lrt
  endif
endif

.PHONY: main
main: CFLAGS += $(FAST_CFLAGS) $(D_INCLUDES)
main: makedirs warthog 
	@echo "linking..."
	$(CC) programs/roadhog.cpp -o ./bin/roadhog -lwarthog $(CFLAGS) $(D_LIBS) $(D_INCLUDES) 

.PHONY: path
path: CFLAGS += -DPATH
path: main

.PHONY: makedirs
makedirs:
	@echo "### Creating output directories ###"
	@$(shell mkdir ./bin)
	@$(shell mkdir ./lib)

.PHONY: warthog
warthog: $(WARTHOG_OBJ)
	@echo "###  Archiving object files ###"
	@echo "sources "$(WARTHOG_SRC)
	ar -crs lib/lib$(@).a $(WARTHOG_OBJ)

-include $(WARTHOG_OBJ:.o=.d)

$(WARTHOG_OBJ): 
	@echo "compiling..."
	$(CC) -c $(@:.o=.cpp) $(CFLAGS) -MM -MP -MT $(dir $(@))$(notdir $(@)) -MF $(@:.o=.d) 
	$(CC) -c $(@:.o=.cpp) -o $(@) $(CFLAGS) 

.PHONY: clean
clean:
	@rm -rf ./bin/*
	@-$(RM) ./lib/*
	@-$(RM) $(WARTHOG_OBJ:.o=.d)
	@-$(RM) $(WARTHOG_OBJ)