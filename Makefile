.PHONY: all clean
task = task01 task02 task03 task04
src = src/task01.cpp src/task02.cpp src/task03.cpp src/task04.cpp src/sub.cpp src/get_data.cpp
obj = obj/task01.o obj/task02.o obj/task03.o obj/task04.o obj/sub.o obj/get_data.o

all: $(task)

$(task): task0% : obj/task0%.o obj/sub.o obj/get_data.o
	g++ -o $@ $^

$(obj) : obj/%.o : src/%.cpp src/net.hpp src/param.hpp src/get_data.hpp src/sub.hpp
	mkdir -p obj
	g++ -o $@ -c $< -I /usr/local/include/eigen3

clean:
	rm -fr obj $(task)
