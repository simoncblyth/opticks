NAME = dateparse

all: $(NAME)
	@echo done


$(NAME) : $(NAME).o
	g++ -L/opt/local/lib -lboost_date_time  $< -o $@

$(NAME).o : $(NAME).cpp
	g++ -c -I/opt/local/include $< -o $@



.PHONY: all clean test


clean : 
	rm *.o speed

test :
	./$(NAME)
