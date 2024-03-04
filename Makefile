all: report clean

report:
	pdflatex -interaction batchmode rapport.tex
    pdflatex -interaction batchmode rapport.tex

verbose:
	pdflatex rapport.tex
	pdflatex rapport.tex

test:
	@printf "==== TESTS ALGORITHMS ====\n"
	@python3 src/algorithms_test.py
	@printf "\n==== TESTS PART 1 ====\n"
	@python3 src/test1.py
	@printf "\n==== TESTS PART 2 ====\n"
	@python3 src/test2.py


clean:
	rm -rf *.log *.aux *.toc