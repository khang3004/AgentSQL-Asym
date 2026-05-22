.PHONY: all clean tikz main

all: tikz main

tikz:
	cd tikz_artifacts && pdflatex -interaction=nonstopmode agentsql_workflow.tex

main:
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.bcf *.run.xml *.synctex.gz *.toc
	rm -f tikz_artifacts/*.aux tikz_artifacts/*.log
