#baut alles
all: AlleAufgaben PDF
#baut nur main.pdf ohne plot
t: PDF
#baut alle Aufgaben.py
p: AlleAufgaben
#baut jede einzelne Aufgabe.py
p1: aufg1.py
p2: aufg2.py
p3: aufg3.py
p4: aufg4.py

# hier Python-Skripte:
AlleAufgaben: matplotlibrc header-matplotlib.tex
	TEXINPUTS= cd Python; python aufg1.py aufg2.py aufg3.py aufg4.py; cd ..

aufg1.py: Python/aufg1.py matplotlibrc header-matplotlib.tex
	TEXINPUTS= cd Python;	python aufg1.py; cd ..

aufg2.py: Python/aufg2.py matplotlibrc header-matplotlib.tex
	TEXINPUTS= cd Python; python aufg2.py; cd ..

aufg3.py: Python/aufg3.py matplotlibrc header-matplotlib.tex
	TEXINPUTS= cd Python; python aufg3.py; cd ..

aufg4.py: Python/aufg4.py matplotlibrc header-matplotlib.tex
	TEXINPUTS= cd Python; python aufg4.py; cd ..

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:
PDF: FORCE | build
	  TEXINPUTS="$(call translate,build:)" \
	  BIBINPUTS=build: \
	  max_print_line=1048576 \
	latexmk \
	  --lualatex \
	  --output-directory=build \
	  --interaction=nonstopmode \
	  --halt-on-error \
	main.tex

build:
	mkdir -p build

clean:
	rm -rf build

FORCE:

.PHONY: all clean
