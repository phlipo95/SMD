ifeq (,$(shell sh -c 'cygpath --version 2> /dev/null'))
  # Unix
  pwd := $$(pwd)
  translate = $1
else
  # Windows mit MSys2/Cygwin
  pwd := $$(cygpath -m "$$(pwd)")
  translate = $(shell echo '$1' | sed 's/:/;/g')
endif

all: build/main.pdf
p1: aufg1.py
p2: aufg2.py
p3: aufg3.py
p4: aufg4.py
p: AlleAufgaben

# hier Python-Skripte:
AlleAufgaben:
	TEXINPUTS="$(call translate,$(pwd):)" make p1 p2 p3 p4

aufg1.py: Python/aufg1.py matplotlibrc header-matplotlib.tex
	TEXINPUTS="$(call translate,$(pwd):)" python Python/aufg1.py

aufg2.py: Python/aufg2.py matplotlibrc header-matplotlib.tex
	TEXINPUTS="$(call translate,$(pwd):)" python Python/aufg2.py

aufg3.py: Python/aufg3.py matplotlibrc header-matplotlib.tex
	TEXINPUTS="$(call translate,$(pwd):)" python Python/aufg3.py

aufg4.py: Python/aufg4.py matplotlibrc header-matplotlib.tex
	TEXINPUTS="$(call translate,$(pwd):)" python Python/aufg4.py

# hier weitere Abhängigkeiten für build/main.pdf deklarieren:

build/main.pdf: FORCE | build
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
