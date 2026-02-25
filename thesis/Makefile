ROOT := $(abspath ..)
PYTHON ?= python3
LATEXMK ?= latexmk
MAIN ?= main.tex

TABLE_SCRIPT := $(ROOT)/thesis/scripts/export_tables.py
REPORT_SCRIPT := $(ROOT)/scripts/generate_thesis_report.py
RESULTS_PATH := $(ROOT)/results/results.jsonl
REPORT_OUT := $(ROOT)/results/reports/thesis

.PHONY: all pdf watch clean tables report

all: pdf

pdf: tables
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN)

watch: tables
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode -halt-on-error $(MAIN)

clean:
	$(LATEXMK) -C
	rm -f *.bbl *.blg *.run.xml *.bcf

tables:
	$(PYTHON) $(TABLE_SCRIPT)

report:
	$(PYTHON) $(REPORT_SCRIPT) --results $(RESULTS_PATH) --out $(REPORT_OUT)
