ROOT := $(abspath ..)
PYTHON ?= python3
LATEXMK ?= latexmk
MAIN ?= main.tex

TABLE_SCRIPT := $(ROOT)/thesis/scripts/export_tables.py
REPORT_SCRIPT := $(ROOT)/scripts/generate_thesis_report.py
DEDUP_SCRIPT := $(ROOT)/scripts/deduplicate_results_ledger.py
RESULTS_PATH := $(ROOT)/results/results_tuned_all.jsonl
REPORT_OUT := $(ROOT)/results/reports/thesis_tuned_all

.PHONY: all pdf watch clean tables report full

all: pdf

pdf: tables
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN)

watch: tables
	$(LATEXMK) -pdf -pvc -interaction=nonstopmode -halt-on-error $(MAIN)

clean:
	$(LATEXMK) -C
	rm -f *.bbl *.blg *.run.xml *.bcf

tables:
	THESIS_REPORT_DIR=$(REPORT_OUT) $(PYTHON) $(TABLE_SCRIPT)

report:
	$(PYTHON) $(REPORT_SCRIPT) --results $(RESULTS_PATH) --out $(REPORT_OUT) --expected-runs 30

full:
	$(PYTHON) $(DEDUP_SCRIPT) --results $(RESULTS_PATH)
	$(PYTHON) $(REPORT_SCRIPT) --results $(RESULTS_PATH) --out $(REPORT_OUT) --expected-runs 30
	THESIS_REPORT_DIR=$(REPORT_OUT) $(PYTHON) $(TABLE_SCRIPT)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(MAIN)
