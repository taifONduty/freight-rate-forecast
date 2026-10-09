PYTHON ?= .venv/bin/python

.PHONY: setup test backtest ablations train predict score figures report all

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

backtest:
	$(PYTHON) -m ratecast backtest

ablations:
	$(PYTHON) -m ratecast ablations

train:
	$(PYTHON) -m ratecast train

predict:
	$(PYTHON) -m ratecast predict

score:
	$(PYTHON) score.py --predictions outputs/validation_predictions.csv \
		--december-predictions outputs/december_chart_inputs.csv --output-dir outputs/scorer_results
	cp outputs/scorer_results/candidate_december.png report/figures/

figures:
	$(PYTHON) -m ratecast figures

report:
	cd report && tectonic report.tex

all: test backtest train predict score figures
