.PHONY: all install run plot clean clean-results reset

all: run

install:
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt

run:
	python3 main.py

plot:
	python3 -m experiments.plots.plot_self_play

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

clean-results:
	rm -rf results/raw/*
	rm -rf results/figures/*

reset: clean clean-results
