.PHONY: install install-qml test

install:
	python -m pip install -e .

install-qml:
	python -m pip install -e '.[qml]'

test:
	python -m unittest discover -s tests
