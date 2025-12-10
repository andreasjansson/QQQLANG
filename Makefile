.PHONY: build run clean

build:
	npm install
	npm run build

run: build
	npm run serve

clean:
	rm -rf node_modules dist
