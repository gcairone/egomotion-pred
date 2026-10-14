# Everything runs in a resource-capped container: CPUS cores, MEM RAM, no swap.
# Host networking: DNS does not resolve on the default bridge network here.
IMAGE ?= egopred-dev
CPUS ?= 4
MEM ?= 4g
# Pipeline data (shards, features, manifest) live on the large data disk, outside the repo.
# Mounted at the same absolute path in the container, so host and container paths agree.
DATA ?= /media/gcairone/Data/egopred/data

DOCKER_RUN = docker run --rm --init --network host --cpus=$(CPUS) --memory=$(MEM) --memory-swap=$(MEM) \
	--user $(shell id -u):$(shell id -g) -e HOME=/tmp \
	-e EGOPRED_DATA=$(DATA) -v $(DATA):$(DATA) \
	-v $(CURDIR):/work -w /work

.PHONY: image test shell run ingest data-dir

image:
	docker build --network host -t $(IMAGE) .

test:
	$(DOCKER_RUN) $(IMAGE) pytest $(ARGS)

shell:
	$(DOCKER_RUN) -it $(IMAGE) bash

# make run ARGS="status"
run:
	$(DOCKER_RUN) $(IMAGE) python -m egopred $(ARGS)

# Front-camera ingest: JOBS archives in parallel inside ONE container, so the CPUS/MEM cap
# holds for the whole run (~300 MB RAM per job). make ingest JOBS=4
JOBS ?= 4
ingest:
	$(DOCKER_RUN) $(IMAGE) bash scripts/ingest_front.sh $(JOBS)

data-dir:
	@echo $(DATA)
