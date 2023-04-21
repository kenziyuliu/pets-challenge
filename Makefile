.PHONY: build pull pack-benchmark pack-submission test-submission test-container

# ================================================================================================
# Settings
# ================================================================================================

ifeq (, $(shell which nvidia-smi))
CPU_OR_GPU ?= cpu
else
CPU_OR_GPU ?= gpu
endif

ifeq (${CPU_OR_GPU}, gpu)
GPU_ARGS = --gpus all
endif

SKIP_GPU ?= false
ifeq (${SKIP_GPU}, true)
GPU_ARGS =
endif

# TAGs corresponding to the local image and the official image
TAG = ${CPU_OR_GPU}-latest
LOCAL_TAG = ${CPU_OR_GPU}-local

REPO = petsprize-competition
REGISTRY_IMAGE = petsprize.azurecr.io/${REPO}:${TAG}
LOCAL_IMAGE = ${REPO}:${LOCAL_TAG}
CONTAINER_NAME = petsprize

# if not TTY (for example GithubActions CI) no interactive tty commands for docker
ifneq (true, ${GITHUB_ACTIONS_NO_TTY})
TTY_ARGS = -it
endif

# option to block or allow internet access from the submission Docker container
ifeq (true, ${BLOCK_INTERNET})
NETWORK_ARGS = --network none
endif

# To run a submission, use local version if that exists; otherwise, use official version
# setting SUBMISSION_IMAGE as an environment variable will override the image
# By default, `make build` (below) tries to build the image from the Dockerfile
SUBMISSION_IMAGE ?= $(shell docker images -q ${LOCAL_IMAGE})
ifeq (,${SUBMISSION_IMAGE})
SUBMISSION_IMAGE := $(shell docker images -q ${REGISTRY_IMAGE})
endif

# Give write access to the submission folder to everyone so Docker user can write when mounted
_submission_write_perms:
	chmod -R 0777 submission/

# ================================================================================================
# Commands for building the container if you are changing the requirements
# ================================================================================================

## Builds the container locally; this builds the image in the Dockerfile
build:
	docker build --build-arg CPU_OR_GPU=${CPU_OR_GPU} -t ${LOCAL_IMAGE} runtime

## Ensures that your locally built container can import all the Python packages successfully when it runs
test-container: build _submission_write_perms
	docker run \
		${TTY_ARGS} \
		--mount type=bind,source="$(shell pwd)"/runtime/tests,target=/tests,readonly \
		${LOCAL_IMAGE} \
		/bin/bash -c "conda run --no-capture-output -n condaenv pytest tests/test_packages.py"

## Start your locally built container and open a bash shell within the running container; same as submission setup except has network access
interact-container: build _submission_write_perms
ifeq (${SUBMISSION_TRACK},)
	$(error Specify the SUBMISSION_TRACK=fincrime or pandemic)
endif
	docker run \
		--mount type=bind,source="$(shell pwd)"/data/${SUBMISSION_TRACK},target=/code_execution/data,readonly \
		--mount type=bind,source="$(shell pwd)"/submission,target=/code_execution/submission \
		--shm-size 8g \
		-it \
		${LOCAL_IMAGE} \
		/bin/bash

## Pulls the official container from Azure Container Registry
## This pulls the image from the challenge organizers
## Note that this specifically uses the OFFICIAL image, not the Dockerfile locally
pull:
	echo "Not yet available" && exit 1
	docker pull ${REGISTRY_IMAGE}

## Creates a submission/submission.zip file from the source code in examples_src
pack-example:
# Don't overwrite so no work is lost accidentally
ifeq (${SUBMISSION_TRACK},)
	$(error Specify the SUBMISSION_TRACK=fincrime or pandemic)
endif
ifneq (,$(wildcard ./submission/submission.zip))
	$(error You already have a submission/submission.zip file. Rename or remove that file (e.g., rm submission/submission.zip).)
endif
	cd examples_src/${SUBMISSION_TRACK}; zip -r ../../submission/submission.zip ./*

## Creates a submission/submission.zip file from the source code in submission_src
pack-submission:
ifeq (${SUBMISSION_TRACK},)
	$(error Specify the SUBMISSION_TRACK=fincrime or pandemic)
endif
# Don't overwrite so no work is lost accidentally
ifneq (,$(wildcard ./submission/submission.zip))
	$(error You already have a submission/submission.zip file. Rename or remove that file (e.g., rm submission/submission.zip).)
endif
	cd submission_src/${SUBMISSION_TRACK}; zip -r ../../submission/submission.zip ./*

# For locally testing the submission source files, use the local image if available
# or fall back to the official image (from make pull)
## Runs container using code from `submission/submission.zip` and data from `data/`
test-submission: _submission_write_perms
# if submission file does not exist
ifeq (,$(wildcard ./submission/submission.zip))
	$(error To test your submission, you must first put a "submission.zip" file in the "submission" folder. \
	  If you want to use the benchmark, you can run `make pack-benchmark` first)
endif
# if container does not exist, error and tell user to pull or build
ifeq (${SUBMISSION_IMAGE},)
	$(error To test your submission, you must first run `make pull` (to get official container) or `make build` \
		(to build a local version if you have changes).)
endif
ifeq (${SUBMISSION_TYPE},)
	$(error Specify the SUBMISSION_TYPE=centralized or federated)
endif
ifeq (${SUBMISSION_TRACK},)
	$(error Specify the SUBMISSION_TRACK=fincrime or pandemic)
endif
# The actual run command is `docker run` which executes the image, which has
# a code entrypoint `/code_execution/entrypoint.sh`
# note that this also limits the RAM to 8GB by default
	docker run \
		${TTY_ARGS} \
		${GPU_ARGS} \
		${NETWORK_ARGS} \
		--env SUBMISSION_TRACK=${SUBMISSION_TRACK} \
		--network none \
		--mount type=bind,source="$(shell pwd)"/data/${SUBMISSION_TRACK},target=/code_execution/data,readonly \
		--mount type=bind,source="$(shell pwd)"/submission,target=/code_execution/submission \
		--shm-size 8g \
		--memory 32g \
		--memory-reservation 24g \
		--name ${CONTAINER_NAME} \
		--rm \
		${SUBMISSION_IMAGE} \
		${SUBMISSION_TYPE}

## Delete temporary Python cache and bytecode files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo
	@echo "$$(tput bold)Settings based on your machine:$$(tput sgr0)"
	@echo SUBMISSION_IMAGE=${SUBMISSION_IMAGE}  "\t# ID of the image that will be used when running test-submission"
	@echo
	@echo "$$(tput bold)Available competition images:$$(tput sgr0)"
	@echo "$(shell docker images --format '{{.Repository}}:{{.Tag}} ({{.ID}}); ' ${REPO})"
	@echo
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
