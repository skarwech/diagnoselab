TASK := harbor_task
SIM  := $(TASK)/environment/sim
HARBOR := uv run harbor

.PHONY: sync test lint skeleton view validate-sim hashes-linux isolation validate-server validate-scoring oracle-cache validate-oracle

sync:  ## copy the package into the sim build context (Harbor builds only inside environment/)
	rsync -a --delete --exclude __pycache__ --exclude eval diagnoselab/ $(SIM)/diagnoselab/
	mkdir -p $(SIM)/docs && cp docs/manual.md $(SIM)/docs/manual.md
	cp diagnoselab/oracle/oracle.py $(TASK)/solution/oracle.py

test:
	uv run pytest -q

lint: sync
	$(HARBOR) lint $(TASK)

skeleton: sync  ## G0: oracle + no-op through real Harbor trials
	$(HARBOR) run -p $(TASK) -a oracle -k 1 -o jobs
	$(HARBOR) run -p $(TASK) -a nop -k 1 -o jobs

view:
	$(HARBOR) view jobs

hashes-linux:  ## F1 cross-OS check: spec hashes from a Linux container (needs Docker)
	docker run --rm -v "$(CURDIR)":/w:ro -w /w python:3.12-slim sh -c \
	  "pip install -q --root-user-action=ignore numpy==2.5.4 >/dev/null && PYTHONPATH=/w python -m diagnoselab.eval.hashes 1000" \
	  > results/hashes_linux.json

validate-sim:  ## F1-F5 valid-when metrics -> results/step2_validation.json -> docs/proof/step2_report.html
	uv run python -m diagnoselab.eval.validate_sim
	uv run python -m diagnoselab.eval.report_sim

isolation:  ## F14: probes inside the agent container via real Harbor trials (needs Docker)
	uv run python -m diagnoselab.eval.run_isolation

validate-server:  ## F6-F14 valid-when metrics -> results/step3_validation.json -> docs/proof/step3_report.html
	uv run python -m diagnoselab.eval.validate_server
	uv run python -m diagnoselab.eval.report_server

validate-scoring:  ## F15-F18 valid-when metrics -> results/step4_validation.json -> docs/proof/step4_report.html
	uv run python -m diagnoselab.eval.validate_scoring
	uv run python -m diagnoselab.eval.report_scoring

oracle-cache:  ## F17: oracle scores per (tier, seed) from the Linux image -> diagnoselab/scoring/oracle_scores.json
	docker run --rm -v "$(CURDIR)":/w:ro -w /w python:3.12-slim sh -c \
	  "pip install -q --root-user-action=ignore numpy==2.5.4 >/dev/null && PYTHONPATH=/w python -m diagnoselab.eval.oracle_cache --seeds 0-199 --workers 8" \
	  > results/oracle_cache_linux.json

validate-oracle:  ## F19 (+F5, F17, F18 normalized) -> results/step5_validation.json -> docs/proof/step5_report.html
	uv run python -m diagnoselab.eval.validate_oracle
	uv run python -m diagnoselab.eval.report_oracle
