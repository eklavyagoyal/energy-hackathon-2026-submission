.PHONY: install test test-fast serve pypsa-sample freeze-topology freeze-realtime freeze-battery clean

VENV := .venv
PY := $(VENV)/bin/python

install:
	~/.local/bin/python3.11 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -c "import pandapower, fastapi, anthropic; print('root deps OK')"

test:
	$(PY) -m pytest tests/ -q

test-fast:
	$(PY) -m pytest tests/test_pypsa_eur_loader.py tests/timeseries/test_geographic_scenarios.py tests/timeseries/test_api.py -q

serve:
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

pypsa-sample:
	$(PY) scripts/download_pypsa_eur.py

freeze-topology:
	$(PY) scripts/freeze_topology.py

freeze-realtime:
	$(PY) scripts/freeze_realtime_demo.py

freeze-battery:
	$(PY) scripts/freeze_battery_demo.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
