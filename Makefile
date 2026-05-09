.PHONY: dev test

dev:
	./scripts/run_all.sh

test:
	python -m pytest tests -q
	cd dvl_correction && python -m unittest discover -s tests
	cd imu-drift && cargo test --locked
	cd imu-drift && cargo test --locked --features sample
	cd frontend/web && npm run typecheck && npm run build
	cd frontend/server && cargo test --locked
