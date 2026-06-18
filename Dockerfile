FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY AI_USAGE.md .
COPY run_eval.py .
COPY mobile_ui_env/ mobile_ui_env/
COPY tests/ tests/

# Install the package in development mode
RUN pip install --no-cache-dir -e ".[dev]"

# Default: run evaluation
CMD ["python", "run_eval.py", "--verbose"]
