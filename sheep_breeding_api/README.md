# Sheep Breeding API - Maintainer Documentation

This directory contains a basic Flask API for interacting with the Sheep Breeding Genomics Toolkit. It features asynchronous task processing via Celery, API key authentication, Pydantic-based request validation, and configuration via environment variables.

## 1. Architecture Overview

The system consists of the following main components:

*   **Flask API (`app.py`):**
    *   Provides HTTP endpoints for client interaction (data uploads, task creation, status checks, result retrieval).
    *   Handles request validation (API key, Pydantic models for request bodies).
    *   Dispatches analysis tasks to Celery.
    *   Stores initial task parameters (like file identifiers and analysis settings) in Redis.
    *   Queries Celery's backend (Redis) for task status and results.
*   **Celery Workers (`celery_app.py`):**
    *   Execute long-running analysis tasks asynchronously (PBLUP, GBLUP, ssGBLUP).
    *   Tasks are defined in `celery_app.py`.
    *   Use functions from the `sheep_breeding_genomics` core toolkit to perform calculations.
    *   Report status and results to the Celery backend (Redis).
*   **Redis:**
    *   Acts as the message broker for Celery (passing tasks from Flask app to workers).
    *   Serves as the result backend for Celery (storing state and results of tasks).
    *   Used by the Flask app to store initial parameters for dispatched tasks for auditing or display purposes.
*   **`sheep_breeding_genomics` Toolkit:**
    *   The core Python library (located in the parent directory) containing modules for:
        *   `data_management`: Data structures, I/O, validation.
        *   `genetic_evaluation`: Relationship matrices, BLUP model solvers.
        *   `breeding_strategies`: Selection, mating schemes (currently not directly used by API tasks but part of the overall toolkit).

Uploaded data files are temporarily stored in the `sheep_breeding_api/uploads/` directory.

## 2. Setup and Installation

1.  **Prerequisites:**
    *   Python (version 3.8+ recommended).
    *   Redis server installed and running.
2.  **Clone the Repository:** (Assuming this is part of a larger project structure)
3.  **Install Dependencies:**
    Navigate to the project root (`sheep_breeding_genomics`) and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    This installs Flask, Celery, Redis client, Pydantic, Pandas, NumPy, SciPy.
4.  **Configuration:** (See Section 3: Configuration)

## 3. Configuration

Key aspects of the API and Celery application are configured via **environment variables**.

*   **`.env.example`:** A template file `sheep_breeding_api/.env.example` is provided. Copy this to `.env` in the same directory and customize it:
    ```bash
    cd sheep_breeding_api
    cp .env.example .env
    # Now edit .env with your specific settings
    ```
*   **Key Environment Variables:**
    *   `FLASK_ENV`: "development" or "production".
    *   `FLASK_SECRET_KEY`: **Critical for Flask.** Strong, unique secret.
    *   `API_KEY`: **Critical for API access.** The API key clients must use.
    *   `REDIS_HOST`: Redis server hostname (default: "localhost").
    *   `REDIS_PORT`: Redis server port (default: "6379").
    *   `REDIS_DB_API`: Redis DB for API task parameters (default: "0").
    *   `REDIS_DB_CELERY`: Redis DB for Celery broker/backend (default: "0").
*   **Loading `.env` (Development):**
    *   For development ease, add `python-dotenv` to `requirements.txt` (if not already there).
    *   Add `from dotenv import load_dotenv; load_dotenv()` at the top of `app.py` and `celery_app.py`.
*   **Production:** In production, set environment variables directly in your deployment environment.
*   **Security:** Default keys in `app.py` for `API_KEY` and `FLASK_SECRET_KEY` are for development only. Ensure strong, unique values are set via environment variables for production. The app logs warnings if defaults are used when `FLASK_ENV=production`.

## 4. Running the System (Development)

1.  **Ensure Redis is Running:** Based on your `REDIS_HOST`/`REDIS_PORT`.
2.  **Start Celery Worker(s):**
    Open a terminal, navigate to the project root (`sheep_breeding_genomics`), and run:
    ```bash
    # Ensure your .env file is in sheep_breeding_api/ and loaded if using python-dotenv,
    # or that environment variables are otherwise set in this terminal session.
    celery -A sheep_breeding_api.celery_app worker -l INFO
    ```
    *(The `-A` flag points to the Celery application instance).*
3.  **Start the Flask API Server:**
    Open another terminal, navigate to the project root, and run:
    ```bash
    # Ensure your .env file is in sheep_breeding_api/ and loaded, or env vars set.
    python -m sheep_breeding_api.app
    ```
    The API should be available at `http://localhost:5000` (or as configured).

## 5. API Endpoint Documentation

All API endpoints (except `/api/health`) require authentication via an API key provided in the `X-API-KEY` header.

*(Existing endpoint documentation for Health Check, Uploads, Create PBLUP/GBLUP/ssGBLUP Task, Get Task Status, Get Task Result should be retained here as previously generated. Ensure they are accurate with request/response models.)*

**Key Data Flow:**
1.  Upload data files (pheno, ped, geno) to respective `/api/upload/...` endpoints. Receive `file_id` (filename) for each.
2.  Submit an analysis task via `/api/run/[pblup|gblup|ssgblup]` using the `file_id`s and other parameters in a JSON body. Receive a `task_id`.
3.  Poll `/api/task/<task_id>/status` for progress.
4.  If 'SUCCESS', get results from `/api/task/<task_id>/result`.

## 6. Celery Tasks (`celery_app.py`)

*   **`run_pblup_analysis_task(...)`**:
    *   Parameters: `phenotypes_file_path`, `pedigree_file_path`, `trait_col`, `var_animal`, `var_residual`, `task_id` (optional ref).
    *   Action: Loads data, calculates NRM, solves MME for PBLUP.
*   **`run_gblup_analysis_task(...)`**:
    *   Parameters: `phenotypes_file_path`, `genomic_data_file_path`, `trait_col`, `var_animal`, `var_residual`, `task_id` (optional ref).
    *   Action: Loads data, calculates GRM, filters phenotypes to genotyped animals, solves MME for GBLUP.
*   **`run_ssgblup_analysis_task(...)`**:
    *   Parameters: `phenotypes_file_path`, `pedigree_file_path`, `genomic_data_file_path`, `trait_col`, `var_animal`, `var_residual`, `grm_method`, `h_inv_tuning_factor`, `task_id` (optional ref).
    *   Action: Loads all data, calculates NRM, GRM, H_inv, solves MME for ssGBLUP.

All tasks report results to the Celery backend (Redis) and update their state. Exceptions are re-raised to be caught by Celery and mark tasks as 'FAILURE'.

## 7. Code Structure

*   **`sheep_breeding_api/`**:
    *   `app.py`: Flask application, API endpoint definitions, request validation (Pydantic), Celery task dispatching, Redis client for initial task params.
    *   `celery_app.py`: Celery application setup, definition of asynchronous analysis tasks. Tasks use the core `sheep_breeding_genomics` toolkit.
    *   `uploads/`: Directory where uploaded data files are stored (temporary for this version).
    *   `.env.example`: Template for environment variable configuration.
    *   `README.md`: This file.
*   **`sheep_breeding_genomics/` (Core Toolkit - Parent Directory):**
    *   `data_management/`: Data structures, I/O, validation.
    *   `genetic_evaluation/`: Matrix calculations, BLUP model solvers.
    *   `breeding_strategies/`: Selection, mating schemes.
*   **`examples/`**: Example scripts using the core toolkit.
*   **`tests/`**: Unit tests for the toolkit and API.

## 8. Logging and Error Handling Strategy

*   **Flask API (`app.py`):**
    *   Uses `app.logger` for logging events (e.g., task dispatch, Redis errors). Debug mode provides more verbose Flask logs.
    *   Pydantic handles request body validation errors automatically, returning 400 responses with details.
    *   API key decorator handles authentication errors (401).
    *   File existence and basic header checks are done in upload/run endpoints, returning 400/404.
    *   Generic `try-except` blocks in task dispatching/status/result endpoints catch other errors and return appropriate JSON error responses (often 500 for unexpected issues).
*   **Celery Tasks (`celery_app.py`):**
    *   Use `print()` for basic logging (visible in Celery worker console). For production, Python's `logging` module configured for Celery would be better.
    *   Tasks re-raise exceptions. Celery catches these, marks the task as 'FAILURE', and stores traceback information in the result backend, accessible via `AsyncResult.info` and `AsyncResult.traceback`.
    *   Tasks use `self.update_state()` for custom progress reporting if needed (currently used for basic steps).

## 9. Deployment Considerations (Summary)

*   **WSGI Server:** Use a production-grade WSGI server (e.g., Gunicorn, uWSGI) for the Flask app instead of the development server.
*   **Celery Workers:** Run Celery workers as background services (e.g., using `systemd` or Supervisor). Scale the number of workers based on load.
*   **Redis:** Ensure Redis is configured for persistence if task results/parameters stored there need to survive Redis restarts. Consider Redis security (authentication, network access).
*   **Environment Variables:** All sensitive configurations (API keys, secret keys, database URLs) MUST be set via environment variables or a secure secrets management system.
*   **File Storage:** The current `uploads` folder is local to the API instance. For multi-node deployments or durable storage, use a shared filesystem (NFS) or cloud storage (S3, Google Cloud Storage) for uploaded files, and pass accessible paths or references to Celery tasks.
*   **Logging:** Configure centralized logging for both Flask and Celery for monitoring and debugging.

## 10. How to Add New Analysis Types

1.  **Core Toolkit:** Implement the new analysis logic within the `sheep_breeding_genomics` toolkit (e.g., a new function in `blup_models.py` or a new module).
2.  **Celery Task (`celery_app.py`):**
    *   Define a new Celery task that takes necessary parameters (file paths, analysis settings).
    *   This task will call the corresponding function(s) from the core toolkit.
    *   Handle results and exceptions similarly to existing tasks.
3.  **Pydantic Model (`app.py`):**
    *   Define a new Pydantic `BaseModel` for the request body of the new analysis if it requires specific parameters beyond file IDs.
4.  **API Endpoint (`app.py`):**
    *   Create a new Flask route (e.g., `/api/run/new_analysis_type`).
    *   Apply `@api_key_required` and `@validate(body=YourNewRequestModel)`.
    *   In the route function, retrieve validated parameters.
    *   Construct file paths.
    *   Dispatch the new Celery task.
    *   Store initial parameters in Redis.
    *   Return the Celery task ID.
5.  **Documentation (`README.md`):**
    *   Document the new endpoint, its request body, and expected behavior.
    *   Update the "Overview" and "Data Flow" sections if necessary.
6.  **Tests:** Add unit tests for the new core toolkit functions, the Celery task, and the API endpoint.

## 11. Testing

The project includes unit tests for the core toolkit and is planned for the API.

*   **Running Core Toolkit Tests:**
    Navigate to the project root (`sheep_breeding_genomics`) and run:
    ```bash
    python -m unittest discover -s tests -p "test_*.py"
    ```
    This will discover and run all tests within the `tests` directory and its subdirectories (e.g., `tests/test_data_management`, `tests/test_genetic_evaluation`).

*   **Running API Tests (Conceptual - if separate tests are written for API logic):**
    If specific API tests are created (e.g., in `tests/test_api`), they might be run similarly or using a Flask test client. For now, API logic is tested via Celery tasks which use the core toolkit. The primary way to test API interaction is manually (e.g. using `curl` or Postman) against a running API, Celery worker, and Redis instance.

---
This provides a more comprehensive guide for maintainers.
