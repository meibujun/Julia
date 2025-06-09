# sheep_breeding_api/app.py

from flask import Flask, request, jsonify
import os
import tempfile
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Adjust path to import from the root of the project for the toolkit
# This assumes the API is run from within its directory or the project root is in PYTHONPATH
import sys
import uuid # For generating unique task IDs
from functools import wraps # For creating the decorator
import json # For serializing data to Redis
from typing import Optional # For Pydantic models
import os # For environment variables

# Get the directory of the current file (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Celery imports
from .celery_app import celery_app, run_pblup_analysis_task, run_gblup_analysis_task, run_ssgblup_analysis_task # Import new ssGBLUP task
from celery.result import AsyncResult

# Redis import
import redis

# Pydantic and Flask-Pydantic imports
from pydantic import BaseModel, Field # BaseModel and Field for request model
from flask_pydantic import validate # validate decorator


from sheep_breeding_genomics.data_management.io_handlers import read_phenotypic_data, read_pedigree_data
# These specific core library imports (calc_nrm, solve_animal_model_mme) are now primarily used within the Celery task.


# --- Pydantic Models ---
class RunPBLUPRequest(BaseModel):
    phenotypes_file_id: str = Field(..., description="Filename of the uploaded phenotypic data CSV (e.g., 'pheno.csv'). Example: 'my_pheno_data.csv'")
    pedigree_file_id: str = Field(..., description="Filename of the uploaded pedigree data CSV (e.g., 'ped.csv'). Example: 'my_ped_data.csv'")
    trait_name: Optional[str] = Field('TraitValue', description="Name of the trait column in the phenotypic data file.")
    heritability: Optional[float] = Field(0.3, ge=0.001, le=0.999, description="Heritability of the trait (between 0.001 and 0.999).")
    assumed_phenotypic_variance: Optional[float] = Field(10.0, gt=0, description="Assumed total phenotypic variance for the trait, used with heritability.")


class RunGBLUPRequest(BaseModel):
    phenotypes_file_id: str = Field(..., description="Filename of the uploaded phenotypic data CSV.")
    genomic_data_file_id: str = Field(..., description="Filename of the uploaded genomic data CSV.")
    trait_name: Optional[str] = Field('TraitValue', description="Name of the trait column in the phenotypic data file.")
    heritability: Optional[float] = Field(0.3, ge=0.001, le=0.999, description="Heritability of the trait (0.001-0.999).")
    assumed_phenotypic_variance: Optional[float] = Field(10.0, gt=0.0, description="Assumed total phenotypic variance for deriving genetic/residual variances.")


class RunSSGBLUPRequest(BaseModel):
    phenotypes_file_id: str = Field(..., description="Filename of the uploaded phenotypic data CSV.")
    pedigree_file_id: str = Field(..., description="Filename of the uploaded pedigree data CSV.")
    genomic_data_file_id: str = Field(..., description="Filename of the uploaded genomic data CSV.")
    trait_name: Optional[str] = Field('TraitValue', description="Name of the trait column in the phenotypic data file.")
    heritability: Optional[float] = Field(0.3, ge=0.001, le=0.999, description="Heritability of the trait (0.001-0.999).")
    assumed_phenotypic_variance: Optional[float] = Field(10.0, gt=0.0, description="Assumed total phenotypic variance for deriving genetic/residual variances.")
    grm_method: Optional[str] = Field("vanraden1", description="Method for GRM calculation (e.g., 'vanraden1').")
    h_inv_tuning_factor: Optional[float] = Field(default=0.0, ge=0.0, le=1.0, description="Tuning factor for GRM (0=pure GRM, 1=pure A22 in H_inv adjustment - conceptual).")


# --- Configuration from Environment Variables ---
# API Key Configuration
DEFAULT_DEV_API_KEY = "default_development_key"
EXPECTED_API_KEY = os.environ.get("API_KEY", DEFAULT_DEV_API_KEY)

# Flask Secret Key (important for session management, flash messages, etc.)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_very_strong_default_secret_for_dev_only")

# Redis Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379)) # Ensure type conversion
REDIS_DB_API = int(os.environ.get("REDIS_DB_API", 0)) # Potentially different DB for API specific data vs Celery
# Initialize Redis client for API specific use (e.g. storing task params)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_API, decode_responses=True)

# Production Mode Check (simple version)
IS_PRODUCTION = os.environ.get("FLASK_ENV") == "production"
if IS_PRODUCTION and EXPECTED_API_KEY == DEFAULT_DEV_API_KEY:
    app.logger.warning("SECURITY WARNING: API_KEY is set to its default development value in a production-like environment.")
if IS_PRODUCTION and app.secret_key == "a_very_strong_default_secret_for_dev_only":
    app.logger.warning("SECURITY WARNING: FLASK_SECRET_KEY is set to its default development value in a production-like environment.")


# Uploads Configuration
UPLOAD_FOLDER = os.path.join(current_dir, 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

# In-memory "database" for uploaded file paths (before task creation).
# This could also be moved to Redis (e.g., using Flask session with Redis backend).
uploaded_data_store = {
    "phenotypes_file": None,
    "pedigree_file": None,
}
# tasks_db (local dict) is removed as initial parameters will be stored in Redis,
# and Celery backend handles status/results.


# --- Authentication Decorator ---
def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')
        if not api_key or api_key != EXPECTED_API_KEY:
            app.logger.warning(f"Unauthorized access attempt. Provided API Key: {api_key}")
            return jsonify({'error': 'Unauthorized. Invalid or missing API Key.'}), 401
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to confirm the API is running.
    """
    return jsonify({'status': 'healthy', 'message': 'Sheep Breeding API is running.'}), 200


@app.route('/api/upload/phenotypes', methods=['POST'])
@api_key_required
def upload_phenotypes():
    """
    Endpoint to upload phenotypic data as a CSV file.
    Performs basic validation: checks for file presence, CSV extension, and expected headers.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Save to a temporary file first to read and validate
            temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
            temp_filepath = os.path.join(temp_dir, file.filename)
            file.save(temp_filepath)

            # Basic CSV validation: try to read and check headers
            try:
                df = pd.read_csv(temp_filepath, nrows=5) # Read only a few rows for header check
                expected_headers = ['AnimalID'] # Minimal check, add more like 'TraitValue' as needed
                # More robust check: if 'TraitValue' is also mandatory
                # expected_headers = ['AnimalID', 'TraitValue']

                missing_headers = [h for h in expected_headers if h not in df.columns]
                if missing_headers:
                    os.remove(temp_filepath) # Clean up temp file
                    os.rmdir(temp_dir)
                    return jsonify({'error': f'Missing expected headers in CSV: {", ".join(missing_headers)}'}), 400

                # If validation passes, you might move the file from temp_dir to a more permanent location
                # or process it directly. For now, we'll just confirm it's saved.
                # The temp_filepath is where it's stored. A real app would manage this path better.

                # For this example, let's "permanently" save it in UPLOAD_FOLDER with original name
                # (In a real app, generate unique names, manage sessions/users etc.)
                # Ensure the filename is secure
                # from werkzeug.utils import secure_filename
                # secure_name = secure_filename(file.filename)
                # For simplicity, using original name.
                permanent_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

                # If a temp file was used and needs to be moved:
                # os.rename(temp_filepath, permanent_filepath)
                # os.rmdir(temp_dir)
                # If file.save() saved directly to permanent_filepath (after ensuring UPLOAD_FOLDER exists), then no rename.
                # The current code saves to temp_filepath then renames.

                # For this version, let's ensure the file is saved to a known location directly for simplicity.
                # The temp_dir logic was a bit convoluted for this simple store.
                # We'll save directly to UPLOAD_FOLDER/filename
                # Re-saving after header check (if file was consumed by pd.read_csv)
                # This means we need to either pass the file object around or save then read.
                # Simpler: save, then read for validation. If validation fails, remove.

                # The file is already saved at temp_filepath. We'll use this path.
                # For this simple API, we just store the path of this (potentially temporary) file.
                # A better approach would be to move it to a managed storage area with a unique ID.

                # Store the path for later use
                uploaded_data_store['phenotypes_file'] = temp_filepath
                # Note: temp_filepath will be inside a directory that will be removed if not careful.
                # Let's save it to a known name inside UPLOAD_FOLDER directly.

                # Revised file saving:
                if os.path.exists(temp_filepath): # Clean up if it was from a previous attempt
                    os.remove(temp_filepath)
                if os.path.exists(temp_dir): # Clean up if it was from a previous attempt
                    os.rmdir(temp_dir)

                # Recreate temp_dir and save file there with original name
                temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER']) # for this session's files
                filepath_to_store = os.path.join(temp_dir, file.filename)
                file.seek(0) # Rewind file pointer before saving again if it was read
                file.save(filepath_to_store)

                # Now validate the saved file
                df_val = pd.read_csv(filepath_to_store, nrows=5)
                missing_headers_val = [h for h in expected_headers if h not in df_val.columns]
                if missing_headers_val:
                    os.remove(filepath_to_store) # remove the problematic file
                    os.rmdir(temp_dir) # remove the temp dir created for this upload
                    return jsonify({'error': f'Missing expected headers in CSV: {", ".join(missing_headers_val)}'}), 400

                # Store the filename (file_id) for later reference by /run/pblup
                # The actual path is UPLOAD_FOLDER + file.filename.
                # For simplicity, we'll use the filename as the file_id.
                # A more robust system would use unique IDs and map them to paths.
                uploaded_data_store['phenotypes_file'] = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                # Move the validated file from temp_dir to the main UPLOAD_FOLDER with its original name
                final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                os.rename(filepath_to_store, final_filepath)
                os.rmdir(temp_dir) # Clean up the temporary directory

                app.logger.info(f"Phenotypes file '{file.filename}' stored at: {final_filepath}")

                return jsonify({
                    'message': 'Phenotypic data CSV uploaded successfully and validated for headers.',
                    'filename': file.filename,
                    'file_id': file.filename # Return the filename as file_id
                }), 201

            except pd.errors.EmptyDataError:
                os.remove(temp_filepath)
                os.rmdir(temp_dir)
                return jsonify({'error': 'Uploaded CSV file is empty.'}), 400
            except pd.errors.ParserError:
                os.remove(temp_filepath)
                os.rmdir(temp_dir)
                return jsonify({'error': 'Failed to parse CSV file. Ensure it is a valid CSV.'}), 400
            except Exception as e: # Catch other potential errors during pandas read or file ops
                if os.path.exists(temp_filepath): os.remove(temp_filepath)
                if os.path.exists(temp_dir): os.rmdir(temp_dir)
                app.logger.error(f"Error processing phenotypic upload: {e}")
                return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

        except Exception as e:
            app.logger.error(f"Error saving uploaded file: {e}")
            return jsonify({'error': f'Failed to save uploaded file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed. Please upload a CSV file.'}), 400


@app.route('/api/upload/pedigree', methods=['POST'])
@api_key_required
def upload_pedigree():
    """
    Endpoint to upload pedigree data as a CSV file.
    Performs basic validation: checks for file presence, CSV extension, and expected headers.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
            temp_filepath = os.path.join(temp_dir, file.filename)
            file.save(temp_filepath)

            try:
                df = pd.read_csv(temp_filepath, nrows=5)
                expected_headers = ['AnimalID', 'SireID', 'DamID']

                missing_headers = [h for h in expected_headers if h not in df.columns]
                if missing_headers:
                    os.remove(temp_filepath)
                    os.rmdir(temp_dir)
                    return jsonify({'error': f'Missing expected headers in CSV: {", ".join(missing_headers)}'}), 400

                # Similar revised saving logic for pedigree
                if os.path.exists(temp_filepath): os.remove(temp_filepath)
                if os.path.exists(temp_dir): os.rmdir(temp_dir)

                temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
                filepath_to_store = os.path.join(temp_dir, file.filename)
                file.seek(0)
                file.save(filepath_to_store)

                df_val = pd.read_csv(filepath_to_store, nrows=5)
                missing_headers_val = [h for h in expected_headers if h not in df_val.columns]
                if missing_headers_val:
                    os.remove(filepath_to_store)
                    os.rmdir(temp_dir)
                    return jsonify({'error': f'Missing expected headers in CSV: {", ".join(missing_headers_val)}'}), 400

                # Store the filename (file_id) for later reference
                final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                os.rename(filepath_to_store, final_filepath)
                os.rmdir(temp_dir)

                uploaded_data_store['pedigree_file'] = final_filepath
                app.logger.info(f"Pedigree file '{file.filename}' stored at: {final_filepath}")

                return jsonify({
                    'message': 'Pedigree data CSV uploaded successfully and validated for headers.',
                    'filename': file.filename,
                    'file_id': file.filename # Return filename as file_id
                }), 201

            except pd.errors.EmptyDataError:
                os.remove(temp_filepath)
                os.rmdir(temp_dir)
                return jsonify({'error': 'Uploaded CSV file is empty.'}), 400
            except pd.errors.ParserError:
                os.remove(temp_filepath)
                os.rmdir(temp_dir)
                return jsonify({'error': 'Failed to parse CSV file. Ensure it is a valid CSV.'}), 400
            except Exception as e:
                if os.path.exists(temp_filepath): os.remove(temp_filepath)
                if os.path.exists(temp_dir): os.rmdir(temp_dir)
                app.logger.error(f"Error processing pedigree upload: {e}")
                return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

        except Exception as e:
            app.logger.error(f"Error saving uploaded pedigree file: {e}")
            return jsonify({'error': f'Failed to save uploaded file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed. Please upload a CSV file.'}), 400


@app.route('/api/upload/genomic', methods=['POST'])
@api_key_required
def upload_genomic_data():
    """
    Endpoint to upload genomic data (SNP data) as a CSV file.
    Validates file presence, extension, and basic headers (e.g., 'AnimalID').
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Save to a temporary directory first
            temp_dir = tempfile.mkdtemp(dir=app.config['UPLOAD_FOLDER'])
            temp_filepath = os.path.join(temp_dir, file.filename)
            file.save(temp_filepath)

            try:
                # Basic CSV validation: check for 'AnimalID' and at least one other column (presumed SNP)
                df_val = pd.read_csv(temp_filepath, nrows=5)
                if 'AnimalID' not in df_val.columns:
                    os.remove(temp_filepath)
                    os.rmdir(temp_dir)
                    return jsonify({'error': "Missing 'AnimalID' header in genomic CSV."}), 400
                if len(df_val.columns) < 2:
                    os.remove(temp_filepath)
                    os.rmdir(temp_dir)
                    return jsonify({'error': "Genomic CSV must contain 'AnimalID' and at least one SNP column."}), 400

                # Move to permanent UPLOAD_FOLDER location
                final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                os.rename(temp_filepath, final_filepath)
                os.rmdir(temp_dir)

                # Store path in uploaded_data_store (optional, if needed for direct reference outside tasks)
                # uploaded_data_store['genomic_file'] = final_filepath
                app.logger.info(f"Genomic data file '{file.filename}' stored at: {final_filepath}")

                return jsonify({
                    'message': 'Genomic data CSV uploaded successfully and validated for headers.',
                    'filename': file.filename,
                    'file_id': file.filename # Return filename as file_id
                }), 201

            except pd.errors.EmptyDataError:
                os.remove(temp_filepath); os.rmdir(temp_dir)
                return jsonify({'error': 'Uploaded genomic CSV file is empty.'}), 400
            except pd.errors.ParserError:
                os.remove(temp_filepath); os.rmdir(temp_dir)
                return jsonify({'error': 'Failed to parse genomic CSV file. Ensure it is a valid CSV.'}), 400
            except Exception as e:
                if os.path.exists(temp_filepath): os.remove(temp_filepath)
                if os.path.exists(temp_dir): os.rmdir(temp_dir)
                app.logger.error(f"Error processing genomic upload: {e}")
                return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

        except Exception as e:
            app.logger.error(f"Error saving uploaded genomic file: {e}")
            return jsonify({'error': f'Failed to save uploaded file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed. Please upload a CSV file for genomic data.'}), 400


if __name__ == '__main__':
    # Note: For development, Flask's built-in server is fine.
    # For production, use a proper WSGI server like Gunicorn or uWSGI.
    # app.run(debug=True, host='0.0.0.0', port=5000) # Commented out for direct DSL execution if needed.


@app.route('/api/run/ssgblup', methods=['POST'])
@api_key_required
@validate()
def create_ssgblup_task(body: RunSSGBLUPRequest):
    """
    Creates an ssGBLUP analysis task using specified phenotypic, pedigree, and genomic file IDs.
    Dispatches the task to Celery and returns a Celery task_id.
    """
    pheno_file_id = body.phenotypes_file_id
    ped_file_id = body.pedigree_file_id
    genomic_file_id = body.genomic_data_file_id
    trait_name = body.trait_name
    heritability = body.heritability
    assumed_phenotypic_variance = body.assumed_phenotypic_variance
    grm_method = body.grm_method
    h_inv_tuning_factor = body.h_inv_tuning_factor

    pheno_file_path = os.path.join(app.config['UPLOAD_FOLDER'], pheno_file_id)
    ped_file_path = os.path.join(app.config['UPLOAD_FOLDER'], ped_file_id)
    genomic_file_path = os.path.join(app.config['UPLOAD_FOLDER'], genomic_file_id)

    if not os.path.exists(pheno_file_path):
        return jsonify({'error': f'Phenotypic data file not found: {pheno_file_id}.'}), 400
    if not os.path.exists(ped_file_path):
        return jsonify({'error': f'Pedigree data file not found: {ped_file_id}.'}), 400
    if not os.path.exists(genomic_file_path):
        return jsonify({'error': f'Genomic data file not found: {genomic_file_id}.'}), 400

    genetic_variance = heritability * assumed_phenotypic_variance
    residual_variance = (1 - heritability) * assumed_phenotypic_variance

    celery_task = run_ssgblup_analysis_task.delay(
        phenotypes_file_path=pheno_file_path,
        pedigree_file_path=ped_file_path,
        genomic_data_file_path=genomic_file_path,
        trait_col=trait_name,
        var_animal=genetic_variance,
        var_residual=residual_variance,
        grm_method=grm_method,
        h_inv_tuning_factor=h_inv_tuning_factor,
        task_id=None
    )

    task_initial_params = {
        'phenotypes_file_id': pheno_file_id,
        'pedigree_file_id': ped_file_id,
        'genomic_data_file_id': genomic_file_id,
        'trait_name': trait_name,
        'heritability': heritability,
        'assumed_phenotypic_variance': assumed_phenotypic_variance,
        'grm_method': grm_method,
        'h_inv_tuning_factor': h_inv_tuning_factor,
        'analysis_type': 'ssGBLUP',
        'calculated_genetic_variance': genetic_variance,
        'calculated_residual_variance': residual_variance
    }
    try:
        redis_client.set(f"task_params:{celery_task.id}", json.dumps(task_initial_params), ex=86400)
        app.logger.info(f"ssGBLUP task dispatched. Celery Task ID: {celery_task.id}. Params: {task_initial_params}")
    except redis.exceptions.RedisError as e:
        app.logger.error(f"RedisError: Failed to store ssGBLUP task parameters for {celery_task.id}: {e}")

    return jsonify({
        'message': 'ssGBLUP analysis task successfully dispatched.',
        'task_id': celery_task.id
    }), 202


@app.route('/api/run/gblup', methods=['POST'])
@api_key_required
@validate()
def create_gblup_task(body: RunGBLUPRequest):
    """
    Creates a GBLUP analysis task using specified phenotypic and genomic file IDs.
    Dispatches the task to Celery and returns a Celery task_id.
    """
    pheno_file_id = body.phenotypes_file_id
    genomic_file_id = body.genomic_data_file_id
    trait_name = body.trait_name
    heritability = body.heritability
    assumed_phenotypic_variance = body.assumed_phenotypic_variance

    pheno_file_path = os.path.join(app.config['UPLOAD_FOLDER'], pheno_file_id)
    genomic_file_path = os.path.join(app.config['UPLOAD_FOLDER'], genomic_file_id)

    if not os.path.exists(pheno_file_path):
        return jsonify({'error': f'Phenotypic data file not found: {pheno_file_id}.'}), 400
    if not os.path.exists(genomic_file_path):
        return jsonify({'error': f'Genomic data file not found: {genomic_file_id}.'}), 400

    genetic_variance = heritability * assumed_phenotypic_variance
    residual_variance = (1 - heritability) * assumed_phenotypic_variance

    celery_task = run_gblup_analysis_task.delay(
        phenotypes_file_path=pheno_file_path,
        genomic_data_file_path=genomic_file_path,
        trait_col=trait_name,
        var_animal=genetic_variance, # This is var_g for GBLUP task
        var_residual=residual_variance,
        task_id=None
    )

    task_initial_params = {
        'phenotypes_file_id': pheno_file_id,
        'genomic_data_file_id': genomic_file_id,
        'trait_name': trait_name,
        'heritability': heritability,
        'assumed_phenotypic_variance': assumed_phenotypic_variance,
        'analysis_type': 'GBLUP',
        'calculated_genetic_variance': genetic_variance,
        'calculated_residual_variance': residual_variance
    }
    try:
        redis_client.set(f"task_params:{celery_task.id}", json.dumps(task_initial_params), ex=86400)
        app.logger.info(f"GBLUP task dispatched. Celery Task ID: {celery_task.id}. Params: {task_initial_params}")
    except redis.exceptions.RedisError as e:
        app.logger.error(f"RedisError: Failed to store GBLUP task parameters for {celery_task.id}: {e}")
        # Optional: Revoke Celery task
        # return jsonify({'error': 'Failed to store task parameters, task dispatch aborted/revoked.'}), 500

    return jsonify({
        'message': 'GBLUP analysis task successfully dispatched.',
        'task_id': celery_task.id
    }), 202


@app.route('/api/run/pblup', methods=['POST'])
@api_key_required
def create_pblup_task():
    """
    Creates a PBLUP analysis task using the most recently uploaded phenotypic and pedigree data.
    Dispatches the task to Celery and returns a task_id.
    """
    pheno_file_path = uploaded_data_store.get('phenotypes_file')
    ped_file_path = uploaded_data_store.get('pedigree_file')

    if not pheno_file_path or not os.path.exists(pheno_file_path):
        return jsonify({'error': 'Phenotypic data not found. Please upload via /api/upload/phenotypes.'}), 400
    if not ped_file_path or not os.path.exists(ped_file_path):
        return jsonify({'error': 'Pedigree data not found. Please upload via /api/upload/pedigree.'}), 400

    # Generate a unique task ID for our own tracking, though Celery will have its own.
    # We can use Celery's ID directly if preferred. For now, let's generate one.
    # internal_task_id = uuid.uuid4().hex # Not strictly needed if Celery ID is the main reference

    # Dispatch Celery task
    celery_task = run_pblup_analysis_task.delay(
        phenotypes_file_path=pheno_file_path,
        pedigree_file_path=ped_file_path,
        task_id=None # Celery generates its own task_id, which we will use.
                     # The task_id param in the Celery func was for optional logging, can be removed from there.
    )

    # Store initial task parameters in Redis, keyed by Celery's task ID.
    # These parameters might be useful for auditing, display, or re-queueing.
    task_params = {
        'phenotypes_file': pheno_file_path,
        'pedigree_file': ped_file_path,
        'analysis_type': 'PBLUP'
        # Add other relevant initial params like user_id, submission_time etc. if available
    }
    try:
        # Store for 24 hours (86400 seconds)
        redis_client.set(f"task_params:{celery_task.id}", json.dumps(task_params), ex=86400)
        app.logger.info(f"PBLUP task dispatched. Celery Task ID: {celery_task.id}. Parameters stored in Redis.")
    except redis.exceptions.RedisError as e:
        app.logger.error(f"RedisError: Failed to store task parameters for {celery_task.id}: {e}")
        # Decide if this failure is critical enough to not accept the task.
        # For now, log error and proceed. Celery task is already queued.
        # Could try to revoke task if Redis store fails: celery_app.control.revoke(celery_task.id)

    return jsonify({
        'message': 'PBLUP analysis task successfully dispatched.',
        'task_id': celery_task.id # Return Celery's task ID
    }), 202 # 202 Accepted


@app.route('/api/task/<string:task_id>/status', methods=['GET'])
@api_key_required
def get_task_status(task_id):
    """
    Retrieves the status of a specific Celery task and its initial parameters from Redis.
    """
    task_celery_result = AsyncResult(task_id, app=celery_app)

    response = {
        'task_id': task_id,
        'status': task_celery_result.state # PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
    }

    # Fetch initial parameters from Redis
    try:
        task_params_json = redis_client.get(f"task_params:{task_id}")
        if task_params_json:
            response['submitted_parameters'] = json.loads(task_params_json)
        else:
            response['submitted_parameters'] = "Not found or expired." # Or {}
    except redis.exceptions.RedisError as e:
        app.logger.error(f"RedisError: Failed to retrieve task parameters for {task_id}: {e}")
        response['submitted_parameters'] = "Error retrieving parameters."
    except json.JSONDecodeError as e:
        app.logger.error(f"JSONDecodeError: Failed to parse task parameters for {task_id} from Redis: {e}")
        response['submitted_parameters'] = "Error parsing stored parameters."


    if task_celery_result.state == 'FAILURE':
        response['error_info'] = str(task_celery_result.info) # Exception info
    elif task_celery_result.state == 'PROGRESS': # If task uses update_state with meta
        response['progress'] = task_celery_result.info

    return jsonify(response), 200


@app.route('/api/task/<string:task_id>/result', methods=['GET'])
@api_key_required
def get_task_result(task_id):
    """
    Retrieves the result of a completed Celery task.
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if not task_result: # Should not happen if task_id is valid Celery ID
        return jsonify({'error': 'Task result object not found (invalid Celery setup or ID format issue).'}), 404

    if task_result.ready(): # Task has finished (either SUCCESS or FAILURE)
        if task_result.successful():
            return jsonify({
                'task_id': task_id,
                'status': task_result.state, # SUCCESS
                'result': task_result.get()
            }), 200
        elif task_result.failed():
            return jsonify({
                'task_id': task_id,
                'status': task_result.state, # FAILURE
                'error': str(task_result.info), # Exception raised by the task
                'traceback': task_result.traceback # Full traceback
            }), 200 # Or 500 if indicating server-side processing error
    else: # Task is PENDING, STARTED, RETRY, etc.
        return jsonify({
            'task_id': task_id,
            'status': task_result.state,
            'message': 'Results are not yet available or task is still processing.'
        }), 202 # 202 Accepted, client should poll again


# The /api/dev/process_task endpoint is removed as Celery workers handle processing.

# Cleanup of temporary files in uploaded_data_store and task-associated files in tasks_db:
# - uploaded_data_store files are overwritten by new uploads. Their temp dirs might not be cleaned until server restart.
# - Task file paths (copied into tasks_db then passed to Celery) are also from these temp dirs.
# A robust system would move these to a managed storage (e.g., task-specific directory) upon task creation
# and implement a cleanup strategy for these managed files after task completion or expiry.

if __name__ == '__main__':
    # Note: For development, Flask's built-in server is fine.
    # For production, use a proper WSGI server like Gunicorn or uWSGI.
    app.run(debug=True, host='0.0.0.0', port=5000)
