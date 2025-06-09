# sheep_breeding_api/celery_app.py

from celery import Celery
import os
import pandas as pd
import numpy as np
import time # For simulating work and demonstrating status updates

# Adjust path to import from the root of the project for the toolkit
import sys
# Get the directory of the current file (celery_app.py)
current_api_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root_dir = os.path.dirname(current_api_dir)
sys.path.insert(0, project_root_dir)

from sheep_breeding_genomics.data_management.io_handlers import read_phenotypic_data, read_pedigree_data
from sheep_breeding_genomics.genetic_evaluation.relationship_matrix import calculate_nrm
from sheep_breeding_genomics.genetic_evaluation.blup_models import solve_animal_model_mme

# Define Celery application
# Configuration for Redis connection from environment variables
REDIS_HOST_CELERY = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT_CELERY = os.environ.get("REDIS_PORT", "6379") # Celery expects string here for URL formatting
REDIS_DB_CELERY = os.environ.get("REDIS_DB_CELERY", "0") # Celery worker might use a different DB

# Construct Broker and Backend URLs
# It's good practice to make the broker/backend URLs configurable.
# If using a different DB for Celery than the API's redis_client, ensure REDIS_DB_CELERY is set.
# For this example, we'll assume it could be the same as REDIS_DB_API or a specific one for Celery.
# If REDIS_DB_CELERY is not set, it defaults to 0.
broker_url = f"redis://{REDIS_HOST_CELERY}:{REDIS_PORT_CELERY}/{REDIS_DB_CELERY}"
result_backend_url = f"redis://{REDIS_HOST_CELERY}:{REDIS_PORT_CELERY}/{REDIS_DB_CELERY}"


celery_app = Celery(
    'sheep_breeding_api_tasks', # Name of the celery application
    broker=broker_url,
    backend=result_backend_url,
    include=['sheep_breeding_api.celery_app'] # Add module where tasks are defined
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # task_track_started=True, # Uncomment to get 'STARTED' state more reliably
    # task_send_sent_event=True, # For monitoring events
)


# This is the global tasks_db simulation. In a real scenario with Celery,
# task status and results are primarily managed by Celery's backend.
# However, Flask might still want to keep a reference or initial record.
# For this exercise, the task will update a shared structure (like a DB or a file-based store if Redis isn't fully used for this).
# The prompt mentions "Celery tasks update our tasks_db". This is complex with separate processes.
# A more standard Celery approach: Flask creates task, Celery worker executes, result/status in Celery backend.
# Flask then queries Celery backend.
# Let's assume Flask's `tasks_db` is for *initiating* tasks and Celery backend is source of truth for status/result.
# The task itself won't directly update a Flask in-memory dict.

@celery_app.task(bind=True) # bind=True gives access to self (the task instance)
def run_pblup_analysis_task(self,
                            phenotypes_file_path: str,
                            pedigree_file_path: str,
                            trait_col: str,
                            var_animal: float,
                            var_residual: float,
                            task_id: str = None): # task_id is optional, Celery provides self.request.id
    """
    Celery task to perform PBLUP analysis.
    Accepts analysis parameters.
    """
    # Use Celery's task ID for logging if our internal task_id is not passed or needed.
    current_task_id = task_id or self.request.id
    print(f"Task {current_task_id}: Starting PBLUP analysis.")
    # self.update_state(state='PROGRESS', meta={'current_step': 'Initializing'})

    try:
        time.sleep(2) # Simulate initial work

        pheno_df = read_phenotypic_data(phenotypes_file_path)
        if pheno_df.empty:
            raise ValueError(f'Task {current_task_id}: Failed to read phenotypic data or file is empty.')

        ped_df = read_pedigree_data(pedigree_file_path)
        if ped_df.empty:
            raise ValueError(f'Task {current_task_id}: Failed to read pedigree data or file is empty.')

        if trait_col not in pheno_df.columns:
             raise ValueError(f"Task {current_task_id}: Phenotypic data must contain the specified trait column '{trait_col}'.")

        time.sleep(2) # Simulate NRM calculation time

        nrm_df = calculate_nrm(ped_df, animal_col='AnimalID', sire_col='SireID', dam_col='DamID', founder_val='0')
        if nrm_df.empty:
            raise ValueError(f'Task {current_task_id}: NRM calculation failed.')
        np.fill_diagonal(nrm_df.values, np.diag(nrm_df.values) + 1e-6) # for invertibility

        time.sleep(3) # Simulate MME solving time

        ebv_df = solve_animal_model_mme(
            phenotypic_df=pheno_df,
            relationship_matrix_df=nrm_df,
            trait_col=trait_col, # Use passed trait_col
            animal_id_col='AnimalID', # Assuming this is standard
            var_animal=var_animal,       # Use passed var_animal
            var_residual=var_residual, # Use passed var_residual
            fixed_effects_cols=[]      # Assuming no additional fixed effects for now
        )

        if ebv_df.empty:
            raise ValueError(f'Task {current_task_id}: PBLUP analysis (MME solving) returned empty results.')

        print(f"Task {current_task_id}: PBLUP analysis completed successfully.")
        return ebv_df.to_dict(orient='records')

    except Exception as e:
        print(f"Task {current_task_id}: Failed. Error: {str(e)}")
        # Update Celery task state with failure info (optional if re-raising)
        # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise # Re-raise for Celery to mark as FAILURE and store traceback

if __name__ == '__main__':
    # This allows running celery worker directly using:
    # celery -A sheep_breeding_api.celery_app worker -l INFO
    # Ensure Redis server is running.
    # You might need to be in the project root directory or ensure PYTHONPATH is set.
    # Example: PYTHONPATH=$PYTHONPATH:/path/to/your/sheep_breeding_genomics_project celery -A sheep_breeding_api.celery_app worker -l INFO
    pass


@celery_app.task(bind=True)
def run_ssgblup_analysis_task(self,
                              phenotypes_file_path: str,
                              pedigree_file_path: str,
                              genomic_data_file_path: str,
                              trait_col: str,
                              var_animal: float, # This will be additive genetic variance (sigma_a^2 or u^2)
                              var_residual: float,
                              grm_method: str = "vanraden1",
                              h_inv_tuning_factor: float = 0.0, # Placeholder for future GRM tuning
                              task_id: str = None):
    """
    Celery task to perform ssGBLUP analysis.
    Note: h_inv_tuning_factor is currently a placeholder for potential GRM adjustments.
    The current calculate_h_inverse_matrix doesn't use it directly for GRM scaling,
    but it's included for future extension. GRM tuning usually happens before G_inv.
    """
    current_task_id = task_id or self.request.id
    print(f"Task {current_task_id}: Starting ssGBLUP analysis.")
    self.update_state(state='PROGRESS', meta={'current_step': 'Initializing ssGBLUP'})

    try:
        # 1. Load Phenotypic Data
        pheno_df = read_phenotypic_data(phenotypes_file_path)
        if pheno_df.empty:
            raise ValueError(f'Task {current_task_id}: Failed to read phenotypic data or file is empty.')
        if trait_col not in pheno_df.columns:
            raise ValueError(f"Task {current_task_id}: Phenotypic data must contain the specified trait column '{trait_col}'.")

        self.update_state(state='PROGRESS', meta={'current_step': 'Loading pedigree data'})
        time.sleep(1)
        # 2. Load Pedigree Data
        ped_df = read_pedigree_data(pedigree_file_path)
        if ped_df.empty:
            raise ValueError(f'Task {current_task_id}: Failed to read pedigree data or file is empty.')

        self.update_state(state='PROGRESS', meta={'current_step': 'Loading genomic data'})
        time.sleep(1)
        # 3. Load Genomic Data
        genomic_data_obj = read_genomic_data(genomic_data_file_path, animal_id_col='AnimalID')
        if genomic_data_obj is None or genomic_data_obj.data.empty:
            raise ValueError(f'Task {current_task_id}: Failed to read genomic data or file is empty/invalid.')

        # (Optional Validations - can be added here if needed, e.g. validate_pedigree_data)
        # from sheep_breeding_genomics.data_management.validators import validate_pedigree_data, validate_genomic_data
        # if not validate_pedigree_data(PedigreeData(ped_df)): # Assuming PedigreeData object for validator
        #     raise ValueError(f"Task {current_task_id}: Pedigree data validation failed.")
        # if not validate_genomic_data(genomic_data_obj)[0]: # Returns (bool, stats_df)
        #     raise ValueError(f"Task {current_task_id}: Genomic data validation failed.")


        self.update_state(state='PROGRESS', meta={'current_step': 'Calculating NRM'})
        time.sleep(2)
        # 4. Calculate NRM
        nrm_df = calculate_nrm(ped_df, animal_col='AnimalID', sire_col='SireID', dam_col='DamID', founder_val='0')
        if nrm_df.empty:
            raise ValueError(f'Task {current_task_id}: NRM calculation failed.')
        # Ensure NRM is invertible for A_inv and A22_inv calculations within H_inv
        np.fill_diagonal(nrm_df.values, np.diag(nrm_df.values) + 1e-6)


        self.update_state(state='PROGRESS', meta={'current_step': 'Calculating GRM'})
        time.sleep(2)
        # 5. Calculate GRM
        grm_df = calculate_grm(genomic_data_obj, method=grm_method)
        if grm_df.empty:
            raise ValueError(f'Task {current_task_id}: GRM calculation failed (method: {grm_method}).')
        # Ensure GRM is invertible for G_inv calculation within H_inv
        np.fill_diagonal(grm_df.values, np.diag(grm_df.values) + 1e-6)

        # 6. GRM Tuning (Conceptual for now, based on h_inv_tuning_factor)
        # This is where GRM might be blended with A22.
        # The current calculate_h_inverse_matrix does not do this blending internally.
        # If blending is needed:
        #   a. Extract A22 from NRM.
        #   b. Ensure GRM and A22 are for the same set of animals and aligned.
        #   c. grm_tuned = grm_df * (1 - h_inv_tuning_factor) + a22_df * h_inv_tuning_factor
        #   d. Pass grm_tuned to calculate_h_inverse_matrix.
        # For now, we pass the original GRM and the factor is illustrative.
        if h_inv_tuning_factor > 0.0:
            app.logger.info(f"Task {current_task_id}: GRM tuning factor {h_inv_tuning_factor} received, but advanced tuning not yet implemented in this version of calculate_h_inverse_matrix. Using raw GRM.")
            # Placeholder for future:
            # genotyped_ids_in_grm = grm_df.index.tolist()
            # a22_df = nrm_df.loc[genotyped_ids_in_grm, genotyped_ids_in_grm]
            # grm_df = grm_df * (1.0 - h_inv_tuning_factor) + a22_df * h_inv_tuning_factor


        self.update_state(state='PROGRESS', meta={'current_step': 'Calculating H-inverse matrix'})
        time.sleep(2)
        # 7. Calculate H-inverse
        h_inv_df = calculate_h_inverse_matrix(nrm_df, grm_df, nrm_inv_df=None) # nrm_inv computed inside
        if h_inv_df.empty:
            raise ValueError(f'Task {current_task_id}: H-inverse matrix calculation failed.')

        # Note: solve_ssgblup_model_mme expects H_inv to be for *all* animals in NRM.
        # Phenotypes should also correspond to animals in H_inv.
        # The current H_inv is structured based on NRM animals.

        self.update_state(state='PROGRESS', meta={'current_step': 'Solving MME for ssGBLUP'})
        time.sleep(3)
        # 8. Solve MME for ssGBLUP
        ssgebv_df = solve_ssgblup_model_mme(
            phenotypic_df=pheno_df, # Use original pheno_df; solver will align with H_inv
            h_inv_df=h_inv_df,
            trait_col=trait_col,
            animal_id_col='AnimalID',
            var_genetic=var_animal, # This is sigma_u^2 (total genetic variance)
            var_residual=var_residual,
            fixed_effects_cols=[]
        )

        if ssgebv_df.empty:
            raise ValueError(f'Task {current_task_id}: ssGBLUP analysis (MME solving) returned empty results.')

        print(f"Task {current_task_id}: ssGBLUP analysis completed successfully.")
        return ssgebv_df.to_dict(orient='records')

    except Exception as e:
        print(f"Task {current_task_id}: ssGBLUP analysis failed. Error: {str(e)}")
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise


@celery_app.task(bind=True)
def run_gblup_analysis_task(self,
                            phenotypes_file_path: str,
                            genomic_data_file_path: str,
                            trait_col: str,
                            var_animal: float, # This will be genomic variance (var_g)
                            var_residual: float,
                            task_id: str = None): # Optional internal task_id reference
    """
    Celery task to perform GBLUP analysis.
    """
    current_task_id = task_id or self.request.id
    print(f"Task {current_task_id}: Starting GBLUP analysis.")
    self.update_state(state='PROGRESS', meta={'current_step': 'Initializing GBLUP'})

    try:
        # 1. Load Phenotypic Data
        pheno_df = read_phenotypic_data(phenotypes_file_path)
        if pheno_df.empty:
            raise ValueError(f'Task {current_task_id}: Failed to read phenotypic data or file is empty.')
        if trait_col not in pheno_df.columns:
            raise ValueError(f"Task {current_task_id}: Phenotypic data must contain the specified trait column '{trait_col}'.")

        # 2. Load Genomic Data
        # Assuming read_genomic_data takes animal_id_col as a known parameter, e.g., 'AnimalID'
        # For this GBLUP, we assume the genomic data file has an 'AnimalID' column.
        genomic_data_obj = read_genomic_data(genomic_data_file_path, animal_id_col='AnimalID') # This returns a GenomicData object
        if genomic_data_obj is None or genomic_data_obj.data.empty:
            raise ValueError(f'Task {current_task_id}: Failed to read genomic data or file is empty/invalid.')

        # Optional: Validate genomic data (e.g., check for NaNs if GRM calculation cannot handle them)
        # from sheep_breeding_genomics.data_management.validators import validate_genomic_data
        # is_geno_valid, _ = validate_genomic_data(genomic_data_obj) # Default validation checks for [0,1,2,nan]
        # if not is_geno_valid:
        #     raise ValueError(f'Task {current_task_id}: Genomic data validation failed.')
        # calculate_grm itself checks for NaNs.

        self.update_state(state='PROGRESS', meta={'current_step': 'Calculating GRM'})
        time.sleep(2) # Simulate work

        # 3. Calculate GRM
        # calculate_grm expects a GenomicData object and no NaNs in genotype matrix.
        grm_df = calculate_grm(genomic_data_obj, method="vanraden1")
        if grm_df.empty:
            raise ValueError(f'Task {current_task_id}: GRM calculation failed.')
        np.fill_diagonal(grm_df.values, np.diag(grm_df.values) + 1e-6) # for invertibility

        # Filter phenotypes for animals present in the GRM
        # GBLUP typically evaluates only genotyped animals that have phenotypes.
        animals_in_grm = grm_df.index.tolist()
        pheno_df_gblup = pheno_df[pheno_df['AnimalID'].isin(animals_in_grm)]
        if pheno_df_gblup.empty:
            raise ValueError(f'Task {current_task_id}: No phenotypic records found for animals present in the GRM.')

        self.update_state(state='PROGRESS', meta={'current_step': 'Solving MME for GBLUP'})
        time.sleep(3) # Simulate work

        # 4. Solve MME using GRM
        gebv_df = solve_animal_model_mme(
            phenotypic_df=pheno_df_gblup,
            relationship_matrix_df=grm_df, # Pass GRM
            trait_col=trait_col,
            animal_id_col='AnimalID', # Assuming 'AnimalID'
            var_animal=var_animal,    # This is var_g
            var_residual=var_residual,
            fixed_effects_cols=[]     # Assuming no additional fixed effects for now
        )

        if gebv_df.empty:
            raise ValueError(f'Task {current_task_id}: GBLUP analysis (MME solving) returned empty results.')

        print(f"Task {current_task_id}: GBLUP analysis completed successfully.")
        return gebv_df.to_dict(orient='records')

    except Exception as e:
        print(f"Task {current_task_id}: GBLUP analysis failed. Error: {str(e)}")
        # Update Celery task state with failure info
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise # Re-raise for Celery to mark as FAILURE and store traceback
