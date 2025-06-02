import pathlib
import os

def load_dataset_py(dataset_name: str, file_name: str, data_subdir: str = "data") -> str:
    """
    Locates and returns the full path to a specified dataset file.
    Assumes datasets are packaged in a `data_subdir` (default 'data')
    relative to the package root.

    Args:
        dataset_name: The name of the dataset (subfolder within data_subdir).
        file_name: The specific file name within the dataset folder.
        data_subdir: The name of the main data directory. Defaults to "data".

    Returns:
        The full string path to the dataset file.

    Raises:
        FileNotFoundError: If the dataset file cannot be found.
    """
    try:
        # Assuming this file (datasets_py.py) is in something like:
        # <package_root>/genostockpy_core/datasets_py.py
        # or <package_root>/datasets_py.py
        # We want to find <package_root>/data/

        current_file_path = pathlib.Path(__file__).resolve() # Path to this .py file

        # Try to find a 'data' directory by going up from current file's location
        # This logic might need adjustment based on final package structure
        package_root_candidate = current_file_path.parent # Assuming this file is in a module dir

        # Search upwards for a directory that contains the data_subdir
        # This is a common pattern if the current file is nested.
        # Limit upward search to avoid going too far up.
        found_root = None
        for _ in range(5): # Try up to 5 levels up
            if (package_root_candidate / data_subdir).is_dir():
                found_root = package_root_candidate
                break
            if package_root_candidate.parent == package_root_candidate: # Reached filesystem root
                break
            package_root_candidate = package_root_candidate.parent

        if not found_root:
             # Fallback: assume 'data' is sibling to current file's dir, or in current dir
            if (current_file_path.parent / data_subdir).is_dir():
                base_data_path = current_file_path.parent / data_subdir
            elif (current_file_path.parent.parent / data_subdir).is_dir(): # If this file is in genostockpy/datasets/
                base_data_path = current_file_path.parent.parent / data_subdir
            else: # Last resort: check if 'data' is in the current working directory (less ideal for pkg)
                # This might be useful for examples run from the root of a project.
                # However, for installed package, __file__ based path is better.
                # For now, let's prioritize paths relative to the package structure.
                raise FileNotFoundError(f"Could not reliably determine package root to find '{data_subdir}' directory from {current_file_path}.")

        else:
            base_data_path = found_root / data_subdir

        dataset_file_path = base_data_path / dataset_name / file_name

        if dataset_file_path.is_file():
            return str(dataset_file_path)
        else:
            # For debugging, list contents if dataset_name folder exists
            dataset_folder_path = base_data_path / dataset_name
            if dataset_folder_path.is_dir():
                # print(f"Contents of {dataset_folder_path}: {os.listdir(dataset_folder_path)}")
                pass
            raise FileNotFoundError(
                f"Dataset file '{file_name}' not found in '{dataset_folder_path}'. "
                f"Searched from base data path: {base_data_path}"
            )
    except Exception as e:
        # Catch any other exception during path resolution and raise as FileNotFoundError for consistency
        raise FileNotFoundError(f"Error locating dataset {dataset_name}/{file_name}: {e}")

