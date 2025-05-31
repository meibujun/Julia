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

if __name__ == '__main__':
    # Example usage (assuming a certain directory structure for testing)
    # To test this, you'd need to create a dummy structure like:
    # <some_dir_to_run_from>/
    #   genostockpy_core/ (or wherever datasets_py.py is)
    #     datasets_py.py
    #   data/
    #     sample_dataset/
    #       sample_file.csv

    # Create dummy structure for testing if run directly
    if not (pathlib.Path.cwd() / "data" / "sample_dataset").exists():
        print("Creating dummy data structure for __main__ test...")
        (pathlib.Path.cwd() / "data" / "sample_dataset").mkdir(parents=True, exist_ok=True)
        with open(pathlib.Path.cwd() / "data" / "sample_dataset" / "sample_file.csv", "w") as f:
            f.write("col1,col2\n1,2\n3,4")

        # Assume datasets_py.py is in a subdirectory for this test to work like in a package
        # e.g. <cwd>/module_dir/datasets_py.py
        # If datasets_py.py is at cwd(), the path logic might need adjustment or test needs specific setup.
        # The current logic tries to find 'data' relative to datasets_py.py location.
        # If run from where datasets_py.py is, and 'data' is sibling to it, it won't find it with current logic.
        # This __main__ is more for illustrative purposes. Proper testing via unittest is better.

    try:
        # This test from __main__ assumes datasets_py.py is in a directory,
        # and 'data' is a sibling to that directory's parent.
        # e.g. project_root/src_module/datasets_py.py and project_root/data/
        # If datasets_py.py is at project_root, then Path(__file__).parent / "data" would be better.
        # The implemented logic is more robust for typical package structures.

        # For a test from __main__, let's assume 'data' is in CWD if we created it above
        # We'd need to ensure load_dataset_py can find it.
        # One way is to set current_file_path.parent.parent to CWD if it's run directly.

        # This example might fail if the relative paths don't match how __file__ resolves when run directly.
        # A proper test would set up the directory structure and call load_dataset_py.

        # To make it work if 'data' is in CWD and this script is also in CWD:
        # We can temporarily adjust how base_data_path is found in the __main__ block
        # by ensuring one of the fallback conditions in load_dataset_py matches.
        # The current `load_dataset_py` is written to be part of a package structure.

        # For a simple test if `data` is in CWD:
        test_path = pathlib.Path.cwd() / "data" / "sample_dataset" / "sample_file.csv"
        if test_path.exists():
             print(f"Dummy file exists at: {test_path}")
             # To test load_dataset_py directly, it needs to find the 'data' dir from its own location.
             # If datasets_py.py is in CWD, and 'data' is in CWD, then
             # (current_file_path.parent / data_subdir) would be (CWD / "data") which is correct.
             # This should work if datasets_py.py is in the same directory as the 'data' folder.
             # For the implemented logic (expecting it to be in a module subdir):
             # Let's assume this script is in project_root/module/datasets_py.py
             # and data is in project_root/data
             # We'd call: load_dataset_py("sample_dataset", "sample_file.csv")

             # This example assumes a structure where this script is in a directory,
             # and the 'data' directory is a sibling to that directory's parent.
             # To simulate this for a direct run where 'data' is in CWD:
             # if we pretend datasets_py.py is in <cwd>/src/datasets_py.py
             # then Path(__file__).parent.parent = cwd
             # then Path(__file__).parent.parent / "data" = <cwd>/data

             # For this simple __main__ test, let's assume the structure where 'data' is a subdir of cwd
             # and datasets_py.py is also in cwd. The fallback logic might catch this.

             # The implemented logic:
             # current_file_path.parent / data_subdir
             # current_file_path.parent.parent / data_subdir
             # If datasets_py.py is in CWD, current_file_path.parent is CWD. So CWD/"data" is checked.

            file_path = load_dataset_py("sample_dataset", "sample_file.csv")
            print(f"Found dataset: {file_path}")
        else:
            print(f"Dummy file {test_path} does not exist for __main__ test.")

    except FileNotFoundError as e:
        print(e)
    finally:
        # Clean up dummy structure
        if (pathlib.Path.cwd() / "data" / "sample_dataset" / "sample_file.csv").exists():
            os.remove(pathlib.Path.cwd() / "data" / "sample_dataset" / "sample_file.csv")
        if (pathlib.Path.cwd() / "data" / "sample_dataset").exists():
            os.rmdir(pathlib.Path.cwd() / "data" / "sample_dataset")
        if (pathlib.Path.cwd() / "data").exists() and not os.listdir(pathlib.Path.cwd() / "data"):
            os.rmdir(pathlib.Path.cwd() / "data")

```

Now, creating `output_utils_py.py`.
