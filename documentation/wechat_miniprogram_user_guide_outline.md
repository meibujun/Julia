# WeChat Mini-Program User Guide (Outline)

## 1. Introduction

*   **1.1. Welcome to the Sheep Breeding Genomics Assistant!**
    *   Purpose: Provide easy mobile access to genetic evaluation tools for sheep breeders.
    *   Benefit: Make informed breeding decisions on the go.
*   **1.2. Key Features at a Glance**
    *   Data Upload: Phenotypes, Pedigree, Genomic data via your phone.
    *   Analyses Available:
        *   Pedigree BLUP (PBLUP)
        *   Genomic BLUP (GBLUP)
        *   Single-Step GBLUP (ssGBLUP)
    *   Task Tracking: Monitor the progress of your analyses.
    *   Result Viewing: Access breeding values directly in the app.

## 2. Getting Started

*   **2.1. Finding and Accessing the Mini-Program**
    *   Searching within WeChat (e.g., "Sheep Genomics Eval").
    *   Scanning a QR code (if applicable).
    *   Accessing from official accounts or shared links.
*   **2.2. Account Management (Conceptual - if implemented)**
    *   Creating a new account (e.g., using WeChat login, phone number).
    *   Logging in and out.
    *   Password recovery / account security.
    *   (If no accounts: "Using the mini-program anonymously or with WeChat temporary identity").

## 3. Preparing Your Data for Upload

*   **3.1. General Data Requirements**
    *   All data must be in CSV (Comma Separated Values) format.
    *   Ensure consistent Animal IDs across all files.
    *   File naming conventions (if any).
*   **3.2. Phenotypic Data (`phenotypes.csv`)**
    *   **Required Columns:**
        *   `AnimalID`: Unique identifier for each animal.
        *   `TraitValue`: The actual measurement for the trait you want to analyze (e.g., FleeceWeight, GrowthRate).
    *   **Optional Columns:** Any other relevant information like birth year, sex, contemporary group (can be used as fixed effects if API supports it later).
    *   **Data Types:** `AnimalID` (text/numeric), `TraitValue` (numeric).
*   **3.3. Pedigree Data (`pedigree.csv`)**
    *   **Required Columns:**
        *   `AnimalID`: Unique identifier for each animal.
        *   `SireID`: Identifier of the animal's sire. Use '0' or leave blank for unknown sires.
        *   `DamID`: Identifier of the animal's dam. Use '0' or leave blank for unknown dams.
    *   **Data Types:** All IDs should be consistent (text/numeric).
    *   **Important:** All parents (SireID, DamID) should also appear as AnimalID in a row if they are not founders, or be listed as founders.
*   **3.4. Genomic Data (`genomic.csv`)**
    *   **Required Columns:**
        *   `AnimalID`: Unique identifier for each animal.
        *   SNP Columns: Each subsequent column represents a SNP marker (e.g., `SNP1`, `SNP_Marker_XYZ`).
    *   **Genotype Coding:**
        *   Expected values: `0`, `1`, `2` (representing the count of a specific allele).
        *   Missing Data: Leave cell blank or use standard missing value representations if supported by the backend (e.g., NaN, -9 - *current backend expects NaNs for missing, or pre-filtered data*).
*   **3.5. Data Quality Tips**
    *   Check for typos in AnimalIDs.
    *   Ensure consistency in unknown parent codes (e.g., always use '0').
    *   Remove header/footer rows from CSVs that are not part of the data.
    *   Verify numeric data is indeed numeric.
*   **3.6. File Size Limitations**
    *   Mention any limitations on CSV file sizes for uploads via the mini-program (e.g., "Max 5MB per file recommended for smooth uploads").

## 4. Running an Analysis

*   **4.1. Navigating to the Analysis Section**
    *   Main menu / "Start Analysis" button.
*   **4.2. Selecting Analysis Type**
    *   Choosing between PBLUP, GBLUP, ssGBLUP.
*   **4.3. Uploading Data Files (Conceptual Screen 2)**
    *   **Phenotypic Data:**
        *   "Upload Phenotypes CSV" button.
        *   Select file from phone storage/WeChat files.
        *   Confirmation of successful upload (e.g., filename displayed, `file_id` received internally).
    *   **Pedigree Data (for PBLUP, ssGBLUP):**
        *   "Upload Pedigree CSV" button.
        *   Selection and confirmation.
    *   **Genomic Data (for GBLUP, ssGBLUP):**
        *   "Upload Genomic CSV" button.
        *   Selection and confirmation.
*   **4.4. Specifying Analysis Parameters**
    *   **Common Parameters:**
        *   `Trait Name`: Name of the column in your phenotypic file to analyze (e.g., "TraitValue", "FleeceWeight"). Defaults to "TraitValue".
        *   `Heritability (h²)`: Estimated heritability of the trait (slider or input field, 0.001-0.999). Defaults to 0.3.
        *   `(Optional/Advanced)` `Assumed Phenotypic Variance`: Defaults to 10.0. Used with heritability to derive genetic/residual variances.
    *   **ssGBLUP Specific Parameters (Conceptual - if exposed):**
        *   `GRM Method`: (e.g., "vanraden1" - likely fixed in initial version).
        *   `H-inverse Tuning Factor`: (e.g., 0.0-1.0 - likely fixed or default in initial version).
*   **4.5. Submitting the Analysis**
    *   "Run Analysis" / "Submit Task" button.
    *   Confirmation message with a `Task ID`. (e.g., "Analysis submitted! Your Task ID is: xxxxxxxx").
    *   Instruction to save or note the Task ID for tracking.

## 5. Tracking Task Status (Conceptual Screen 3)

*   **5.1. Accessing Task List/Status Page**
    *   "My Tasks" / "Analysis History" section.
*   **5.2. Finding Your Task**
    *   Entering the `Task ID`.
    *   Or, listing recent tasks if account management is implemented.
*   **5.3. Understanding Task Statuses:**
    *   `PENDING`: Your analysis is waiting in the queue to be processed.
    *   `STARTED`/`PROCESSING`: Your analysis is currently being run by the server.
    *   `SUCCESS`/`COMPLETED`: Your analysis finished successfully. Results are available.
    *   `FAILURE`/`FAILED`: The analysis could not be completed due to an error. (See Troubleshooting).

## 6. Viewing and Interpreting Results (Conceptual Screen 3a)

*   **6.1. Accessing Results for a Completed Task**
    *   From the task status page, if 'COMPLETED', a "View Results" button will appear.
*   **6.2. Understanding Breeding Values (EBVs/GEBVs/ssGEBVs)**
    *   Results displayed: List of AnimalIDs and their corresponding breeding values.
    *   Brief explanation:
        *   Higher values are generally better (for positively desired traits).
        *   These values predict the genetic merit an animal will pass to its offspring.
        *   Compare values within the same analysis run.
    *   (Optional) Simple sorting or filtering of results in the mini-program.
*   **6.3. Downloading Results (Conceptual)**
    *   Option to export results (e.g., as a simple CSV or text if feasible).

## 7. Troubleshooting / FAQ

*   **7.1. Common Error Messages:**
    *   "File upload failed: Invalid format." (Ensure CSV, check for corruption).
    *   "Missing expected headers: AnimalID..." (Check your CSV column names against requirements).
    *   "Task Failed: NRM Calculation Error." (Likely an issue with pedigree structure, e.g., animals without parents defined as founders).
    *   "Task Failed: GRM Calculation Error." (Check genomic data format, missing values if not handled).
    *   "Task Failed: MME Solving Error." (Could be data quality, very small dataset, or inconsistent data leading to matrix issues).
*   **7.2. What if my analysis fails?**
    *   Review the error message provided.
    *   Double-check your input data files for formatting and consistency.
    *   Try with a smaller, known-good dataset if possible to isolate the issue.
*   **7.3. Data Privacy and Security**
    *   Brief statement on how data is handled (e.g., "Uploaded files are temporarily stored for analysis and then...").
*   **7.4. Contact Support**
    *   Email address or contact method for help.

## 8. Glossary (Optional)

*   **AnimalID:** Unique identifier for an animal.
*   **EBV (Estimated Breeding Value):** An estimate of an animal's genetic merit for a trait based on pedigree and phenotypic data.
*   **GEBV (Genomic Estimated Breeding Value):** An EBV that incorporates genomic (SNP) data.
*   **ssGEBV (Single-Step Genomic Estimated Breeding Value):** An EBV from an analysis that combines pedigree, phenotypic, and genomic data for all animals simultaneously.
*   **Heritability (h²):** The proportion of phenotypic variation in a trait that is due to genetic factors.
*   **NRM (Numerator Relationship Matrix):** A matrix describing pedigree relationships between animals.
*   **GRM (Genomic Relationship Matrix):** A matrix describing relationships based on shared SNP markers.
*   **SNP (Single Nucleotide Polymorphism):** A common type of genetic variation.

---
This outline provides a good structure for the user guide.
