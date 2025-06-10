# Sheep Breeding Management System - API Design

## Introduction

This document outlines the API endpoints for the Sheep Breeding Management System. The API provides a RESTful interface for frontend applications to interact with backend services, including data management, genetic evaluation, mating planning, and a natural language interface.

Each endpoint description includes:
*   **Endpoint Path**
*   **HTTP Method**
*   **Description**
*   **Request Body (if any)**: JSON structure.
*   **Success Response**: Status code and JSON structure.
*   **Error Responses (examples)**: Status codes and potential error message structure.

Conceptual mapping to backend modules:
*   Animal, Phenotypic Record, Trait, and Pedigree management endpoints conceptually map to `data_manager.py`.
*   Genetic Evaluation endpoints conceptually map to `genetic_evaluator.py`.
*   Mating Plan endpoints conceptually map to `mating_planner.py`.
*   NLP Interface endpoint conceptually maps to `nlp_interface_designer.py` which would then dispatch to other services.

---

## 1. Animal Management

These endpoints manage animal records in the system.

### 1.1 Add a new animal
*   **Endpoint Path:** `/animals`
*   **HTTP Method:** `POST`
*   **Description:** Adds a new animal to the system. The `IsActive` flag is conceptually managed by the backend, defaulting to `true`.
*   **Request Body:**
    ```json
    {
        "Eartag": "L001",
        "Sex": "Male",
        "BirthDate": "2023-04-15",
        "Breed": "Merino",
        "BirthWeight": 4.2,
        "WeaningWeight": 25.5,
        "PurchaseDate": null,
        "SaleDate": null,
        "DeathDate": null,
        "CurrentOwnerID": 1,
        "Notes": "Healthy lamb."
    }
    ```
*   **Success Response:**
    *   Status: `201 Created`
    *   Body:
        ```json
        {
            "AnimalID": 123,
            "Eartag": "L001",
            "Sex": "Male",
            "BirthDate": "2023-04-15",
            // ... other fields including IsActive
            "message": "Animal added successfully."
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Missing required fields (e.g., Eartag, Sex, BirthDate)"}` or `{"error": "Invalid data format for BirthDate."}`
    *   `409 Conflict`: `{"error": "Animal with Eartag L001 already exists."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 1.2 List all animals
*   **Endpoint Path:** `/animals`
*   **HTTP Method:** `GET`
*   **Description:** Retrieves a list of animals, with optional pagination and filtering.
*   **Query Parameters (Optional):**
    *   `page` (integer, default 1): Page number for pagination.
    *   `limit` (integer, default 20): Number of animals per page.
    *   `breed` (string): Filter by breed.
    *   `sex` (string): Filter by sex.
    *   `is_active` (boolean, default true): Filter by active status.
    *   `born_after` (date `YYYY-MM-DD`): Filter animals born after this date.
    *   `born_before` (date `YYYY-MM-DD`): Filter animals born before this date.
*   **Success Response:**
    *   Status: `200 OK`
    *   Body:
        ```json
        {
            "page": 1,
            "limit": 20,
            "total_animals": 150,
            "total_pages": 8,
            "data": [
                {
                    "AnimalID": 123,
                    "Eartag": "L001",
                    "Sex": "Male",
                    "BirthDate": "2023-04-15",
                    "Breed": "Merino",
                    "IsActive": true
                    // ... other summary fields
                }
                // ... more animals
            ]
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Invalid page or limit parameter."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 1.3 Get details for a specific animal
*   **Endpoint Path:** `/animals/{animal_id}`
*   **HTTP Method:** `GET`
*   **Description:** Retrieves detailed information for a specific animal by its `AnimalID`.
*   **Success Response:**
    *   Status: `200 OK`
    *   Body:
        ```json
        {
            "AnimalID": 123,
            "Eartag": "L001",
            "Sex": "Male",
            "BirthDate": "2023-04-15",
            "Breed": "Merino",
            "BirthWeight": 4.2,
            "WeaningWeight": 25.5,
            "PurchaseDate": null,
            "SaleDate": null,
            "DeathDate": null,
            "CurrentOwnerID": 1,
            "Notes": "Healthy lamb.",
            "IsActive": true
            // Potentially sire/dam basic info or links
        }
        ```
*   **Error Responses:**
    *   `404 Not Found`: `{"error": "Animal with ID {animal_id} not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 1.4 Update an animal's details
*   **Endpoint Path:** `/animals/{animal_id}`
*   **HTTP Method:** `PUT`
*   **Description:** Updates information for an existing animal. Only fields provided in the request body are updated.
*   **Request Body:** (Example: updating WeaningWeight and Notes)
    ```json
    {
        "WeaningWeight": 26.0,
        "Notes": "Updated notes.",
        "Breed": "Poll Dorset"
    }
    ```
*   **Success Response:**
    *   Status: `200 OK`
    *   Body: (The updated animal object)
        ```json
        {
            "AnimalID": 123,
            "Eartag": "L001",
            // ... all fields, updated and existing
            "WeaningWeight": 26.0,
            "Notes": "Updated notes.",
            "Breed": "Poll Dorset",
            "message": "Animal details updated successfully."
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Invalid data format for WeaningWeight."}`
    *   `404 Not Found`: `{"error": "Animal with ID {animal_id} not found."}`
    *   `409 Conflict`: If update violates a unique constraint (e.g., Eartag).
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 1.5 Deactivate (soft delete) an animal
*   **Endpoint Path:** `/animals/{animal_id}`
*   **HTTP Method:** `DELETE`
*   **Description:** Marks an animal as inactive (soft delete). The actual record is not removed from the database.
*   **Success Response:**
    *   Status: `200 OK` (or `204 No Content`)
    *   Body:
        ```json
        {
            "message": "Animal {animal_id} deactivated successfully."
        }
        ```
*   **Error Responses:**
    *   `404 Not Found`: `{"error": "Animal with ID {animal_id} not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

---

## 2. Phenotypic Record Management

Endpoints for managing phenotypic measurements of animals.

### 2.1 Add a new phenotypic record
*   **Endpoint Path:** `/phenotypic_records`
*   **HTTP Method:** `POST`
*   **Description:** Adds a new phenotypic measurement for an animal.
*   **Request Body:**
    ```json
    {
        "AnimalID": 123,
        "TraitID": 101,
        "MeasurementDate": "2023-07-20",
        "Value": 25.5,
        "RecordedByUserID": 2,
        "Notes": "Standard weaning weight measurement."
    }
    ```
*   **Success Response:**
    *   Status: `201 Created`
    *   Body:
        ```json
        {
            "RecordID": 501,
            "AnimalID": 123,
            "TraitID": 101,
            "MeasurementDate": "2023-07-20",
            "Value": 25.5,
            // ... other fields
            "message": "Phenotypic record added successfully."
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Missing required fields (AnimalID, TraitID, Value, MeasurementDate)."}`
    *   `404 Not Found`: `{"error": "AnimalID 123 or TraitID 101 not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 2.2 Get all phenotypic records for a specific animal
*   **Endpoint Path:** `/animals/{animal_id}/phenotypic_records`
*   **HTTP Method:** `GET`
*   **Description:** Retrieves all phenotypic records for a given `AnimalID`.
*   **Query Parameters (Optional):**
    *   `trait_id` (integer): Filter records by a specific `TraitID`.
*   **Success Response:**
    *   Status: `200 OK`
    *   Body:
        ```json
        [
            {
                "RecordID": 501,
                "AnimalID": 123,
                "TraitID": 101,
                "TraitName": "Weaning Weight", // Joined data
                "MeasurementDate": "2023-07-20",
                "Value": 25.5,
                "UnitOfMeasure": "kg", // Joined data
                "Notes": "Standard weaning weight measurement."
            }
            // ... more records
        ]
        ```
*   **Error Responses:**
    *   `404 Not Found`: `{"error": "Animal with ID {animal_id} not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 2.3 Get all phenotypic records for a specific trait
*   **Endpoint Path:** `/traits/{trait_id}/phenotypic_records`
*   **HTTP Method:** `GET`
*   **Description:** Retrieves all phenotypic records for a given `TraitID`.
*   **Success Response:**
    *   Status: `200 OK`
    *   Body: (Similar structure to 2.2, but records are for the given trait across animals)
        ```json
        [
            {
                "RecordID": 501,
                "AnimalID": 123,
                "Eartag": "L001", // Joined data
                "TraitID": 101,
                "MeasurementDate": "2023-07-20",
                "Value": 25.5,
                "Notes": "Standard weaning weight measurement."
            }
            // ... more records
        ]
        ```
*   **Error Responses:**
    *   `404 Not Found`: `{"error": "Trait with ID {trait_id} not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 2.4 Update a phenotypic record
*   **Endpoint Path:** `/phenotypic_records/{record_id}`
*   **HTTP Method:** `PUT`
*   **Description:** Updates an existing phenotypic record.
*   **Request Body:** (Fields to update)
    ```json
    {
        "Value": 26.0,
        "MeasurementDate": "2023-07-21",
        "Notes": "Re-measured."
    }
    ```
*   **Success Response:**
    *   Status: `200 OK`
    *   Body: (The updated phenotypic record object)
        ```json
        {
            "RecordID": 501,
            "AnimalID": 123,
            "TraitID": 101,
            "MeasurementDate": "2023-07-21",
            "Value": 26.0,
            "Notes": "Re-measured.",
            "message": "Phenotypic record updated successfully."
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Invalid data format for Value."}`
    *   `404 Not Found`: `{"error": "Phenotypic record with ID {record_id} not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 2.5 Delete a phenotypic record
*   **Endpoint Path:** `/phenotypic_records/{record_id}`
*   **HTTP Method:** `DELETE`
*   **Description:** Deletes a phenotypic record.
*   **Success Response:**
    *   Status: `200 OK` (or `204 No Content`)
    *   Body:
        ```json
        {
            "message": "Phenotypic record {record_id} deleted successfully."
        }
        ```
*   **Error Responses:**
    *   `404 Not Found`: `{"error": "Phenotypic record with ID {record_id} not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

---

## 3. Trait Management

Endpoints for managing traits.

### 3.1 Add a new trait
*   **Endpoint Path:** `/traits`
*   **HTTP Method:** `POST`
*   **Description:** Adds a new trait to the system.
*   **Request Body:**
    ```json
    {
        "TraitName": "Post-Weaning Growth Rate",
        "UnitOfMeasure": "g/day",
        "Description": "Average daily gain after weaning.",
        "Category": "Growth"
    }
    ```
*   **Success Response:**
    *   Status: `201 Created`
    *   Body:
        ```json
        {
            "TraitID": 105,
            "TraitName": "Post-Weaning Growth Rate",
            // ... other fields
            "message": "Trait added successfully."
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Missing required field TraitName."}`
    *   `409 Conflict`: `{"error": "Trait with name 'Post-Weaning Growth Rate' already exists."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 3.2 List all traits
*   **Endpoint Path:** `/traits`
*   **HTTP Method:** `GET`
*   **Description:** Retrieves a list of all traits.
*   **Success Response:**
    *   Status: `200 OK`
    *   Body:
        ```json
        [
            {
                "TraitID": 101,
                "TraitName": "Weaning Weight",
                "UnitOfMeasure": "kg",
                "Description": "Weight at weaning.",
                "Category": "Growth"
            }
            // ... more traits
        ]
        ```
*   **Error Responses:**
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 3.3 Get details for a specific trait
*   **Endpoint Path:** `/traits/{trait_id}`
*   **HTTP Method:** `GET`
*   **Description:** Retrieves detailed information for a specific trait.
*   **Success Response:**
    *   Status: `200 OK`
    *   Body: (Similar to one item in the list from 3.2)
*   **Error Responses:**
    *   `404 Not Found`: `{"error": "Trait with ID {trait_id} not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 3.4 Update a trait
*   **Endpoint Path:** `/traits/{trait_id}`
*   **HTTP Method:** `PUT`
*   **Description:** Updates an existing trait.
*   **Request Body:** (Fields to update)
    ```json
    {
        "UnitOfMeasure": "grams/day",
        "Description": "Average daily gain from birth to weaning, measured in grams per day."
    }
    ```
*   **Success Response:**
    *   Status: `200 OK`
    *   Body: (The updated trait object)
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Invalid data."}`
    *   `404 Not Found`: `{"error": "Trait with ID {trait_id} not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

---

## 4. Pedigree Management

Endpoints for managing animal pedigrees.

### 4.1 Add or update a pedigree link
*   **Endpoint Path:** `/pedigrees`
*   **HTTP Method:** `POST` (or `PUT` if AnimalID for pedigree link is part of path, e.g. `/animals/{animal_id}/pedigree`)
*   **Description:** Adds or updates the sire and dam for a given animal.
*   **Request Body:**
    ```json
    {
        "AnimalID": 123,
        "SireID": 55,  // Can be null
        "DamID": 78,   // Can be null
        "Notes": "Natural mating."
    }
    ```
*   **Success Response:**
    *   Status: `201 Created` (if new) or `200 OK` (if updated)
    *   Body:
        ```json
        {
            "AnimalID": 123,
            "SireID": 55,
            "DamID": 78,
            "Notes": "Natural mating.",
            "message": "Pedigree link processed successfully."
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "AnimalID is required."}`
    *   `404 Not Found`: `{"error": "AnimalID 123, SireID 55, or DamID 78 not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 4.2 Get sire and dam for an animal
*   **Endpoint Path:** `/animals/{animal_id}/pedigree`
*   **HTTP Method:** `GET`
*   **Description:** Retrieves the sire and dam for a specific animal.
*   **Success Response:**
    *   Status: `200 OK`
    *   Body:
        ```json
        {
            "AnimalID": 123,
            "SireID": 55,
            "SireEartag": "S001", // Joined data
            "DamID": 78,
            "DamEartag": "D002",   // Joined data
            "Notes": "Natural mating."
        }
        ```
*   **Error Responses:**
    *   `404 Not Found`: `{"error": "Pedigree information for AnimalID {animal_id} not found."}` (could also mean animal itself not found)
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 4.3 Get offspring for an animal
*   **Endpoint Path:** `/animals/{animal_id}/offspring`
*   **HTTP Method:** `GET`
*   **Description:** Retrieves all direct offspring (where the given animal is a sire or dam).
*   **Success Response:**
    *   Status: `200 OK`
    *   Body:
        ```json
        [
            {
                "AnimalID": 201,
                "Eartag": "L050",
                "Sex": "Female",
                "BirthDate": "2023-08-01",
                "RoleAsParent": "Sire" // Indicates {animal_id} was the sire of this offspring
            },
            {
                "AnimalID": 205,
                "Eartag": "L055",
                "Sex": "Male",
                "BirthDate": "2023-08-05",
                "RoleAsParent": "Sire"
            }
            // ... more offspring
        ]
        ```
*   **Error Responses:**
    *   `404 Not Found`: `{"error": "Animal with ID {animal_id} not found."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

---

## 5. Genetic Evaluation

Endpoints for genetic evaluations (EBVs, relationship matrix).

### 5.1 Get EBVs for a specific animal
*   **Endpoint Path:** `/animals/{animal_id}/ebvs`
*   **HTTP Method:** `GET`
*   **Description:** Retrieves Estimated Breeding Values (EBVs) for a specific animal.
*   **Query Parameters (Optional):**
    *   `trait_id` (integer): Get EBV for a specific trait.
    *   `evaluation_id` (string): Specify which evaluation run's EBVs to retrieve if multiple exist.
*   **Success Response:**
    *   Status: `200 OK`
    *   Body: (If specific trait_id requested)
        ```json
        {
            "AnimalID": 123,
            "TraitID": 101,
            "TraitName": "Weaning Weight",
            "EBV": 1.25,
            "Accuracy": 0.65, // If available
            "EvaluationID": "eval_202401"
        }
        ```
    *   Body: (If no specific trait_id, list of all available EBVs for the animal)
        ```json
        {
            "AnimalID": 123,
            "Eartag": "S010",
            "ebvs": [
                {"TraitID": 101, "TraitName": "Weaning Weight", "EBV": 1.25, "Accuracy": 0.65},
                {"TraitID": 102, "TraitName": "Fleece Weight", "EBV": 0.30, "Accuracy": 0.58}
            ],
            "EvaluationID": "eval_202401"
        }
        ```
*   **Error Responses:**
    *   `404 Not Found`: `{"error": "Animal with ID {animal_id} not found or no EBVs available."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

### 5.2 (Conceptual) Trigger a BLUP evaluation
*   **Endpoint Path:** `/genetic_evaluations/run_blup`
*   **HTTP Method:** `POST`
*   **Description:** Conceptually triggers a BLUP evaluation run. This would likely be an asynchronous operation.
*   **Request Body:**
    ```json
    {
        "trait_id": 101,
        "animal_ids_of_interest": [123, 124, 125], // or "all_active", "by_breed:Merino"
        "fixed_effects_model": {"sex": true, "birth_year_season": true},
        "evaluation_description": "Annual Weaning Weight Evaluation 2024"
    }
    ```
*   **Success Response:**
    *   Status: `202 Accepted`
    *   Body:
        ```json
        {
            "job_id": "blup_job_98765",
            "status_url": "/genetic_evaluations/status/blup_job_98765",
            "message": "BLUP evaluation job accepted."
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Invalid parameters for BLUP evaluation."}`
    *   `500 Internal Server Error`: `{"error": "Failed to initiate BLUP evaluation."}`

### 5.3 (Conceptual) Get relationship matrix
*   **Endpoint Path:** `/relationships/matrix`
*   **HTTP Method:** `GET` (or `POST` if requesting for a large specific list of animals)
*   **Description:** Retrieves the additive relationship matrix (A) or a part of it. This could be computationally intensive and might require careful parameterization.
*   **Query Parameters (Example):**
    *   `animal_ids` (comma-separated string): "123,124,125" - for a subset of the matrix.
    *   `format` (string, default "json_sparse"): "json_full", "csv".
*   **Success Response:**
    *   Status: `200 OK`
    *   Body: (Example for json_sparse, actual A matrix is very large)
        ```json
        {
            "animal_to_index_map": {"123": 0, "124": 1, "125": 2},
            "matrix_sparse": { // Example: ((row, col), value)
                "0,0": 1.05, "0,1": 0.125, "1,0": 0.125, "1,1": 1.02, ...
            },
            "matrix_type": "additive_relationship"
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Invalid animal_ids list or format."}`
    *   `500 Internal Server Error`: `{"error": "Failed to retrieve relationship matrix."}`

---

## 6. Mating Plans

Endpoints for assisting with mating plans.

### 6.1 Request mate suggestions
*   **Endpoint Path:** `/mating_plans/suggest_mates`
*   **HTTP Method:** `POST`
*   **Description:** Provides mate suggestions based on genetic merit and inbreeding constraints.
*   **Request Body:**
    ```json
    {
        "target_animals_ids": [1, 2], // e.g., Ewe IDs
        "potential_mates_ids": [11, 12, 13], // e.g., Ram IDs
        "selection_criteria": { // TraitID: weight
            "101": 0.6, // Weaning Weight
            "102": 0.4  // Fleece Weight
        },
        "max_inbreeding_threshold": 0.0625 // e.g., 6.25%
    }
    ```
*   **Success Response:**
    *   Status: `200 OK`
    *   Body:
        ```json
        {
            "1": [ // Target Animal ID 1
                {
                    "mate_id": 11, // Ram ID
                    "expected_inbreeding": 0.03125,
                    "expected_progeny_ebvs": {"101": 0.85, "102": 0.25},
                    "selection_index": 0.61
                }
                // ... other ranked mates for animal 1
            ],
            "2": [ // Target Animal ID 2
                // ... ranked mates for animal 2
            ]
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Missing required fields or invalid selection criteria."}`
    *   `404 Not Found`: `{"error": "One or more animal IDs not found or lack EBV data."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred during mate suggestion."}`

### 6.2 Calculate expected progeny values
*   **Endpoint Path:** `/mating_plans/calculate_expected_progeny`
*   **HTTP Method:** `POST`
*   **Description:** Calculates expected progeny EBVs and inbreeding for a specific sire/dam pair.
*   **Request Body:**
    ```json
    {
        "sire_id": 11,
        "dam_id": 1,
        "traits_of_interest": [101, 102] // TraitIDs
    }
    ```
*   **Success Response:**
    *   Status: `200 OK`
    *   Body:
        ```json
        {
            "sire_id": 11,
            "dam_id": 1,
            "expected_inbreeding": 0.03125,
            "expected_progeny_ebvs": {
                "101": { // Weaning Weight
                    "sire_ebv": 1.0,
                    "dam_ebv": 0.5,
                    "progeny_ebv": 0.75
                },
                "102": { // Fleece Weight
                    "sire_ebv": 0.4,
                    "dam_ebv": 0.2,
                    "progeny_ebv": 0.30
                }
            }
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Missing sire_id or dam_id."}`
    *   `404 Not Found`: `{"error": "SireID, DamID, or TraitID not found, or EBVs/relationship data missing."}`
    *   `500 Internal Server Error`: `{"error": "An unexpected error occurred."}`

---

## 7. NLP Interface

Endpoint for interacting with the system using natural language.

### 7.1 Submit a natural language query
*   **Endpoint Path:** `/nlp_query`
*   **HTTP Method:** `POST`
*   **Description:** Processes a natural language query, attempts to understand intent and entities, and returns a natural language response and optionally structured data.
*   **Request Body:**
    ```json
    {
        "query": "Suggest mates for ewe D025, prioritize growth rate and limit inbreeding to 3%."
    }
    ```
*   **Success Response:**
    *   Status: `200 OK`
    *   Body:
        ```json
        {
            "natural_language_response": "For ewe D025, based on 'growth rate' and max inbreeding 3.0%, potential suggestions are: Ram S010 (EBV Index: +1.5, Inbreeding: 2.5%), Ram S012 (EBV Index: +1.2, Inbreeding: 1.0%).",
            "identified_intent": "SUGGEST_MATES",
            "extracted_entities": {
                "target_animals_ids": ["D025"],
                "selection_criteria_raw": "growth rate",
                "max_inbreeding_threshold": "3%"
            },
            "structured_data": { // Optional: Could be the same as suggest_mates response
                "D025": [
                    {"mate_id": "S010", "ebv_index": 1.5, "inbreeding": 0.025},
                    {"mate_id": "S012", "ebv_index": 1.2, "inbreeding": 0.010}
                ]
            },
            "conceptual_action_taken": "mating_planner.suggest_mates(...)"
        }
        ```
*   **Error Responses:**
    *   `400 Bad Request`: `{"error": "Query string is missing or empty."}`
    *   `500 Internal Server Error`: `{"error": "Error processing NLP query."}`
    *   `200 OK` with a response indicating misunderstanding:
        ```json
        {
            "natural_language_response": "I'm sorry, I could not understand your request regarding '...'. Could you please rephrase?",
            "identified_intent": "UNKNOWN",
            "extracted_entities": {},
            "structured_data": null
        }
        ```
---
```
