# Sheep Breeding Management System - Database Schema

This document details the database schema for the Sheep Breeding Management System.

## Table of Contents
1.  [Animals](#animals)
2.  [Traits](#traits)
3.  [PhenotypicRecords](#phenotypicrecords)
4.  [Pedigrees](#pedigrees)
5.  [Markers](#markers)
6.  [GenomicData](#genomicdata)
7.  [MatingPlans](#matingplans)
8.  [Users](#users)
9.  [Relationships Summary](#relationships-summary)

---

## 1. Animals Table
Stores individual animal information.

| Column Name     | Data Type        | Constraints                                             | Description                                     |
|-----------------|------------------|---------------------------------------------------------|-------------------------------------------------|
| `AnimalID`      | `INTEGER`        | `PRIMARY KEY`, `NOT NULL`                               | Unique identifier for each animal.              |
| `Eartag`        | `VARCHAR(50)`    | `UNIQUE`, `NOT NULL`                                    | Unique ear tag number for the animal.           |
| `Sex`           | `VARCHAR(10)`    | `NOT NULL`, `CHECK (Sex IN ('Male', 'Female', 'Unknown'))` | Sex of the animal (Male, Female, Unknown).      |
| `BirthDate`     | `DATE`           | `NOT NULL`                                              | Date of birth of the animal.                    |
| `Breed`         | `VARCHAR(100)`   |                                                         | Breed of the animal.                            |
| `BirthWeight`   | `DECIMAL(5,2)`   |                                                         | Weight of the animal at birth (e.g., in kg).    |
| `WeaningWeight` | `DECIMAL(5,2)`   |                                                         | Weight of the animal at weaning (e.g., in kg).  |
| `PurchaseDate`  | `DATE`           |                                                         | Date the animal was purchased, if applicable.   |
| `SaleDate`      | `DATE`           |                                                         | Date the animal was sold, if applicable.        |
| `DeathDate`     | `DATE`           |                                                         | Date the animal died, if applicable.            |
| `CurrentOwnerID`| `INTEGER`        | `FOREIGN KEY REFERENCES Users(UserID)`                  | Foreign key referencing the current owner (User).|
| `Notes`         | `TEXT`           |                                                         | Any additional notes about the animal.          |

---

## 2. Traits Table
Stores information about the different traits being measured.

| Column Name     | Data Type        | Constraints                      | Description                                   |
|-----------------|------------------|----------------------------------|-----------------------------------------------|
| `TraitID`       | `INTEGER`        | `PRIMARY KEY`, `NOT NULL`        | Unique identifier for each trait.             |
| `TraitName`     | `VARCHAR(100)`   | `UNIQUE`, `NOT NULL`             | Name of the trait (e.g., 'Fleece Weight').    |
| `UnitOfMeasure` | `VARCHAR(50)`    |                                  | Unit of measurement for the trait (e.g., 'kg', 'cm'). |
| `Description`   | `TEXT`           |                                  | Detailed description of the trait.            |
| `Category`      | `VARCHAR(50)`    |                                  | Category of the trait (e.g., 'Growth', 'Wool', 'Reproduction'). |

---

## 3. PhenotypicRecords Table
Stores various phenotypic measurements for animals.

| Column Name       | Data Type      | Constraints                                               | Description                                      |
|-------------------|----------------|-----------------------------------------------------------|--------------------------------------------------|
| `RecordID`        | `INTEGER`      | `PRIMARY KEY`, `NOT NULL`                                 | Unique identifier for each phenotypic record.    |
| `AnimalID`        | `INTEGER`      | `NOT NULL`, `FOREIGN KEY REFERENCES Animals(AnimalID)`    | Foreign key referencing the animal measured.     |
| `TraitID`         | `INTEGER`      | `NOT NULL`, `FOREIGN KEY REFERENCES Traits(TraitID)`      | Foreign key referencing the trait measured.      |
| `MeasurementDate` | `DATE`         | `NOT NULL`                                                | Date the measurement was taken.                  |
| `Value`           | `DECIMAL(10,4)`| `NOT NULL`                                                | Value of the measurement.                        |
| `RecordedByUserID`| `INTEGER`      | `FOREIGN KEY REFERENCES Users(UserID)`                    | Foreign key referencing the user who recorded the data. |
| `Notes`           | `TEXT`         |                                                           | Any additional notes about the measurement.      |

---

## 4. Pedigrees Table
Stores sire and dam information for each animal.

| Column Name | Data Type | Constraints                                                                 | Description                                       |
|-------------|-----------|-----------------------------------------------------------------------------|---------------------------------------------------|
| `AnimalID`  | `INTEGER` | `PRIMARY KEY`, `NOT NULL`, `FOREIGN KEY REFERENCES Animals(AnimalID)`       | Foreign key referencing the animal.               |
| `SireID`    | `INTEGER` | `FOREIGN KEY REFERENCES Animals(AnimalID) ON DELETE SET NULL ON UPDATE CASCADE` | Foreign key referencing the animal's sire.        |
| `DamID`     | `INTEGER` | `FOREIGN KEY REFERENCES Animals(AnimalID) ON DELETE SET NULL ON UPDATE CASCADE` | Foreign key referencing the animal's dam.         |
| `Notes`     | `TEXT`    |                                                                             | Any additional notes about the pedigree record.   |

*Note: `ON DELETE SET NULL` for SireID and DamID allows an animal to remain in the system even if its parents are removed. `ON UPDATE CASCADE` ensures that if an AnimalID changes in the `Animals` table, it's updated here too (though changing primary keys is generally discouraged).*

---

## 5. Markers Table
Stores information about genomic markers.

| Column Name | Data Type      | Constraints               | Description                                     |
|-------------|----------------|---------------------------|-------------------------------------------------|
| `MarkerID`  | `INTEGER`      | `PRIMARY KEY`, `NOT NULL` | Unique identifier for each genomic marker.      |
| `MarkerName`| `VARCHAR(100)` | `UNIQUE`, `NOT NULL`      | Name or code of the marker (e.g., 'SNP12345').  |
| `Chromosome`| `VARCHAR(10)`  |                           | Chromosome on which the marker is located.      |
| `Position`  | `BIGINT`       |                           | Position of the marker on the chromosome.       |
| `ReferenceAllele` | `VARCHAR(10)` |                      | The reference allele for the marker.            |
| `AlternateAllele` | `VARCHAR(10)` |                      | The alternate allele for the marker.            |

---

## 6. GenomicData Table
Stores genomic information for animals, including genotypes and Genomic Estimated Breeding Values (GEBVs).

| Column Name    | Data Type      | Constraints                                                                                                | Description                                                      |
|----------------|----------------|------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| `GenomicDataID`| `INTEGER`      | `PRIMARY KEY`, `NOT NULL`                                                                                  | Unique identifier for each genomic data entry.                   |
| `AnimalID`     | `INTEGER`      | `NOT NULL`, `FOREIGN KEY REFERENCES Animals(AnimalID)`                                                     | Foreign key referencing the animal.                              |
| `MarkerID`     | `INTEGER`      | `FOREIGN KEY REFERENCES Markers(MarkerID)`                                                                 | Foreign key referencing the genomic marker (for genotype data).  |
| `Genotype`     | `VARCHAR(10)`  |                                                                                                            | Genotype of the animal for the specific marker (e.g., 'AA', 'AG', 'GG'). |
| `GEBV_TraitID` | `INTEGER`      | `FOREIGN KEY REFERENCES Traits(TraitID)`                                                                   | Foreign key referencing the trait for which the GEBV is calculated. |
| `GEBV_Value`   | `DECIMAL(10,4)`|                                                                                                            | Genomic Estimated Breeding Value for the specified trait.        |
| `RecordDate`   | `DATE`         | `NOT NULL`                                                                                                 | Date the genomic data was recorded or calculated.                |
| `Source`       | `VARCHAR(100)` |                                                                                                            | Source of the genomic data (e.g., 'LabX', 'CalculationY').       |

*Note: A single row in this table can either store a genotype for a specific marker OR a GEBV for a specific trait. If both MarkerID and GEBV_TraitID are present, it implies the GEBV is specific to that marker, which is less common. Typically, one would be NULL. A `CHECK` constraint could enforce this logic if needed, or the table could be split into `Genotypes` and `GEBVs` for stricter normalization.*
*A composite primary key like `(AnimalID, MarkerID)` for genotypes or `(AnimalID, GEBV_TraitID)` for GEBVs could be used if `GenomicDataID` is omitted and the table is split.*

---

## 7. MatingPlans Table
Stores information about planned matings.

| Column Name   | Data Type      | Constraints                                            | Description                                     |
|---------------|----------------|--------------------------------------------------------|-------------------------------------------------|
| `PlanID`      | `INTEGER`      | `PRIMARY KEY`, `NOT NULL`                              | Unique identifier for each mating plan.         |
| `SireID`      | `INTEGER`      | `NOT NULL`, `FOREIGN KEY REFERENCES Animals(AnimalID)` | Foreign key referencing the planned sire.       |
| `DamID`       | `INTEGER`      | `NOT NULL`, `FOREIGN KEY REFERENCES Animals(AnimalID)` | Foreign key referencing the planned dam.        |
| `PlannedDate` | `DATE`         |                                                        | Planned date for the mating.                    |
| `Objective`   | `TEXT`         |                                                        | Objective or reason for this mating plan.       |
| `Status`      | `VARCHAR(50)`  | `DEFAULT 'Planned'`                                    | Status of the plan (e.g., 'Planned', 'Completed', 'Cancelled'). |
| `Notes`       | `TEXT`         |                                                        | Any additional notes about the mating plan.     |

---

## 8. Users Table
Stores basic user information for system access and ownership.

| Column Name    | Data Type        | Constraints                                  | Description                                      |
|----------------|------------------|----------------------------------------------|--------------------------------------------------|
| `UserID`       | `INTEGER`        | `PRIMARY KEY`, `NOT NULL`                    | Unique identifier for each user.                 |
| `Username`     | `VARCHAR(50)`    | `UNIQUE`, `NOT NULL`                         | Unique username for login.                       |
| `HashedPassword`| `VARCHAR(255)`  | `NOT NULL`                                   | Hashed password for security.                    |
| `FullName`     | `VARCHAR(100)`   |                                              | Full name of the user.                           |
| `Email`        | `VARCHAR(100)`   | `UNIQUE`                                     | Email address of the user.                       |
| `Role`         | `VARCHAR(50)`    | `NOT NULL`, `DEFAULT 'User'`                 | Role of the user (e.g., 'Admin', 'Manager', 'User'). |
| `RegistrationDate`| `TIMESTAMP`   | `DEFAULT CURRENT_TIMESTAMP`                  | Date and time of user registration.              |
| `LastLoginDate`| `TIMESTAMP`      |                                              | Date and time of the user's last login.          |

---

## 9. Relationships Summary

*   **Animals to PhenotypicRecords**: One-to-Many (One animal can have many phenotypic records).
    *   `PhenotypicRecords.AnimalID` -> `Animals.AnimalID`
*   **Traits to PhenotypicRecords**: One-to-Many (One trait can be measured in many phenotypic records).
    *   `PhenotypicRecords.TraitID` -> `Traits.TraitID`
*   **Animals to Pedigrees**: One-to-One (Each animal has one pedigree entry for its parents).
    *   `Pedigrees.AnimalID` -> `Animals.AnimalID`
    *   `Pedigrees.SireID` -> `Animals.AnimalID`
    *   `Pedigrees.DamID` -> `Animals.AnimalID`
*   **Animals to GenomicData**: One-to-Many (One animal can have multiple genomic data entries - e.g., genotypes for many markers, or GEBVs for many traits).
    *   `GenomicData.AnimalID` -> `Animals.AnimalID`
*   **Markers to GenomicData**: One-to-Many (One marker can be part of many genomic data entries, typically for genotypes across different animals).
    *   `GenomicData.MarkerID` -> `Markers.MarkerID`
*   **Traits to GenomicData**: One-to-Many (One trait can have GEBVs recorded for many animals).
    *   `GenomicData.GEBV_TraitID` -> `Traits.TraitID`
*   **Animals to MatingPlans**: Many-to-Many (An animal can be a sire in multiple plans and a dam in multiple plans).
    *   `MatingPlans.SireID` -> `Animals.AnimalID`
    *   `MatingPlans.DamID` -> `Animals.AnimalID`
*   **Users to Animals**: One-to-Many (One user can be the owner of many animals).
    *   `Animals.CurrentOwnerID` -> `Users.UserID`
*   **Users to PhenotypicRecords**: One-to-Many (One user can record many phenotypic records).
    *   `PhenotypicRecords.RecordedByUserID` -> `Users.UserID`

This schema provides a comprehensive structure for managing sheep breeding data, encompassing animal details, phenotypic traits, pedigree, genomics, mating plans, and user management.
Further refinements, such as specific `CHECK` constraints or additional audit logging tables, could be added based on more granular system requirements.
For instance, ensuring `SireID` in `MatingPlans` refers to a 'Male' animal and `DamID` to a 'Female' animal would require application-level logic or more complex database constraints (e.g., triggers or referencing specific views).
The `GenomicData` table is designed to be flexible. If GEBVs are always calculated for a standard set of traits and not tied to individual markers, and genotypes are always for specific markers, splitting this table into `AnimalGenotypes` and `AnimalGEBVs` might offer a cleaner design. However, the current combined table allows for potential future scenarios where a GEBV might be directly associated with a specific marker's influence.
```
