"""
Data Management functions for the Sheep Breeding Management System.

This module provides functions to interact with the database for managing
animals, phenotypic records, pedigrees, and traits.

Placeholder for database connection:
The functions in this module expect a database connection object (`db_connection`)
as their first argument. This object should have methods like `cursor()` and
`commit()` typical of DB-API 2.0 compliant connectors (e.g., sqlite3, psycopg2).

SQL queries are defined as strings. Actual execution and error handling
depend on the specific database connector used.
"""

# Placeholder for a potential custom exception class
class DataError(Exception):
    """Base class for exceptions in this module."""
    pass

class RecordNotFoundError(DataError):
    """Raised when a record is not found."""
    pass

class DuplicateRecordError(DataError):
    """Raised when attempting to insert a duplicate record."""
    pass


# --- Animal Management ---

def add_animal(db_connection, animal_data: dict):
    """
    Inserts a new animal into the Animals table.
     conceptually adding IsActive BOOLEAN DEFAULT TRUE
    Args:
        db_connection: Database connection object.
        animal_data: A dictionary containing animal information.
                     Expected keys: Eartag, Sex, BirthDate, Breed,
                                    BirthWeight (optional), WeaningWeight (optional),
                                    PurchaseDate (optional), SaleDate (optional),
                                    DeathDate (optional), CurrentOwnerID (optional),
                                    Notes (optional).
                                    An 'IsActive' field with default TRUE is assumed.
    Returns:
        The AnimalID of the newly added animal, or None if insertion fails.
    """
    # Assuming IsActive column is added to Animals table as per requirements
    # `IsActive` defaults to TRUE at DB level or explicitly set here.
    sql = """
        INSERT INTO Animals (Eartag, Sex, BirthDate, Breed, BirthWeight, WeaningWeight,
                             PurchaseDate, SaleDate, DeathDate, CurrentOwnerID, Notes, IsActive)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE)
    """
    # For PostgreSQL, use %s instead of ?
    # sql_pg = """
    #     INSERT INTO Animals (Eartag, Sex, BirthDate, Breed, BirthWeight, WeaningWeight,
    #                          PurchaseDate, SaleDate, DeathDate, CurrentOwnerID, Notes, IsActive)
    #     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
    # """
    try:
        cursor = db_connection.cursor()
        # The order of values in the tuple must match the order of columns in the SQL statement
        # and the expected keys in animal_data.
        cursor.execute(sql, (
            animal_data['Eartag'],
            animal_data['Sex'],
            animal_data['BirthDate'],
            animal_data.get('Breed'),
            animal_data.get('BirthWeight'),
            animal_data.get('WeaningWeight'),
            animal_data.get('PurchaseDate'),
            animal_data.get('SaleDate'),
            animal_data.get('DeathDate'),
            animal_data.get('CurrentOwnerID'),
            animal_data.get('Notes')
            # IsActive is set to TRUE by default in the query
        ))
        db_connection.commit()
        return cursor.lastrowid  # For SQLite. For PostgreSQL, use RETURNING AnimalID
    except Exception as e:
        # Placeholder for actual error logging and handling
        print(f"Error adding animal: {e}")
        # db_connection.rollback() # Rollback in case of error
        return None

def get_animal(db_connection, animal_id: int):
    """
    Retrieves an animal's details from the Animals table by AnimalID.
    Also retrieves the IsActive status.
    Args:
        db_connection: Database connection object.
        animal_id: The ID of the animal to retrieve.
    Returns:
        A dictionary representing the animal's details, or None if not found.
    """
    # Assuming IsActive column exists
    sql = "SELECT AnimalID, Eartag, Sex, BirthDate, Breed, BirthWeight, WeaningWeight, PurchaseDate, SaleDate, DeathDate, CurrentOwnerID, Notes, IsActive FROM Animals WHERE AnimalID = ?"
    # sql_pg = "SELECT AnimalID, Eartag, Sex, BirthDate, Breed, BirthWeight, WeaningWeight, PurchaseDate, SaleDate, DeathDate, CurrentOwnerID, Notes, IsActive FROM Animals WHERE AnimalID = %s"
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql, (animal_id,))
        row = cursor.fetchone()
        if row:
            # Convert row to dictionary
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    except Exception as e:
        print(f"Error getting animal: {e}")
        return None

def update_animal(db_connection, animal_id: int, update_data: dict):
    """
    Updates an animal's information in the Animals table.
    Args:
        db_connection: Database connection object.
        animal_id: The ID of the animal to update.
        update_data: A dictionary containing the fields to update and their new values.
                     Allowed keys: Eartag, Sex, BirthDate, Breed, BirthWeight,
                                   WeaningWeight, PurchaseDate, SaleDate, DeathDate,
                                   CurrentOwnerID, Notes, IsActive.
    Returns:
        True if update was successful, False otherwise.
    """
    if not update_data:
        return False

    set_clauses = []
    values = []
    for key, value in update_data.items():
        # Ensure only valid columns are updated to prevent SQL injection if keys are directly from user input
        # (though here keys are controlled by the application)
        valid_columns = ["Eartag", "Sex", "BirthDate", "Breed", "BirthWeight",
                         "WeaningWeight", "PurchaseDate", "SaleDate", "DeathDate",
                         "CurrentOwnerID", "Notes", "IsActive"]
        if key in valid_columns:
            set_clauses.append(f"{key} = ?") # For PostgreSQL: {key} = %s
            values.append(value)
        else:
            print(f"Warning: Invalid column '{key}' in update_data ignored.")


    if not set_clauses:
        print("No valid fields to update.")
        return False

    sql = f"UPDATE Animals SET {', '.join(set_clauses)} WHERE AnimalID = ?"
    # sql_pg = f"UPDATE Animals SET {', '.join(set_clauses)} WHERE AnimalID = %s"
    values.append(animal_id)

    try:
        cursor = db_connection.cursor()
        cursor.execute(sql, tuple(values))
        db_connection.commit()
        return cursor.rowcount > 0 # Checks if any row was affected
    except Exception as e:
        print(f"Error updating animal: {e}")
        # db_connection.rollback()
        return False

def deactivate_animal(db_connection, animal_id: int):
    """
    Marks an animal as inactive by setting the IsActive flag to FALSE.
    This is a soft delete. Assumes 'IsActive' column exists in Animals table.
    Args:
        db_connection: Database connection object.
        animal_id: The ID of the animal to deactivate.
    Returns:
        True if deactivation was successful, False otherwise.
    """
    sql = "UPDATE Animals SET IsActive = FALSE WHERE AnimalID = ?"
    # sql_pg = "UPDATE Animals SET IsActive = FALSE WHERE AnimalID = %s"
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql, (animal_id,))
        db_connection.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error deactivating animal: {e}")
        # db_connection.rollback()
        return False

# --- Phenotypic Data Management ---

def add_phenotypic_record(db_connection, record_data: dict):
    """
    Adds a new record to the PhenotypicRecords table.
    Args:
        db_connection: Database connection object.
        record_data: A dictionary containing phenotypic record information.
                     Expected keys: AnimalID, TraitID, MeasurementDate, Value,
                                    RecordedByUserID (optional), Notes (optional).
    Returns:
        The RecordID of the newly added record, or None if insertion fails.
    """
    sql = """
        INSERT INTO PhenotypicRecords (AnimalID, TraitID, MeasurementDate, Value, RecordedByUserID, Notes)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    # sql_pg = """
    #     INSERT INTO PhenotypicRecords (AnimalID, TraitID, MeasurementDate, Value, RecordedByUserID, Notes)
    #     VALUES (%s, %s, %s, %s, %s, %s)
    # """
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql, (
            record_data['AnimalID'],
            record_data['TraitID'],
            record_data['MeasurementDate'],
            record_data['Value'],
            record_data.get('RecordedByUserID'),
            record_data.get('Notes')
        ))
        db_connection.commit()
        return cursor.lastrowid # For PostgreSQL, use RETURNING RecordID
    except Exception as e:
        print(f"Error adding phenotypic record: {e}")
        # db_connection.rollback()
        return None

def get_phenotypic_records_for_animal(db_connection, animal_id: int):
    """
    Retrieves all phenotypic records for a given animal.
    Args:
        db_connection: Database connection object.
        animal_id: The ID of the animal.
    Returns:
        A list of dictionaries, where each dictionary represents a phenotypic record.
        Returns an empty list if no records are found or in case of error.
    """
    sql = "SELECT RecordID, AnimalID, TraitID, MeasurementDate, Value, RecordedByUserID, Notes FROM PhenotypicRecords WHERE AnimalID = ?"
    # sql_pg = "SELECT RecordID, AnimalID, TraitID, MeasurementDate, Value, RecordedByUserID, Notes FROM PhenotypicRecords WHERE AnimalID = %s"
    records = []
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql, (animal_id,))
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            records.append(dict(zip(columns, row)))
        return records
    except Exception as e:
        print(f"Error getting phenotypic records for animal: {e}")
        return []

def get_phenotypic_records_for_trait(db_connection, trait_id: int):
    """
    Retrieves all phenotypic records for a given trait.
    Args:
        db_connection: Database connection object.
        trait_id: The ID of the trait.
    Returns:
        A list of dictionaries, where each dictionary represents a phenotypic record.
        Returns an empty list if no records are found or in case of error.
    """
    sql = "SELECT RecordID, AnimalID, TraitID, MeasurementDate, Value, RecordedByUserID, Notes FROM PhenotypicRecords WHERE TraitID = ?"
    # sql_pg = "SELECT RecordID, AnimalID, TraitID, MeasurementDate, Value, RecordedByUserID, Notes FROM PhenotypicRecords WHERE TraitID = %s"
    records = []
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql, (trait_id,))
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            records.append(dict(zip(columns, row)))
        return records
    except Exception as e:
        print(f"Error getting phenotypic records for trait: {e}")
        return []

# --- Pedigree Management ---

def add_pedigree_link(db_connection, animal_id: int, sire_id: int = None, dam_id: int = None, notes: str = None):
    """
    Adds or updates sire and dam information in the Pedigrees table.
    It attempts an UPSERT operation: inserts if AnimalID doesn't exist, updates if it does.
    Args:
        db_connection: Database connection object.
        animal_id: The ID of the animal.
        sire_id: The ID of the sire (optional).
        dam_id: The ID of the dam (optional).
        notes: Optional notes for the pedigree entry.
    Returns:
        True if the operation was successful, False otherwise.
    """
    # Basic Validation (conceptual - actual checks would involve more queries)
    # 1. Check if animal_id exists in Animals table.
    #    (SELECT 1 FROM Animals WHERE AnimalID = ?)
    # 2. If sire_id is provided, check if it exists in Animals table.
    #    (SELECT 1 FROM Animals WHERE AnimalID = ?)
    # 3. If dam_id is provided, check if it exists in Animals table.
    #    (SELECT 1 FROM Animals WHERE AnimalID = ?)
    # Optional Advanced Validation:
    # - Check if sire_id animal is Male.
    # - Check if dam_id animal is Female.
    # These checks would typically be done before calling this function or within a more complex service layer.

    # Using INSERT OR REPLACE for SQLite (UPSERT)
    # For PostgreSQL, use INSERT ... ON CONFLICT (AnimalID) DO UPDATE SET ...
    sql = """
        INSERT OR REPLACE INTO Pedigrees (AnimalID, SireID, DamID, Notes)
        VALUES (?, ?, ?, ?)
    """
    # sql_pg_upsert = """
    #     INSERT INTO Pedigrees (AnimalID, SireID, DamID, Notes)
    #     VALUES (%s, %s, %s, %s)
    #     ON CONFLICT (AnimalID) DO UPDATE
    #     SET SireID = EXCLUDED.SireID,
    #         DamID = EXCLUDED.DamID,
    #         Notes = EXCLUDED.Notes;
    # """
    try:
        cursor = db_connection.cursor()
        # Perform pre-checks if necessary (e.g., ensure sire_id and dam_id are valid AnimalIDs)
        # For example:
        # if sire_id and not get_animal(db_connection, sire_id):
        #     raise DataError(f"Sire with ID {sire_id} not found.")
        # if dam_id and not get_animal(db_connection, dam_id):
        #     raise DataError(f"Dam with ID {dam_id} not found.")

        cursor.execute(sql, (animal_id, sire_id, dam_id, notes))
        db_connection.commit()
        return True
    except Exception as e: # Replace with more specific exceptions like sqlite3.IntegrityError
        print(f"Error adding or updating pedigree link: {e}")
        # db_connection.rollback()
        return False

def get_pedigree(db_connection, animal_id: int):
    """
    Retrieves the sire and dam for a given animal.
    Args:
        db_connection: Database connection object.
        animal_id: The ID of the animal.
    Returns:
        A dictionary with 'SireID' and 'DamID', or None if not found.
    """
    sql = "SELECT AnimalID, SireID, DamID, Notes FROM Pedigrees WHERE AnimalID = ?"
    # sql_pg = "SELECT AnimalID, SireID, DamID, Notes FROM Pedigrees WHERE AnimalID = %s"
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql, (animal_id,))
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    except Exception as e:
        print(f"Error getting pedigree: {e}")
        return None

def get_offspring(db_connection, parent_id: int):
    """
    Retrieves all offspring for a given parent (either sire or dam).
    Args:
        db_connection: Database connection object.
        parent_id: The ID of the parent animal.
    Returns:
        A list of dictionaries, where each dictionary is an offspring's AnimalID.
        Returns an empty list if no offspring are found or in case of error.
    """
    # This query assumes we want basic info of the offspring (AnimalID and its Sex for clarity)
    # It joins Pedigrees with Animals to fetch offspring details.
    sql = """
        SELECT A.AnimalID, A.Eartag, A.Sex, A.BirthDate
        FROM Pedigrees P
        JOIN Animals A ON P.AnimalID = A.AnimalID
        WHERE P.SireID = ? OR P.DamID = ?
        AND A.IsActive = TRUE  -- Assuming we only want active offspring
    """
    # sql_pg = """
    #     SELECT A.AnimalID, A.Eartag, A.Sex, A.BirthDate
    #     FROM Pedigrees P
    #     JOIN Animals A ON P.AnimalID = A.AnimalID
    #     WHERE P.SireID = %s OR P.DamID = %s
    #     AND A.IsActive = TRUE
    # """
    offspring = []
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql, (parent_id, parent_id))
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            offspring.append(dict(zip(columns, row)))
        return offspring
    except Exception as e:
        print(f"Error getting offspring: {e}")
        return []

# --- Trait Management ---

def add_trait(db_connection, trait_data: dict):
    """
    Adds a new trait to the Traits table.
    Args:
        db_connection: Database connection object.
        trait_data: A dictionary containing trait information.
                    Expected keys: TraitName, UnitOfMeasure (optional),
                                   Description (optional), Category (optional).
    Returns:
        The TraitID of the newly added trait, or None if insertion fails.
    """
    sql = """
        INSERT INTO Traits (TraitName, UnitOfMeasure, Description, Category)
        VALUES (?, ?, ?, ?)
    """
    # sql_pg = """
    #     INSERT INTO Traits (TraitName, UnitOfMeasure, Description, Category)
    #     VALUES (%s, %s, %s, %s)
    # """
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql, (
            trait_data['TraitName'],
            trait_data.get('UnitOfMeasure'),
            trait_data.get('Description'),
            trait_data.get('Category')
        ))
        db_connection.commit()
        return cursor.lastrowid # For PostgreSQL, use RETURNING TraitID
    except Exception as e: # Be more specific, e.g., sqlite3.IntegrityError for UNIQUE constraint
        print(f"Error adding trait: {e}")
        # db_connection.rollback()
        return None

def get_trait(db_connection, trait_id: int):
    """
    Retrieves trait information by TraitID.
    Args:
        db_connection: Database connection object.
        trait_id: The ID of the trait to retrieve.
    Returns:
        A dictionary representing the trait, or None if not found.
    """
    sql = "SELECT TraitID, TraitName, UnitOfMeasure, Description, Category FROM Traits WHERE TraitID = ?"
    # sql_pg = "SELECT TraitID, TraitName, UnitOfMeasure, Description, Category FROM Traits WHERE TraitID = %s"
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql, (trait_id,))
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None
    except Exception as e:
        print(f"Error getting trait: {e}")
        return None

def get_all_traits(db_connection):
    """
    Retrieves all traits from the Traits table.
    Args:
        db_connection: Database connection object.
    Returns:
        A list of dictionaries, where each dictionary represents a trait.
        Returns an empty list if no traits are found or in case of error.
    """
    sql = "SELECT TraitID, TraitName, UnitOfMeasure, Description, Category FROM Traits"
    traits = []
    try:
        cursor = db_connection.cursor()
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        for row in cursor.fetchall():
            traits.append(dict(zip(columns, row)))
        return traits
    except Exception as e:
        print(f"Error getting all traits: {e}")
        return []

if __name__ == '__main__':
    # This is a placeholder for example usage or testing.
    # It requires a live database connection (e.g., SQLite in-memory) to run.
    print("data_manager.py loaded. Contains functions for data management.")
    print("To test these functions, you would need to set up a database connection.")

    # Example (conceptual - would need sqlite3 and schema setup):
    # import sqlite3
    # conn = sqlite3.connect(':memory:') # In-memory database for testing

    # # You would need to execute SQL to create tables based on database_schema.md first
    # # For example:
    # # cursor = conn.cursor()
    # # cursor.execute('''
    # # CREATE TABLE Animals (
    # #     AnimalID INTEGER PRIMARY KEY AUTOINCREMENT, Eartag TEXT UNIQUE NOT NULL,
    # #     Sex TEXT NOT NULL, BirthDate DATE NOT NULL, Breed TEXT, BirthWeight REAL,
    # #     WeaningWeight REAL, PurchaseDate DATE, SaleDate DATE, DeathDate DATE,
    # #     CurrentOwnerID INTEGER, Notes TEXT, IsActive BOOLEAN DEFAULT TRUE,
    # #     FOREIGN KEY (CurrentOwnerID) REFERENCES Users(UserID)
    # # );
    # # ''')
    # # cursor.execute('''
    # # CREATE TABLE Traits (
    # #    TraitID INTEGER PRIMARY KEY AUTOINCREMENT, TraitName TEXT UNIQUE NOT NULL,
    # #    UnitOfMeasure TEXT, Description TEXT, Category TEXT
    # # );
    # # ''')
    # # cursor.execute('''
    # # CREATE TABLE PhenotypicRecords (
    # #     RecordID INTEGER PRIMARY KEY AUTOINCREMENT, AnimalID INTEGER NOT NULL,
    # #     TraitID INTEGER NOT NULL, MeasurementDate DATE NOT NULL, Value REAL NOT NULL,
    # #     RecordedByUserID INTEGER, Notes TEXT,
    # #     FOREIGN KEY (AnimalID) REFERENCES Animals(AnimalID),
    # #     FOREIGN KEY (TraitID) REFERENCES Traits(TraitID),
    # #     FOREIGN KEY (RecordedByUserID) REFERENCES Users(UserID)
    # # );
    # # ''')
    # # cursor.execute('''
    # # CREATE TABLE Pedigrees (
    # #     AnimalID INTEGER PRIMARY KEY, SireID INTEGER, DamID INTEGER, Notes TEXT,
    # #     FOREIGN KEY (AnimalID) REFERENCES Animals(AnimalID),
    # #     FOREIGN KEY (SireID) REFERENCES Animals(AnimalID),
    # #     FOREIGN KEY (DamID) REFERENCES Animals(AnimalID)
    # # );
    # # ''')
    # # conn.commit()


    # # --- Test Animal Management ---
    # animal1_data = {'Eartag': 'ET001', 'Sex': 'Male', 'BirthDate': '2023-01-15', 'Breed': 'Merino'}
    # animal1_id = add_animal(conn, animal1_data)
    # print(f"Added animal 1 with ID: {animal1_id}")

    # if animal1_id:
    #     retrieved_animal1 = get_animal(conn, animal1_id)
    #     print(f"Retrieved animal 1: {retrieved_animal1}")
    #
    #     update_success = update_animal(conn, animal1_id, {'Notes': 'Champion Ram Prospect', 'BirthWeight': 5.2})
    #     print(f"Animal 1 update success: {update_success}")
    #     retrieved_animal1_updated = get_animal(conn, animal1_id)
    #     print(f"Retrieved animal 1 (updated): {retrieved_animal1_updated}")

    # animal2_data = {'Eartag': 'ET002', 'Sex': 'Female', 'BirthDate': '2023-02-20', 'Breed': 'Dorper'}
    # animal2_id = add_animal(conn, animal2_data)
    # print(f"Added animal 2 with ID: {animal2_id}")

    # if animal1_id:
    #    deactivate_success = deactivate_animal(conn, animal1_id)
    #    print(f"Animal 1 deactivate success: {deactivate_success}")
    #    retrieved_animal1_deactivated = get_animal(conn, animal1_id)
    #    print(f"Retrieved animal 1 (deactivated): {retrieved_animal1_deactivated}")


    # # --- Test Trait Management ---
    # trait1_data = {'TraitName': 'Weaning Weight', 'UnitOfMeasure': 'kg', 'Category': 'Growth'}
    # trait1_id = add_trait(conn, trait1_data)
    # print(f"Added trait 1 with ID: {trait1_id}")
    #
    # trait2_data = {'TraitName': 'Fleece Diameter', 'UnitOfMeasure': 'microns', 'Category': 'Wool'}
    # trait2_id = add_trait(conn, trait2_data)
    # print(f"Added trait 2 with ID: {trait2_id}")

    # if trait1_id:
    #     retrieved_trait1 = get_trait(conn, trait1_id)
    #     print(f"Retrieved trait 1: {retrieved_trait1}")
    #
    # all_traits = get_all_traits(conn)
    # print(f"All traits: {all_traits}")


    # # --- Test Phenotypic Data Management ---
    # if animal2_id and trait1_id:
    #     record1_data = {'AnimalID': animal2_id, 'TraitID': trait1_id, 'MeasurementDate': '2023-05-20', 'Value': 25.5}
    #     record1_id = add_phenotypic_record(conn, record1_data)
    #     print(f"Added phenotypic record 1 with ID: {record1_id}")
    #
    #     animal_records = get_phenotypic_records_for_animal(conn, animal2_id)
    #     print(f"Phenotypic records for animal {animal2_id}: {animal_records}")
    #
    # if trait1_id:
    #     trait_records = get_phenotypic_records_for_trait(conn, trait1_id)
    #     print(f"Phenotypic records for trait {trait1_id}: {trait_records}")


    # # --- Test Pedigree Management ---
    # # Assume animal1_id is sire, animal2_id is dam (need to ensure they exist and animal1 is active)
    # # For this test, let's re-activate animal1 if it was deactivated
    # # if animal1_id: update_animal(conn, animal1_id, {'IsActive': True})
    #
    # animal3_data = {'Eartag': 'ET003', 'Sex': 'Male', 'BirthDate': '2024-01-01', 'Breed': 'Cross'}
    # animal3_id = add_animal(conn, animal3_data)
    # print(f"Added animal 3 with ID: {animal3_id}")
    #
    # if animal3_id and animal1_id and animal2_id:
    #    # Conceptual check:
    #    # sire_info = get_animal(conn, animal1_id)
    #    # dam_info = get_animal(conn, animal2_id)
    #    # if sire_info and sire_info['Sex'] == 'Male' and dam_info and dam_info['Sex'] == 'Female':
    #    pedigree_link_success = add_pedigree_link(conn, animal_id=animal3_id, sire_id=animal1_id, dam_id=animal2_id, notes="First cross")
    #    print(f"Pedigree link for animal {animal3_id} success: {pedigree_link_success}")
    #
    #    pedigree_info = get_pedigree(conn, animal3_id)
    #    print(f"Pedigree for animal {animal3_id}: {pedigree_info}")
    #
    # if animal1_id: # animal1 is a sire
    #     offspring_of_animal1 = get_offspring(conn, animal1_id)
    #     print(f"Offspring of animal {animal1_id}: {offspring_of_animal1}")
    #
    # if animal2_id: # animal2 is a dam
    #     offspring_of_animal2 = get_offspring(conn, animal2_id)
    #     print(f"Offspring of animal {animal2_id}: {offspring_of_animal2}")
    #
    # if conn:
    #     conn.close()
    pass
```
