# Sheep Breeding Management System - UI Conceptualization

## Introduction

This document outlines the conceptual design of User Interface (UI) elements and views for the Sheep Breeding Management System. The goal is to provide a user-friendly interface that supports both structured data interaction and natural language commands for efficient flock management.

For each UI element or view, the following aspects are described:
*   **View/Component Name**
*   **Purpose**
*   **Key Information/Controls**
*   **Interaction with NLP (if relevant)**

---

## 1. Main Navigation Structure

*   **View/Component Name:** Main Navigation
*   **Purpose:** Allows users to easily switch between different sections and functionalities of the application.
*   **Key Information/Controls:**
    *   Typically a **persistent sidebar** or a **top navigation bar**.
    *   Navigation Links:
        *   Dashboard
        *   Animals (leading to Animal List View)
        *   Traits
        *   Genetic Evaluations (could be a dropdown for EBV/GEBV overview, BLUP runs)
        *   Mating Plans
        *   Reports
        *   System Settings / Admin (for admin users)
    *   User Profile/Logout button.
    *   Possibly a global search bar (could be integrated with or separate from NLP input).
*   **Interaction with NLP (if relevant):** While primary navigation is through clicks, NLP commands might implicitly "navigate" the user by presenting results that are typically found in a specific view (e.g., "show me animal L001" takes user context to that animal's details).

---

## 2. Dashboard

*   **View/Component Name:** Dashboard
*   **Purpose:** Provides a high-level, at-a-glance overview of the flock, key performance indicators, recent activities, and quick access points.
*   **Key Information/Controls:**
    *   **Key Flock Statistics (Widgets/Cards):**
        *   Total Active Animals
        *   Number of Sires / Dams
        *   Animals by Breed (simple chart or list)
        *   Upcoming births/weanings (if such data is tracked)
    *   **Trait Summary (Optional):** Average EBVs for key traits across the flock or specific groups.
    *   **Recent Activity Feed:** Log of recent additions, recordings, or system events (e.g., "Lamb L005 added," "Weaning weights for 20 lambs recorded").
    *   **Alerts/Notifications:** Important system messages, health alerts, or tasks needing attention.
    *   **Quick Access Buttons:** "Add New Animal," "Record Phenotype," "Suggest Mates."
    *   **Embedded NLP Input Bar:** Prominently displayed for immediate command input (see Section 3).
*   **Interaction with NLP (if relevant):** The dashboard is a primary location for the NLP input. Users might start here to issue commands. NLP queries like "show flock summary" would essentially present dashboard-like information.

---

## 3. NLP Interaction Area

*   **View/Component Name:** NLP Input Bar & Conversation Display
*   **Purpose:** The primary interface for users to interact with the system using natural language commands.
*   **Key Information/Controls:**
    *   **Text Input Field:** Large, clear input box for typing natural language queries.
        *   May include a microphone icon for voice input (future enhancement).
        *   Could offer auto-suggestions based on common commands or entity names as the user types.
    *   **Submit Button:** To send the query.
    *   **Conversation History Display:**
        *   Shows a chronological log of user queries and system responses (both text and potentially structured data previews).
        *   Allows users to review past interactions.
        *   System responses might include clickable elements if structured data is returned (e.g., a link to an animal's detail page).
    *   **(Optional) Quick Command Buttons/Chips:** Buttons for very common actions like "List active rams," "Help," etc., that pre-fill the NLP input.
*   **Interaction with NLP (if relevant):** This IS the NLP interaction point. Results from NLP queries might update other views or provide direct answers within the conversation display.

---

## 4. Animal Management Views

### 4.1 Animal List View

*   **View/Component Name:** Animal List View
*   **Purpose:** Allows users to browse, search, filter, and sort the list of animals in the system. Provides a gateway to individual animal details.
*   **Key Information/Controls:**
    *   **Animal Table:**
        *   Columns: AnimalID (or internal system ID), Eartag (prominent), Sex, Birth Date, Breed, Sire Eartag, Dam Eartag, IsActive status.
        *   Optional columns for key EBVs (e.g., Weaning Weight EBV, Fleece Weight EBV).
        *   Each row is clickable, leading to the Animal Detail View for that animal.
    *   **Search Bar:** Global search within the list (e.g., by Eartag, notes).
    *   **Filtering Controls:**
        *   Dropdowns or facets for: Breed, Sex, IsActive, Year of Birth.
        *   Advanced filter options (e.g., filter by animals with specific traits recorded, animals with/without EBVs).
    *   **Sorting Controls:** Clickable table headers to sort by any column.
    *   **Pagination Controls:** For navigating through large lists of animals.
    *   **"Add New Animal" Button:** Navigates to the Animal Add/Edit Form.
    *   **Bulk Action Controls (Optional):** Select multiple animals to perform actions like "Deactivate selected," "Add group phenotype."
*   **Interaction with NLP (if relevant):**
    *   NLP queries like "list all active Merino rams born in 2022" would populate this view with the corresponding filtered and sorted list.
    *   "Search for animal with eartag X007" would filter the list or directly navigate to the detail view if unique.

### 4.2 Animal Detail View

*   **View/Component Name:** Animal Detail View
*   **Purpose:** Provides a comprehensive view of all information related to a single animal.
*   **Key Information/Controls:**
    *   **Header Section:** Eartag, AnimalID, Sex, Breed, Current Status (Active/Inactive/Sold). Photo (if available).
    *   **Tabs or Sections for Different Information Categories:**
        *   **Basic Info:** Birth Date, Birth Weight, Weaning Weight, Purchase/Sale/Death Dates, Notes. Edit button leading to Animal Add/Edit Form.
        *   **Pedigree:**
            *   Display of Sire & Dam (with links to their detail views).
            *   Visual pedigree chart (e.g., 3-5 generations).
            *   List of known offspring (with links).
        *   **Phenotypic Records:** A sortable, filterable table/list of all recorded phenotypes for the animal (Trait Name, Value, Unit, Measurement Date, Notes). Button to "Add New Phenotype Record."
        *   **Breeding Values (EBVs/GEBVs):** A list or table of all available EBVs and GEBVs for different traits, including accuracy and evaluation date/ID. May include charts showing trends if historical EBVs are stored.
        *   **Genomic Data (if applicable):** Summary of genomic tests done, link to detailed marker data (if UI supports).
        *   **Mating History/Plans (if applicable):** Past matings, current mating plans involving this animal.
    *   **Action Buttons:** "Edit Animal," "Deactivate Animal," "Record Phenotype for this Animal," "Suggest Mates for this Animal" (if applicable based on sex).
*   **Interaction with NLP (if relevant):**
    *   Triggered by NLP queries like "show details for animal L001" or "what are the EBVs for S010?".
    *   NLP could be used to ask follow-up questions while on this page, e.g., "when was its last fleece weight recorded?"

### 4.3 Animal Add/Edit Form

*   **View/Component Name:** Animal Add/Edit Form
*   **Purpose:** Provides a structured interface for manual data entry or modification of animal details.
*   **Key Information/Controls:**
    *   **Form Fields for all attributes in the `Animals` table:**
        *   Eartag (validation for uniqueness if adding)
        *   Sex (dropdown)
        *   BirthDate (date picker)
        *   Breed (dropdown or text input with autocomplete)
        *   BirthWeight, WeaningWeight (numeric inputs)
        *   SireID, DamID (text input with autocomplete/search functionality to find existing animals, or option to leave blank for unknown)
        *   PurchaseDate, SaleDate, DeathDate (date pickers)
        *   CurrentOwnerID (dropdown, if users manage ownership)
        *   Notes (text area)
        *   IsActive (checkbox, usually for editing)
    *   **Save/Submit Button**
    *   **Cancel Button**
*   **Interaction with NLP (if relevant):**
    *   While NLP can be used for adding animals ("add lamb..."), this form serves as an alternative for detailed or bulk entry.
    *   If an NLP "add animal" command is incomplete, the system might present this form pre-filled with extracted entities, asking the user to complete or confirm.

---

## 5. Phenotypic Data Views

### 5.1 Phenotype Entry Form

*   **View/Component Name:** Phenotype Entry Form
*   **Purpose:** A structured form for adding one or more phenotypic records, often for a specific animal or a group of animals.
*   **Key Information/Controls:**
    *   **Animal Selector:** Input field to select AnimalID/Eartag (with autocomplete/search). May be pre-filled if accessed from Animal Detail View.
    *   **Trait Selector:** Dropdown list of defined traits.
    *   **Measurement Date:** Date picker (defaults to today).
    *   **Value:** Numeric input field. Unit of measure displayed based on selected trait.
    *   **RecordedByUserID:** (Often auto-filled with current user).
    *   **Notes:** Text area.
    *   **Save Button / Add Another Button:** To save the current record and either close or clear the form for the next entry.
    *   **(Optional) Group Entry Mode:** Select multiple animals first, then enter the same trait measurement for all (e.g., weaning date for a group).
*   **Interaction with NLP (if relevant):**
    *   Complements NLP for phenotype recording ("record weaning weight 25kg for L001 on 2023-07-20").
    *   Useful for correcting data entered via NLP or for bulk/detailed entry.

### 5.2 Phenotype Display Area (within Animal Detail View)

*   **View/Component Name:** Phenotype Display Area
*   **Purpose:** Lists phenotypic records associated with an animal, typically within the Animal Detail View.
*   **Key Information/Controls:**
    *   Table or list format.
    *   Columns: Trait Name, Measurement Date, Value, Unit, Recorded By, Notes.
    *   Sorting and filtering options for the list.
    *   Edit/Delete buttons for each record (with permissions).
*   **Interaction with NLP (if relevant):** NLP queries like "show L001's weaning weights" would filter this display or present the information directly.

---

## 6. Genetic Evaluation Views

### 6.1 EBV/GEBV Display

*   **View/Component Name:** EBV/GEBV Display Component
*   **Purpose:** Clearly presents Estimated Breeding Values (EBVs) or Genomic EBVs (GEBVs) to the user.
*   **Key Information/Controls:**
    *   **In Animal List View:** Selectable columns for key EBVs. Values displayed directly in the table. Color-coding or visual cues for high/low values might be used.
    *   **In Animal Detail View:** A dedicated section/tab.
        *   Table listing: Trait Name, EBV/GEBV Value, Accuracy (if available), Percentile Rank (optional), Date of Evaluation.
        *   Graphical display (optional): Bar charts comparing animal's EBVs to breed average, or trend lines if historical EBVs are available.
*   **Interaction with NLP (if relevant):** NLP queries like "what are S010's fleece EBVs?" or "compare EBVs for L001 and L002 for growth" would populate or highlight these display areas.

### 6.2 (Conceptual) BLUP Run Interface

*   **View/Component Name:** BLUP Run Configuration View
*   **Purpose:** Allows expert users to configure and initiate a BLUP genetic evaluation run.
*   **Key Information/Controls:**
    *   Form fields to select:
        *   Trait(s) for evaluation.
        *   Set of animals to include (e.g., all active, specific breeds, or based on other criteria).
        *   Fixed effects to include in the model.
        *   Date range for phenotypic data.
    *   "Run Evaluation" Button.
    *   Display area for status of past and ongoing evaluations (Job ID, Status, Completion Time, Link to results).
*   **Interaction with NLP (if relevant):** While complex runs might need a form, simpler NLP commands like "run BLUP for weaning weight for 2023 lambs" could trigger a run with default parameters or pre-fill this form.

---

## 7. Mating Plan Views

### 7.1 Mating Plan Setup/Criteria Input

*   **View/Component Name:** Mating Plan Configuration View
*   **Purpose:** Allows users to define parameters for generating mating suggestions.
*   **Key Information/Controls:**
    *   **Target Animal Selection:**
        *   Multi-select list or filterable table to choose target animals (e.g., a group of ewes).
    *   **Potential Mate Selection:**
        *   Multi-select list or filterable table to choose potential mates (e.g., available rams).
    *   **Selection Criteria Definition:**
        *   Interface to select multiple traits of interest.
        *   Input fields to assign weights or importance to each selected trait for creating a selection index.
        *   (Optional) Desired gains or thresholds for specific traits.
    *   **Constraints:**
        *   Input field for Maximum Allowable Inbreeding Coefficient (e.g., 3%, 5%).
        *   (Optional) Other constraints like maximum number of mates per sire.
    *   "Generate Suggestions" Button.
*   **Interaction with NLP (if relevant):** NLP commands like "suggest mates for my Merino ewes with top growth rams, limit inbreeding to 4%" would pre-fill or directly trigger the logic behind this view. The form allows for more granular control.

### 7.2 Suggested Matings Display

*   **View/Component Name:** Suggested Matings Results View
*   **Purpose:** Displays the ranked list of suggested matings based on the criteria defined by the user.
*   **Key Information/Controls:**
    *   **Primary Display:** Table or card view.
        *   For each target animal (e.g., ewe):
            *   A ranked list of suggested mates (e.g., rams).
            *   For each suggested pairing:
                *   Mate ID (Eartag)
                *   Calculated Selection Index Score
                *   Expected Progeny Inbreeding Coefficient
                *   Expected Progeny EBVs for key traits
                *   (Optional) Warnings or flags (e.g., high inbreeding close to threshold).
    *   **Sorting and Filtering:** Options to sort suggested mates by different criteria (index, specific EBV, inbreeding).
    *   **Export Options:** Button to export the suggestions (e.g., to CSV).
    *   **Save Plan Button (Optional):** To save a specific set of chosen pairings as a "Mating Plan."
*   **Interaction with NLP (if relevant):** This view is populated by the results of a "suggest mates" command, whether initiated via NLP or the Mating Plan Setup form. Users might ask follow-up NLP questions like "show me more details about Ram X from these suggestions."

---

## 8. Trait Management View

*   **View/Component Name:** Trait Management View
*   **Purpose:** Allows administrators or users with appropriate permissions to define and manage traits that can be recorded and evaluated in the system.
*   **Key Information/Controls:**
    *   **Trait List Table:**
        *   Columns: TraitID, Trait Name, Description, Unit of Measure, Category.
        *   Edit/Delete buttons for each trait (with checks for usage before deletion).
    *   **"Add New Trait" Button:** Opens a form/modal for trait creation.
    *   **Trait Add/Edit Form:**
        *   Fields: Trait Name (text input), Description (text area), Unit of Measure (text input, e.g., kg, cm, score 1-5), Category (dropdown, e.g., Growth, Wool, Reproduction, Health).
        *   Save/Cancel buttons.
*   **Interaction with NLP (if relevant):** Less direct interaction, but NLP relies on well-defined traits. For example, "record fleece weight" uses the 'fleece weight' trait defined here. If a user tries to record a phenotype for an unrecognized trait via NLP, the system might respond, "Trait 'XYZ' is not defined. Would you like to add it?" potentially linking to this view.

---

## 9. User/System Administration (Briefly)

*   **View/Component Name:** Admin Panel
*   **Purpose:** For system administrators to manage users, settings, and overall system health.
*   **Key Information/Controls (Conceptual):**
    *   User management (add, edit, deactivate users, assign roles/permissions).
    *   System settings (e.g., default values, integration parameters).
    *   Audit logs.
    *   Database backup/restore options.
    *   Management of breed codes, contemporary group definitions, etc.
*   **Interaction with NLP (if relevant):** Generally minimal. Administrative tasks are typically performed via structured interfaces.

---
```
