"""
nlp_interface_designer.py

This script designs and conceptually develops a Natural Language Processing (NLP)
interface for the Sheep Breeding Management System. It simulates intent recognition,
entity extraction, and dispatching actions to backend modules.

This script does NOT make actual LLM calls or interact with live backend modules.
It's a design exercise to illustrate the flow.
"""
import re

# 1. Define a List of Supported Intents (Example Commands)
SUPPORTED_COMMANDS = [
    "Add a new lamb with eartag L001, sex male, birth date 2023-04-15, sire S010, dam D025.",
    "Add new animal eartag K030 sex female birthdate 2022-03-10 breed Merino sire S001 dam D005 birth weight 4.5 weaning weight 28 notes 'test animal'",
    "Record weaning weight 25.5 kg for animal L001 on 2023-07-20.",
    "Log phenotype: animal L002, trait fleece diameter, value 18.5 microns, date 2024-01-15.",
    "Show details for animal S010.",
    "Get info for sheep D025.",
    "What are the EBVs for trait 'fleece weight' for ram S010?",
    "Get EBV for animal L001, trait 'growth rate'.",
    "Suggest mates for ewe D025, prioritize growth rate and limit inbreeding to 3%.",
    "Find optimal pairings for rams R001, R002 with ewes E010, E011 considering wool quality index and max inbreeding 2.5%.",
    "What is the expected inbreeding if I mate S010 with D025?",
    "Calculate inbreeding for progeny of sire S005 and dam D015.",
    "List all animals born after 2023-01-01.",
    "Show sheep born before 2022-06-01 with breed Texel.",
    "Update animal L001: set weaning weight to 27.2 kg and add note 're-weighed'.",
    "Deactivate animal S010."
]

# --- Intent and Entity Definitions (Conceptual) ---
INTENT_ADD_ANIMAL = "ADD_ANIMAL"
INTENT_RECORD_PHENOTYPE = "RECORD_PHENOTYPE"
INTENT_GET_ANIMAL_DETAILS = "GET_ANIMAL_DETAILS"
INTENT_GET_EBVS = "GET_EBVS"
INTENT_SUGGEST_MATES = "SUGGEST_MATES"
INTENT_CALCULATE_INBREEDING = "CALCULATE_INBREEDING"
INTENT_LIST_ANIMALS = "LIST_ANIMALS"
INTENT_UPDATE_ANIMAL = "UPDATE_ANIMAL"
INTENT_DEACTIVATE_ANIMAL = "DEACTIVATE_ANIMAL"
INTENT_UNKNOWN = "UNKNOWN"


# 2. Simple Parser (`parse_command` function)
def parse_command(command: str):
    """
    Parses a natural language command to identify intent and extract entities.
    Uses basic keyword spotting and regular expressions. This is a simplified simulation.
    """
    command_lower = command.lower()
    intent = INTENT_UNKNOWN
    entities = {}

    # --- Order of regex matters: more specific ones first ---

    # Suggest Mates
    match = re.search(r"(suggest mates|find pairings|optimal pairings) for (ewe|ewes|ram|rams) ([\w\s,]+?)(,|and|considering|prioritize) ([\w\s,]+?)( and limit inbreeding to | max inbreeding )([\d\.]+%)", command_lower)
    if not match: # Simpler version
        match = re.search(r"(suggest mates|find pairings) for (ewe|ewes|ram|rams) ([\w\s,]+?)( prioritize | considering )([\w\s]+)( and limit inbreeding to | max inbreeding )([\d\.]+%)", command_lower)
    if match:
        intent = INTENT_SUGGEST_MATES
        entities['target_animals_ids'] = [animal.strip() for animal in match.group(3).split(',')]
        entities['selection_criteria_raw'] = match.group(5).strip() # Further parsing needed for actual criteria
        entities['max_inbreeding_threshold'] = match.group(7).strip()
        # A more sophisticated parser would differentiate target_animals from potential_mates if listed separately
        if "with" in command_lower and "rams" in match.group(2): # "pairings for rams R1, R2 with ewes E1, E2"
             potential_mates_part = re.search(r"with (ewe|ewes) ([\w\s,]+?) considering", command_lower)
             if potential_mates_part:
                 entities['potential_mates_ids'] = [animal.strip() for animal in potential_mates_part.group(2).split(',')]
        elif "with" in command_lower and "ewes" in match.group(2): # "suggest mates for ewes E1, E2 with rams R1, R2"
            potential_mates_part = re.search(r"with (ram|rams) ([\w\s,]+?) prioritize", command_lower)
            if potential_mates_part:
                 entities['potential_mates_ids'] = [animal.strip() for animal in potential_mates_part.group(2).split(',')]


    # Calculate Expected Inbreeding
    elif "expected inbreeding" in command_lower or "calculate inbreeding for progeny of" in command_lower:
        match = re.search(r"(?:mate|sire) (\w+) (?:with|and) (?:dam|ewe) (\w+)", command_lower)
        if match:
            intent = INTENT_CALCULATE_INBREEDING
            entities['sire_id'] = match.group(1).upper()
            entities['dam_id'] = match.group(2).upper()

    # Add Animal (more complex regex)
    elif "add a new" in command_lower or "add new animal" in command_lower:
        intent = INTENT_ADD_ANIMAL
        match_eartag = re.search(r"eartag (\w+)", command_lower)
        if match_eartag: entities['Eartag'] = match_eartag.group(1).upper()

        match_sex = re.search(r"sex (male|female|m|f)", command_lower)
        if match_sex: entities['Sex'] = 'Male' if match_sex.group(1).startswith('m') else 'Female'

        match_birth_date = re.search(r"(?:birth date|birthdate) (\d{4}-\d{2}-\d{2})", command_lower)
        if match_birth_date: entities['BirthDate'] = match_birth_date.group(1)

        match_sire = re.search(r"sire (\w+)", command_lower)
        if match_sire: entities['SireID'] = match_sire.group(1).upper()

        match_dam = re.search(r"dam (\w+)", command_lower)
        if match_dam: entities['DamID'] = match_dam.group(1).upper()

        match_breed = re.search(r"breed (\w+)", command_lower)
        if match_breed: entities['Breed'] = match_breed.group(1)

        match_bw = re.search(r"birth weight ([\d\.]+)", command_lower)
        if match_bw: entities['BirthWeight'] = float(match_bw.group(1))

        match_ww = re.search(r"weaning weight ([\d\.]+)", command_lower)
        if match_ww: entities['WeaningWeight'] = float(match_ww.group(1))

        match_notes = re.search(r"notes '([^']+)'", command_lower)
        if match_notes: entities['Notes'] = match_notes.group(1)


    # Record Phenotype
    elif "record" in command_lower or "log phenotype" in command_lower:
        intent = INTENT_RECORD_PHENOTYPE
        match_animal = re.search(r"animal (\w+)", command_lower)
        if match_animal: entities['animal_id'] = match_animal.group(1).upper()

        # Try to get trait name, might be multi-word or in quotes
        match_trait = re.search(r"trait ['\"]?([\w\s]+?)['\"]?(?:,| value)", command_lower) # handles 'fleece weight' or fleece weight
        if not match_trait: # simpler form like "record weaning weight 25.5"
            match_trait_simple = re.search(r"record ([\w\s]+?) ([\d\.]+)", command_lower)
            if match_trait_simple:
                 entities['trait_name'] = match_trait_simple.group(1).strip()
                 entities['value'] = float(match_trait_simple.group(2))

        if match_trait: entities['trait_name'] = match_trait.group(1).strip()

        match_value = re.search(r"value ([\d\.]+)", command_lower) # kg, microns etc. might be part of trait name or ignored for now
        if match_value: entities['value'] = float(match_value.group(1))

        match_unit = re.search(r"value [\d\.]+ (\w+)", command_lower)
        if match_unit: entities['unit'] = match_unit.group(1)

        match_date = re.search(r"(?:on|date) (\d{4}-\d{2}-\d{2})", command_lower)
        if match_date: entities['date'] = match_date.group(1)


    # Get EBVs
    elif "what are the ebvs" in command_lower or "get ebv for" in command_lower:
        intent = INTENT_GET_EBVS
        match_animal = re.search(r"(?:for|animal|ram|ewe) (\w+)", command_lower)
        if match_animal: entities['animal_id'] = match_animal.group(1).upper()

        match_trait = re.search(r"trait ['\"]([\w\s]+?)['\"]", command_lower) # Trait in quotes
        if not match_trait:
             match_trait = re.search(r"trait ([\w\s]+?)(?: for|$)", command_lower) # Trait not in quotes

        if match_trait: entities['trait_name'] = match_trait.group(1).strip()


    # Update Animal
    elif "update animal" in command_lower:
        intent = INTENT_UPDATE_ANIMAL
        match_animal = re.search(r"update animal (\w+)", command_lower)
        if match_animal: entities['animal_id'] = match_animal.group(1).upper()

        # Example: "set weaning weight to 27.2 kg"
        updates_raw = re.findall(r"set ([\w\s]+?) to ([\w\d\.\s']+?)(?: and|$|,)", command_lower)
        parsed_updates = {}
        for field, value in updates_raw:
            field_key = "".join(word.capitalize() for word in field.strip().split()) # WeaningWeight
            # Basic value parsing, could be improved
            if re.match(r"[\d\.]+", value.strip()):
                parsed_updates[field_key] = float(value.strip().split()[0]) # take first part if "27.2 kg"
            else:
                parsed_updates[field_key] = value.strip().replace("'", "")
        if parsed_updates: entities['update_data'] = parsed_updates

        match_notes = re.search(r"add note '([^']+)'", command_lower) # Specific handling for notes
        if match_notes:
            if 'update_data' not in entities: entities['update_data'] = {}
            entities['update_data']['Notes'] = match_notes.group(1)


    # Deactivate Animal
    elif "deactivate animal" in command_lower:
        intent = INTENT_DEACTIVATE_ANIMAL
        match_animal = re.search(r"deactivate animal (\w+)", command_lower)
        if match_animal: entities['animal_id'] = match_animal.group(1).upper()


    # Get Animal Details
    elif "show details for animal" in command_lower or "get info for sheep" in command_lower:
        intent = INTENT_GET_ANIMAL_DETAILS
        match = re.search(r"(?:animal|sheep) (\w+)", command_lower)
        if match: entities['animal_id'] = match.group(1).upper()


    # List Animals
    elif "list all animals" in command_lower or "show sheep" in command_lower:
        intent = INTENT_LIST_ANIMALS
        match_after = re.search(r"born after (\d{4}-\d{2}-\d{2})", command_lower)
        if match_after: entities['born_after_date'] = match_after.group(1)

        match_before = re.search(r"born before (\d{4}-\d{2}-\d{2})", command_lower)
        if match_before: entities['born_before_date'] = match_before.group(1)

        match_breed = re.search(r"with breed (\w+)", command_lower)
        if match_breed: entities['breed'] = match_breed.group(1)


    return {'intent': intent, 'entities': entities}


# 3. Dispatcher (`dispatch_action` function)
def dispatch_action(parsed_output):
    """
    Dispatches action based on parsed intent and entities.
    Simulates calling backend functions and generating a user-friendly response.
    """
    intent = parsed_output['intent']
    entities = parsed_output['entities']
    response = "I'm sorry, I didn't understand that."
    action_details = ""

    if intent == INTENT_ADD_ANIMAL:
        action_details = f"Conceptual call: data_manager.add_animal(db_conn, animal_data={entities})"
        eartag = entities.get('Eartag', 'UnknownEartag')
        response = f"Okay, I've conceptually added lamb {eartag} to the database."
        # Real system: actual_id = data_manager.add_animal(...); response = f"Added animal {eartag} with ID {actual_id}"

    elif intent == INTENT_RECORD_PHENOTYPE:
        action_details = f"Conceptual call: data_manager.add_phenotypic_record(db_conn, record_data={entities})"
        animal_id = entities.get('animal_id', 'UnknownAnimal')
        trait_name = entities.get('trait_name', 'UnknownTrait')
        value = entities.get('value', 'N/A')
        response = f"Recorded {trait_name} of {value} for animal {animal_id}."

    elif intent == INTENT_GET_ANIMAL_DETAILS:
        animal_id = entities.get('animal_id', 'UnknownAnimal')
        action_details = f"Conceptual call: data_manager.get_animal(db_conn, animal_id='{animal_id}')"
        # Simulated data
        sim_data = f"Eartag: {animal_id}, Sex: Male, BirthDate: 2022-01-01, Breed: Merino, Sire: S001, Dam: D002, IsActive: True"
        response = f"Details for {animal_id}: {sim_data}."

    elif intent == INTENT_GET_EBVS:
        animal_id = entities.get('animal_id', 'UnknownAnimal')
        trait_name = entities.get('trait_name', 'UnknownTrait')
        action_details = f"Conceptual call: genetic_evaluator.get_ebvs_for_animal_and_trait(db_conn, animal_id='{animal_id}', trait_name='{trait_name}')"
        # Simulated EBV
        sim_ebv = round(0.5 + (-0.1 if 'fleece' in trait_name else 0.2), 2)
        response = f"The EBV for {trait_name} for {animal_id} is {sim_ebv}."

    elif intent == INTENT_SUGGEST_MATES:
        target_ids = entities.get('target_animals_ids', [])
        criteria = entities.get('selection_criteria_raw', 'general merit')
        inbreeding = entities.get('max_inbreeding_threshold', 'N/A')
        action_details = (f"Conceptual call: mating_planner.suggest_mates(db_conn, "
                          f"target_animals_ids={target_ids}, ..., "
                          f"selection_criteria='{criteria}', max_inbreeding_threshold='{inbreeding}')")
        # Simulated suggestions
        sim_suggestions = f"Ram X (EBV Index: +1.2, Inbreeding: 2.1%), Ram Y (EBV Index: +1.1, Inbreeding: 1.5%)"
        response = f"For {', '.join(target_ids)}, based on '{criteria}' and max inbreeding {inbreeding}, suggestions are: {sim_suggestions}."

    elif intent == INTENT_CALCULATE_INBREEDING:
        sire_id = entities.get('sire_id', 'UnknownSire')
        dam_id = entities.get('dam_id', 'UnknownDam')
        action_details = (f"Conceptual call: mating_planner.calculate_expected_inbreeding("
                          f"sire_id='{sire_id}', dam_id='{dam_id}', relationship_matrix, ...)")
        # Simulated inbreeding
        sim_f = round(abs(hash(sire_id) - hash(dam_id)) % 500 / 10000, 4) # dummy value 0-5%
        response = f"The expected inbreeding if mating {sire_id} with {dam_id} is {sim_f*100:.2f}%."

    elif intent == INTENT_LIST_ANIMALS:
        filters = {k: v for k, v in entities.items()}
        action_details = f"Conceptual call: data_manager.list_animals(db_conn, filters={filters})"
        # Simulated list
        sim_list = "L001 (Born 2023-04-15), K030 (Born 2022-03-10)"
        response = f"Animals matching criteria: {sim_list}."
        if not filters:
            response = f"All animals: {sim_list} (list might be truncated)."

    elif intent == INTENT_UPDATE_ANIMAL:
        animal_id = entities.get('animal_id', 'UnknownAnimal')
        update_data = entities.get('update_data', {})
        action_details = f"Conceptual call: data_manager.update_animal(db_conn, animal_id='{animal_id}', update_data={update_data})"
        response = f"Okay, I've updated animal {animal_id} with {update_data}."

    elif intent == INTENT_DEACTIVATE_ANIMAL:
        animal_id = entities.get('animal_id', 'UnknownAnimal')
        action_details = f"Conceptual call: data_manager.deactivate_animal(db_conn, animal_id='{animal_id}')"
        response = f"Okay, animal {animal_id} has been marked as inactive."

    else: # INTENT_UNKNOWN
        action_details = "No action taken."
        response = "I'm sorry, I could not understand your request. Please try rephrasing."

    return response, action_details


# 4. Main Interaction Loop (Conceptual)
if __name__ == '__main__':
    print("NLP Interface Designer - Conceptual Test\n")

    # Add a few more complex or slightly different commands for testing
    test_commands = SUPPORTED_COMMANDS + [
        "Show me sheep D025 details.",
        "What's the inbreeding for S010 and D025?", # Should map to CALCULATE_INBREEDING
        "Add an animal: eartag N001, sex f, birthdate 2024-01-01.",
        "record fleece weight 3.1 kg for sheep N001 on 2024-06-01",
        "find mates for ewe E001, consider fleece quality and keep inbreeding below 1.5%",
        "update animal L001 set Notes to 'High quality fleece prospect'",
        "I want to list animals of breed Poll Dorset born after 2023-03-01" # variation of LIST_ANIMALS
    ]

    for i, command in enumerate(test_commands):
        print(f"--- Command {i+1} ---")
        print(f"User: {command}")
        parsed = parse_command(command)
        print(f"Parsed: Intent='{parsed['intent']}', Entities={parsed['entities']}")
        simulated_response, conceptual_action = dispatch_action(parsed)
        print(f"Action: {conceptual_action}")
        print(f"System: {simulated_response}\n")

```
