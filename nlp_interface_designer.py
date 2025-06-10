"""
nlp_interface_designer.py

This script designs and conceptually develops an ENHANCED Natural Language Processing (NLP)
interface for the Sheep Breeding Management System. It simulates:
- Multi-stage intent recognition and entity extraction.
- Dispatching actions to backend modules with more structured data.
- Generating more informative and context-aware responses.
- Basic multi-turn conversation handling using session context.

This script does NOT make actual LLM calls or interact with live backend modules.
It's a design exercise to illustrate the enhanced flow.
"""
import re
import datetime # For date parsing and generation

# --- Session Context (Simulated) ---
session_context = {
    "last_query_results": None,      # Could store IDs, a summary, or full data
    "last_subject_type": None,       # e.g., "animal_list", "animal_detail"
    "last_filters_applied": None,    # Store filters used to get last_query_results
    "pending_clarification": None,   # { "question": "Which animal?", "expected_entity": "animal_id", "original_parsed": {}}
    "user_preferences": {"date_format": "YYYY-MM-DD"} # Example
}

# --- Enhanced Intent and Entity Definitions ---

# Primary Intents
PI_QUERY_DATA = "QUERY_DATA"
PI_ADD_DATA = "ADD_DATA"
PI_UPDATE_DATA = "UPDATE_DATA"
PI_DELETE_DATA = "DELETE_DATA"
PI_RUN_ANALYSIS = "RUN_ANALYSIS" # For BLUP, Inbreeding, Mate Suggestion
PI_SYSTEM_COMMAND = "SYSTEM_COMMAND" # Help, define trait, etc.
PI_CONTEXTUAL_FOLLOW_UP = "CONTEXTUAL_FOLLOW_UP"
PI_CLARIFICATION_RESPONSE = "CLARIFICATION_RESPONSE"
PI_UNKNOWN = "UNKNOWN"

# Subjects (examples, can be expanded)
SUBJ_ANIMAL = "animal"
SUBJ_PHENOTYPE = "phenotype"
SUBJ_EBV = "ebv"
SUBJ_GEBV = "gebv"
SUBJ_PEDIGREE = "pedigree"
SUBJ_INBREEDING = "inbreeding"
SUBJ_MATES = "mates"
SUBJ_BLUP_RUN = "blup_run"
SUBJ_TRAIT = "trait"
SUBJ_RELATIONSHIP_MATRIX = "relationship_matrix"
SUBJ_GENETIC_TREND = "genetic_trend"
SUBJ_NONE = "none" # For general commands or when subject is implicit in primary intent

# --- Conceptual Entity Extraction Helpers (Simulated with Regex) ---

def _parse_animal_identifiers(text_chunk):
    """Parses single or multiple animal IDs (e.g., eartags)."""
    ids = re.findall(r'\b([A-Z0-9]{3,10})\b', text_chunk.upper()) # Simple eartag like pattern
    if not ids:
        # Try to find "animal L001", "ram S001", "ewe D002"
        ids_descriptive = re.findall(r'(?:animal|ram|ewe) ([A-Z0-9]{3,10})', text_chunk.upper())
        ids.extend(ids_descriptive)

    # Handle "L001 and L002", "L001, L002, L003"
    if not ids and (',' in text_chunk or ' and ' in text_chunk):
        parts = re.split(r',| and ', text_chunk)
        for part in parts:
            id_match = re.search(r'\b([A-Z0-9]{3,10})\b', part.strip().upper())
            if id_match:
                ids.append(id_match.group(1))

    return list(set(ids)) # Unique IDs

def _parse_trait_names(text_chunk):
    """Parses trait names, potentially multi-word or in quotes."""
    # Prioritize quoted traits
    quoted_traits = re.findall(r"['\"]([\w\s]+?)['\"]", text_chunk)
    if quoted_traits:
        return [t.strip() for t in quoted_traits]

    # Keywords often precede trait names
    # This is very heuristic and would be much better with true NLU/NER
    potential_traits = []
    if "trait" in text_chunk:
        # "trait weaning weight", "traits fleece weight and staple length"
        m = re.search(r"trait(?:s)?\s+([\w\s]+?)(?: for| and |$|,)", text_chunk)
        if m:
            trait_phrase = m.group(1)
            # Split if "and" or "," is present, assuming simple cases
            if " and " in trait_phrase:
                potential_traits.extend([t.strip() for t in trait_phrase.split(" and ")])
            elif "," in trait_phrase:
                 potential_traits.extend([t.strip() for t in trait_phrase.split(",")])
            else:
                potential_traits.append(trait_phrase.strip())

    # Fallback for simple cases like "record weaning weight"
    if not potential_traits:
        simple_trait_match = re.search(r"(?:record|log|average|compare ebvs for|ebvs for|trend for)\s+([\w\s]+?)\s+(?:[\d\.]+|for|between|over|per|$)", text_chunk)
        if simple_trait_match:
            potential_traits.append(simple_trait_match.group(1).strip())

    # Known traits could be used here for validation in a real system
    return [t for t in potential_traits if t] # Filter empty

def _parse_dates_and_ranges(text_chunk):
    """Parses dates (YYYY-MM-DD) and simple date ranges."""
    dates = {}
    # Specific dates
    specific_dates = re.findall(r"(\d{4}-\d{2}-\d{2})", text_chunk)
    if specific_dates:
        dates['specific_dates'] = specific_dates

    # Ranges
    if "after" in text_chunk:
        m = re.search(r"after (\d{4}-\d{2}-\d{2})", text_chunk)
        if m: dates['born_after_date'] = m.group(1)
    if "before" in text_chunk:
        m = re.search(r"before (\d{4}-\d{2}-\d{2})", text_chunk)
        if m: dates['born_before_date'] = m.group(1)
    if "between" in text_chunk and "and" in text_chunk : # date range
        m = re.search(r"between (\d{4}-\d{2}-\d{2}) and (\d{4}-\d{2}-\d{2})", text_chunk)
        if m :
            dates['date_range_start'] = m.group(1)
            dates['date_range_end'] = m.group(2)
    if "last" in text_chunk and ("year" in text_chunk or "month" in text_chunk or "day" in text_chunk):
        m = re.search(r"last (\d+)?\s*(year|month|day)s?", text_chunk)
        if m:
            num = int(m.group(1)) if m.group(1) else 1
            unit = m.group(2)
            dates['relative_range'] = {'count': num, 'unit': unit}
    if "today" in text_chunk:
        dates['specific_dates'] = dates.get('specific_dates', []) + [datetime.date.today().isoformat()]

    return dates

def _parse_numeric_values_and_ranges(text_chunk, field_name="value"):
    """Parses numeric values and ranges (e.g., > 4.2, between 28 and 32)."""
    conditions = []
    # e.g. "greater than 4.2", "weight > 4.2"
    m_gt = re.search(r"(?:greater than|>) ([\d\.]+)", text_chunk)
    if m_gt: conditions.append({'field': field_name, 'op': '>', 'value': float(m_gt.group(1))})

    m_lt = re.search(r"(?:less than|<) ([\d\.]+)", text_chunk)
    if m_lt: conditions.append({'field': field_name, 'op': '<', 'value': float(m_lt.group(1))})

    m_eq = re.search(r"(?:equal to|=) ([\d\.]+)", text_chunk) # Less common for ranges
    if m_eq: conditions.append({'field': field_name, 'op': '=', 'value': float(m_eq.group(1))})

    m_between = re.search(r"between ([\d\.]+) and ([\d\.]+)", text_chunk)
    if m_between:
        conditions.append({'field': field_name, 'op': '>=', 'value': float(m_between.group(1))})
        conditions.append({'field': field_name, 'op': '<=', 'value': float(m_between.group(2))})

    # Direct value, e.g., "weight 25.5 kg"
    m_direct = re.search(f"{field_name.lower()} ([\d\.]+)", text_chunk) # Simplistic
    if not conditions and m_direct:
         conditions.append({'field': field_name, 'op': '=', 'value': float(m_direct.group(1))})
    elif not conditions: # Try to find a standalone number if no conditions yet
        m_standalone = re.search(r"([\d\.]+)\s*(?:kg|microns|mm|%|days|points)?", text_chunk)
        if m_standalone:
            # Check if this number was already captured by a more specific regex (e.g. date)
            # This part is tricky without full parsing context
            if not re.search(r"\d{4}-\d{2}-\d{2}", m_standalone.group(1)): # Avoid capturing parts of dates
                 conditions.append({'field': field_name, 'op': '=', 'value': float(m_standalone.group(1))})


    return conditions

def _parse_conditions(text_chunk):
    """Parses multiple conditions like 'field operator value'."""
    conditions = []
    # Example: "birth weight > 4.2 kg and weaning weight between 28kg and 32kg"
    # This is complex; we'll simulate by looking for known field names + ranges
    if "birth weight" in text_chunk:
        conditions.extend(_parse_numeric_values_and_ranges(text_chunk, "BirthWeight"))
    if "weaning weight" in text_chunk:
        bw_chunk_match = re.search(r"weaning weight ([\w\s\d\.]+)(?:and|$|,)", text_chunk) # isolate relevant part
        if bw_chunk_match:
            conditions.extend(_parse_numeric_values_and_ranges(bw_chunk_match.group(1), "WeaningWeight"))
    if "fleece diameter" in text_chunk:
        fd_chunk_match = re.search(r"fleece diameter ([\w\s\d\.]+)(?:and|$|,)", text_chunk)
        if fd_chunk_match:
            conditions.extend(_parse_numeric_values_and_ranges(fd_chunk_match.group(1), "FleeceDiameter"))
    # Add more fields as needed: Sex, Breed, etc.
    sex_match = re.search(r"sex (male|female)", text_chunk)
    if sex_match: conditions.append({'field': 'Sex', 'op': '=', 'value': sex_match.group(1).capitalize()})

    breed_match = re.search(r"breed (\w+)", text_chunk) # Simple breed match
    if breed_match: conditions.append({'field': 'Breed', 'op': '=', 'value': breed_match.group(1).capitalize()})

    # Active status
    if "active" in text_chunk and "inactive" not in text_chunk :
        conditions.append({'field': 'IsActive', 'op': '=', 'value': True})
    if "inactive" in text_chunk:
        conditions.append({'field': 'IsActive', 'op': '=', 'value': False})

    return conditions


def _parse_sorting_criteria(text_chunk):
    """Parses sorting criteria, e.g., 'order by birth date descending'."""
    sort_criteria = {}
    m = re.search(r"(?:order by|sorted by) ([\w\s]+?)( (ascending|descending|asc|desc))?$", text_chunk)
    if m:
        sort_criteria['field'] = m.group(1).strip().replace(" ", "_") # e.g. birth_date
        if m.group(3) and m.group(3).startswith("desc"):
            sort_criteria['direction'] = "DESC"
        else:
            sort_eria['direction'] = "ASC"
    return sort_criteria

def _parse_aggregation_requests(text_chunk):
    """Parses aggregation requests like 'average X', 'count Y'."""
    aggregations = []
    if "average" in text_chunk or "avg" in text_chunk:
        m = re.search(r"(?:average|avg) ([\w\s]+?)(?: for| per|$)", text_chunk)
        if m: aggregations.append({'func': 'AVG', 'field': m.group(1).strip()})
    if "count" in text_chunk:
        # "count animals per breed" vs "count animals"
        m_per = re.search(r"count ([\w\s]+?) per ([\w\s]+)", text_chunk)
        if m_per:
             aggregations.append({'func': 'COUNT', 'field': m_per.group(1).strip(), 'group_by': m_per.group(2).strip()})
        else:
            m = re.search(r"count ([\w\s]+)", text_chunk) # e.g. "count active animals"
            if m: aggregations.append({'func': 'COUNT', 'field': m.group(1).strip()})
    if "minimum" in text_chunk or "min" in text_chunk:
        m = re.search(r"(?:minimum|min) ([\w\s]+?)(?: for| per|$)", text_chunk)
        if m: aggregations.append({'func': 'MIN', 'field': m.group(1).strip()})
    if "maximum" in text_chunk or "max" in text_chunk:
        m = re.search(r"(?:maximum|max) ([\w\s]+?)(?: for| per|$)", text_chunk)
        if m: aggregations.append({'func': 'MAX', 'field': m.group(1).strip()})
    return aggregations

def _parse_contextual_cues(text_chunk):
    """Identifies contextual cues like 'these', 'of those', 'it'."""
    context_entities = {}
    if re.search(r"\b(these|those|them)\b", text_chunk):
        context_entities['use_context_subject'] = True
        # Could try to identify what "these" refers to more specifically if needed
        m_field = re.search(r"for (?:these|those) animals, show their ([\w\s]+)", text_chunk)
        if m_field:
            context_entities['requested_fields_for_context'] = [f.strip() for f in m_field.group(1).split('and')]

    if re.search(r"\b(it|its)\b", text_chunk) and session_context.get("last_subject_type") == "animal_detail":
        context_entities['use_context_subject'] = True # Referring to the animal in detail view
        context_entities['context_animal_id'] = session_context.get("last_query_results", {}).get("AnimalID")

    return context_entities

# --- Refactored `parse_command` ---
def parse_command(command: str, current_session_context: dict):
    """
    Parses a natural language command to identify intent and extract entities
    using a multi-stage approach and simulated entity extraction helpers.
    """
    command_lower = command.lower()
    parsed_output = {
        'primary_intent': PI_UNKNOWN,
        'subject': SUBJ_NONE,
        'entities': {},
        'ambiguity_info': None,
        'original_command': command
    }

    # 0. Handle clarification response first
    if current_session_context.get("pending_clarification"):
        parsed_output['primary_intent'] = PI_CLARIFICATION_RESPONSE
        # Assume the entire command is the answer to the clarification
        parsed_output['entities']['clarification_answer'] = command.strip()
        return parsed_output

    # 1. Primary Intent Identification (Keyword-based)
    # This is a simplified keyword matching. A real system would use ML/LLM.
    if "list" in command_lower or "show" in command_lower or \
       "what is" in command_lower or "what are" in command_lower or \
       "find" in command_lower or "get" in command_lower or "display" in command_lower or \
       "count" in command_lower or "average" in command_lower or "minimum" in command_lower or "maximum" in command_lower:
        parsed_output['primary_intent'] = PI_QUERY_DATA
    elif "add" in command_lower or "new animal" in command_lower or "create" in command_lower:
        parsed_output['primary_intent'] = PI_ADD_DATA
    elif "update" in command_lower or "set" in command_lower or "change" in command_lower or "modify" in command_lower:
        parsed_output['primary_intent'] = PI_UPDATE_DATA
    elif "delete" in command_lower or "remove" in command_lower or "deactivate" in command_lower or "archive" in command_lower :
        parsed_output['primary_intent'] = PI_DELETE_DATA
    elif "run blup" in command_lower or "estimate breeding values" in command_lower or \
         "suggest mates" in command_lower or "find pairings" in command_lower or \
         "calculate inbreeding" in command_lower or "compare ebvs" in command_lower or \
         "genetic trend" in command_lower or "relationship coefficient" in command_lower:
        parsed_output['primary_intent'] = PI_RUN_ANALYSIS
    elif "define trait" in command_lower: # Simple example for system command
        parsed_output['primary_intent'] = PI_SYSTEM_COMMAND
    elif re.search(r"\b(these|those|them|it|its|that evaluation|that run)\b", command_lower) and \
         (current_session_context.get("last_query_results") or current_session_context.get("last_subject_type")):
        parsed_output['primary_intent'] = PI_CONTEXTUAL_FOLLOW_UP

    # Refine primary intent if it's too general (e.g. PI_QUERY_DATA) based on keywords

    # 2. Secondary Keyword/Subject Analysis & Entity Extraction
    entities = {}
    context_cues = _parse_contextual_cues(command_lower)
    entities.update(context_cues)

    if parsed_output['primary_intent'] == PI_CONTEXTUAL_FOLLOW_UP:
        # For follow-ups, the subject is often implied or refers to previous results
        entities['animal_ids'] = _parse_animal_identifiers(command_lower) # if new animals mentioned
        entities['trait_names'] = _parse_trait_names(command_lower)
        # The actual subject might be determined in dispatch based on session_context

    elif parsed_output['primary_intent'] == PI_ADD_DATA:
        if "animal" in command_lower or "lamb" in command_lower:
            parsed_output['subject'] = SUBJ_ANIMAL
            # Simplified: assume all details are for one animal
            entities['Eartag'] = (re.search(r"eartag (\w+)", command_lower) or {}).get(1, "").upper()
            entities['Sex'] = (re.search(r"sex (male|female|m|f)", command_lower) or {}).get(1, "")
            if entities['Sex']: entities['Sex'] = 'Male' if entities['Sex'].startswith('m') else 'Female'
            entities['BirthDate'] = (re.search(r"(?:birth date|birthdate) (\d{4}-\d{2}-\d{2})", command_lower) or {}).get(1)
            entities['SireID'] = (re.search(r"sire (\w+)", command_lower) or {}).get(1, "").upper()
            entities['DamID'] = (re.search(r"dam (\w+)", command_lower) or {}).get(1, "").upper()
            entities['Breed'] = (re.search(r"breed (\w+)", command_lower) or {}).get(1, "").capitalize()
            bw_match = re.search(r"birth weight ([\d\.]+)", command_lower)
            if bw_match: entities['BirthWeight'] = float(bw_match.group(1))
            # ... and other fields
            if not entities.get('Eartag') or not entities.get('Sex') or not entities.get('BirthDate'):
                parsed_output['ambiguity_info'] = {'missing_entity': ['Eartag', 'Sex', 'BirthDate']}
        elif "phenotype" in command_lower or "record" in command_lower or "log" in command_lower: # record is often phenotype
             parsed_output['subject'] = SUBJ_PHENOTYPE
             entities['animal_ids'] = _parse_animal_identifiers(command_lower)
             entities['trait_names'] = _parse_trait_names(command_lower)
             numeric_vals = _parse_numeric_values_and_ranges(command_lower)
             if numeric_vals: entities['value'] = numeric_vals[0]['value'] # take first found
             date_info = _parse_dates_and_ranges(command_lower)
             if date_info.get('specific_dates'): entities['date'] = date_info['specific_dates'][0]
             elif date_info.get('relative_range') and date_info['relative_range']['unit'] == 'day' and date_info['relative_range']['count'] == 0: # "today"
                 entities['date'] = datetime.date.today().isoformat()

             if not entities.get('animal_ids') or not entities.get('trait_names') or 'value' not in entities:
                 parsed_output['ambiguity_info'] = {'missing_entity': ['animal_id', 'trait_name', 'value']}


    elif parsed_output['primary_intent'] == PI_QUERY_DATA:
        entities['animal_ids'] = _parse_animal_identifiers(command_lower)
        entities['trait_names'] = _parse_trait_names(command_lower)
        entities['conditions'] = _parse_conditions(command_lower)
        entities['date_filters'] = _parse_dates_and_ranges(command_lower)
        entities['sort_criteria'] = _parse_sorting_criteria(command_lower)
        entities['aggregations'] = _parse_aggregation_requests(command_lower)

        if "ebv" in command_lower or "breeding value" in command_lower:
            parsed_output['subject'] = SUBJ_EBV
        elif "animal" in command_lower or "sheep" in command_lower or "lamb" in command_lower or "ram" in command_lower or "ewe" in command_lower or entities.get('animal_ids'):
            parsed_output['subject'] = SUBJ_ANIMAL
        elif "phenotype" in command_lower or "measurement" in command_lower or entities.get('trait_names'): # If trait mentioned, likely phenotype query
            parsed_output['subject'] = SUBJ_PHENOTYPE
        elif "trend" in command_lower:
            parsed_output['subject'] = SUBJ_GENETIC_TREND
        # Default to animal if conditions or aggregations exist but no clear subject
        elif not parsed_output['subject'] and (entities['conditions'] or entities['aggregations']):
             parsed_output['subject'] = SUBJ_ANIMAL


    elif parsed_output['primary_intent'] == PI_RUN_ANALYSIS:
        if "blup" in command_lower or "estimate breeding values" in command_lower:
            parsed_output['subject'] = SUBJ_BLUP_RUN
            entities['trait_names'] = _parse_trait_names(command_lower)
            # Simplified: "using 'sex' and 'birth type' as fixed effects"
            fixed_effects_match = re.search(r"using ([\w\s,'\-]+?) as fixed effects", command_lower)
            if fixed_effects_match:
                entities['fixed_effects'] = [fe.strip().replace("'", "") for fe in fixed_effects_match.group(1).split('and')]
            animal_group_match = re.search(r"for (all [\w\s]+?)(?:, using|, include|$)", command_lower) # "for all lambs born in spring 2023"
            if animal_group_match: entities['animal_group_description'] = animal_group_match.group(1)

        elif "suggest mates" in command_lower or "find pairings" in command_lower:
            parsed_output['subject'] = SUBJ_MATES
            # This regex is basic, real parsing would be more involved
            target_match = re.search(r"for (?:ewes?|rams?) ([\w\s,]+?)(?: using| consider| prioritize|,)", command_lower)
            if target_match: entities['target_animals_ids'] = _parse_animal_identifiers(target_match.group(1))

            potential_match = re.search(r"with (?:ewes?|rams?) ([\w\s,]+?)(?: using| consider| prioritize|,)", command_lower)
            if potential_match: entities['potential_mates_ids'] = _parse_animal_identifiers(potential_match.group(1))

            criteria_match = re.search(r"(?:using|consider|prioritize) ([\w\s\d\.\*\+\-]+?)(?: and limit| but exclude|,)", command_lower)
            if criteria_match: entities['selection_criteria_raw'] = criteria_match.group(1).strip()

            inbreeding_match = re.search(r"(?:limit inbreeding to|max inbreeding|inbreeding under) ([\d\.]+%?)", command_lower)
            if inbreeding_match: entities['max_inbreeding_threshold'] = inbreeding_match.group(1)

            exclude_match = re.search(r"exclude (?:any offspring of|animals related to|ram|ewe) ([\w\s,]+)", command_lower)
            if exclude_match: entities['exclusions_raw'] = exclude_match.group(1).strip()


        elif "inbreeding" in command_lower:
            parsed_output['subject'] = SUBJ_INBREEDING
            # "inbreeding if I mate S010 with D025" or "inbreeding for animal X001"
            pair_match = re.search(r"(?:mate|sire) (\w+) (?:with|and) (?:dam|ewe) (\w+)", command_lower)
            single_match = re.search(r"inbreeding coefficient for animal (\w+)", command_lower)
            if pair_match:
                entities['sire_id'] = pair_match.group(1).upper()
                entities['dam_id'] = pair_match.group(2).upper()
            elif single_match:
                entities['animal_id'] = single_match.group(1).upper() # Inbreeding of the animal itself
            else: # Try to find two animal IDs if not explicitly sire/dam
                ids = _parse_animal_identifiers(command_lower)
                if len(ids) >= 2:
                    entities['sire_id'] = ids[0]
                    entities['dam_id'] = ids[1]
                elif len(ids) == 1:
                     entities['animal_id'] = ids[0]


    elif parsed_output['primary_intent'] == PI_SYSTEM_COMMAND:
        if "define new trait" in command_lower:
            parsed_output['subject'] = SUBJ_TRAIT
            name_match = re.search(r"trait: ['\"]?([\w\s]+?)['\"]?,", command_lower) # "trait: 'Eye Muscle Depth',"
            if name_match: entities['TraitName'] = name_match.group(1).strip()
            abbr_match = re.search(r"abbreviation (\w+),", command_lower)
            if abbr_match: entities['Abbreviation'] = abbr_match.group(1)
            unit_match = re.search(r"unit (\w+),?", command_lower)
            if unit_match: entities['UnitOfMeasure'] = unit_match.group(1)
            cat_match = re.search(r"category (\w+)", command_lower)
            if cat_match: entities['Category'] = cat_match.group(1)
            if not entities.get('TraitName'):
                 parsed_output['ambiguity_info'] = {'missing_entity': ['TraitName']}


    elif parsed_output['primary_intent'] == PI_UPDATE_DATA:
        # Simplified update parsing
        parsed_output['subject'] = SUBJ_ANIMAL # Defaulting to animal for now
        animal_id_match = re.search(r"update animal (\w+)", command_lower)
        if animal_id_match: entities['animal_id'] = animal_id_match.group(1).upper()

        updates_raw = re.findall(r"set ([\w\s]+?) to ([\w\d\.\s\'\"]+?)(?: and|$|,)", command_lower)
        parsed_updates = {}
        for field, value in updates_raw:
            field_key = "".join(word.capitalize() for word in field.strip().split()) # WeaningWeight
            value_clean = value.strip().replace("'", "").replace('"', '')
            if re.fullmatch(r"[\d\.]+", value_clean.split()[0]): # if first part is number
                parsed_updates[field_key] = float(value_clean.split()[0])
            else:
                parsed_updates[field_key] = value_clean
        if parsed_updates: entities['update_data'] = parsed_updates
        if not entities.get('animal_id') or not entities.get('update_data'):
            parsed_output['ambiguity_info'] = {'missing_entity': ['animal_id', 'update_data']}

    elif parsed_output['primary_intent'] == PI_DELETE_DATA:
        if "animal" in command_lower: # "deactivate animal S010"
            parsed_output['subject'] = SUBJ_ANIMAL
            entities['animal_ids'] = _parse_animal_identifiers(command_lower)
            if not entities.get('animal_ids'):
                parsed_output['ambiguity_info'] = {'missing_entity': ['animal_id']}

    # Fallback if intent was identified but subject remains none
    if parsed_output['primary_intent'] != PI_UNKNOWN and parsed_output['subject'] == SUBJ_NONE:
        if "animal" in command_lower or "sheep" in command_lower or "lamb" in command_lower:
            parsed_output['subject'] = SUBJ_ANIMAL
        elif "phenotype" in command_lower or "measurement" in command_lower:
            parsed_output['subject'] = SUBJ_PHENOTYPE
        elif "ebv" in command_lower:
            parsed_output['subject'] = SUBJ_EBV
        # (add more subject detection logic here if needed)

    parsed_output['entities'] = entities
    return parsed_output


# --- Revised `dispatch_action` ---
def dispatch_action(parsed_output, current_session_context):
    """
    Dispatches action based on parsed intent and entities, uses session context,
    and generates more informative responses.
    """
    primary_intent = parsed_output['primary_intent']
    subject = parsed_output['subject']
    entities = parsed_output['entities']
    ambiguity_info = parsed_output.get('ambiguity_info')

    response = "I'm sorry, I didn't quite understand that request. Could you try rephrasing?"
    action_details = "No specific action identified."
    # Clear pending clarification unless it's a clarification response itself
    if primary_intent != PI_CLARIFICATION_RESPONSE and "pending_clarification" in current_session_context:
        del current_session_context["pending_clarification"]


    if primary_intent == PI_CLARIFICATION_RESPONSE:
        pending_q = current_session_context.pop("pending_clarification", None)
        if pending_q:
            # User provided an answer. We need to merge this answer with the original parsed command
            # and re-process it. This is a simplified re-dispatch.
            original_parsed = pending_q['original_parsed']
            answered_entity_key = pending_q['expected_entity'] # e.g. 'animal_id' or 'trait_name'

            # Update the original entities with the new answer
            # This part is tricky: if expected entity was 'animal_id', the answer might be just "L001"
            # or "animal L001". We need to parse the answer in the context of the expected entity.
            if answered_entity_key == 'animal_id':
                 # Assume the answer is the animal_id or contains it
                ans_animal_ids = _parse_animal_identifiers(entities['clarification_answer'])
                if ans_animal_ids:
                    # This logic depends on whether original_parsed['entities']['animal_ids'] was a list or single item
                    original_parsed['entities'][answered_entity_key] = ans_animal_ids[0] if len(ans_animal_ids)==1 else ans_animal_ids
                else: # Could not parse animal ID from answer
                    response = f"I still couldn't identify the {answered_entity_key} from your response '{entities['clarification_answer']}'. Let's try again."
                    current_session_context['pending_clarification'] = pending_q # Re-ask
                    return response, "Awaiting further clarification."

            elif answered_entity_key == 'trait_name':
                ans_trait_names = _parse_trait_names(entities['clarification_answer'])
                if ans_trait_names:
                     original_parsed['entities'][answered_entity_key] = ans_trait_names[0] if len(ans_trait_names)==1 else ans_trait_names
                else:
                    response = f"I still couldn't identify the {answered_entity_key} from your response '{entities['clarification_answer']}'. Let's try again."
                    current_session_context['pending_clarification'] = pending_q # Re-ask
                    return response, "Awaiting further clarification."
            else:
                original_parsed['entities'][answered_entity_key] = entities['clarification_answer'] # General case

            # Clear ambiguity before re-dispatching
            if 'ambiguity_info' in original_parsed: del original_parsed['ambiguity_info']

            print(f"Re-dispatching with clarified info: {original_parsed}")
            # Recursive call or pass to a handler. For simulation, we'll just re-call dispatch_action.
            return dispatch_action(original_parsed, current_session_context)
        else:
            response = "I wasn't waiting for a clarification, but thanks for the information!"
            action_details = "No pending clarification found."

    elif ambiguity_info:
        missing_entities_list = ambiguity_info.get('missing_entity', ['required information'])
        # Formulate a clarifying question
        if 'Eartag' in missing_entities_list and subject == SUBJ_ANIMAL and primary_intent == PI_ADD_DATA :
             question = "I can help add a new animal. What is the eartag for this animal?"
             current_session_context['pending_clarification'] = {'question': question, 'expected_entity': 'Eartag', 'original_parsed': parsed_output}
        elif 'animal_id' in missing_entities_list:
            question = f"Which animal are you referring to? Please provide the Eartag or ID."
            current_session_context['pending_clarification'] = {'question': question, 'expected_entity': 'animal_id', 'original_parsed': parsed_output}
        elif 'trait_name' in missing_entities_list:
            question = f"Which trait are you interested in?"
            current_session_context['pending_clarification'] = {'question': question, 'expected_entity': 'trait_name', 'original_parsed': parsed_output}
        else:
            question = f"I need a bit more information. Specifically, what is the {missing_entities_list[0]}?"
            current_session_context['pending_clarification'] = {'question': question, 'expected_entity': missing_entities_list[0], 'original_parsed': parsed_output}

        response = question
        action_details = "Awaiting clarification from user."


    elif primary_intent == PI_QUERY_DATA:
        if subject == SUBJ_ANIMAL:
            filters_summary = []
            if entities.get('conditions'): filters_summary.extend([f"{c['field']} {c['op']} {c['value']}" for c in entities['conditions']])
            if entities.get('date_filters'): filters_summary.extend([f"{k}: {v}" for k,v in entities['date_filters'].items()])

            action_details = f"Conceptual call: data_manager.list_animals(filters={entities})"
            if filters_summary:
                response = f"Okay, searching for animals where {' and '.join(filters_summary)}. (Results would appear in 'Animal List View')"
            else:
                response = "Okay, listing all animals. (Results would appear in 'Animal List View')"
            # Simulate some results for context
            current_session_context['last_query_results'] = {'ids': ['L001', 'L002', 'K030'], 'count': 3, 'filters': entities}
            current_session_context['last_subject_type'] = 'animal_list'
            if entities.get('aggregations'):
                agg = entities['aggregations'][0]
                response = f"The {agg['func'].lower()} {agg['field']} is [simulated_value]. Grouped by {agg.get('group_by', 'N/A')} if specified."
                if agg.get('group_by'):
                     response += " (This could be shown in a chart or summary table in the UI - see 'Dashboard' or 'Reports' in ui_conceptualization.md)"


        elif subject == SUBJ_EBV:
            animal_id = entities.get('animal_ids')[0] if entities.get('animal_ids') else "[UnknownAnimal]"
            trait_name = entities.get('trait_names')[0] if entities.get('trait_names') else "[UnknownTrait]"
            action_details = f"Conceptual call: genetic_evaluator.get_ebvs(animal_id='{animal_id}', trait_name='{trait_name}')"
            response = f"The EBV for {trait_name} for animal {animal_id} is [simulated_ebv_value]. (Viewable in 'Animal Detail View' under EBV/GEBV tab - see ui_conceptualization.md)"
            current_session_context['last_query_results'] = {'animal_id': animal_id, 'trait_name': trait_name, 'ebv': 0.5} # dummy
            current_session_context['last_subject_type'] = 'ebv_detail'

        # Add more query subjects...
        else:
            response = "I can query data, but I'm not sure which subject you're asking about (animal, ebv, phenotype etc.)."


    elif primary_intent == PI_ADD_DATA:
        if subject == SUBJ_ANIMAL:
            eartag = entities.get('Eartag', '[UnknownEartag]')
            action_details = f"Conceptual call: data_manager.add_animal(animal_data={entities})"
            response = f"Confirmed: Adding new animal {eartag}. This record will be saved to the database."
            current_session_context['last_query_results'] = {'eartag': eartag, 'action': 'added'}
            current_session_context['last_subject_type'] = 'animal_detail_added'
        elif subject == SUBJ_PHENOTYPE:
            animal_id = entities.get('animal_ids')[0] if entities.get('animal_ids') else "[UnknownAnimal]"
            trait_name = entities.get('trait_names')[0] if entities.get('trait_names') else "[UnknownTrait]"
            value = entities.get('value', '[N/A]')
            action_details = f"Conceptual call: data_manager.add_phenotypic_record(record_data={entities})"
            response = f"Okay, recording {trait_name} of {value} for animal {animal_id}."
            current_session_context['last_query_results'] = {'animal_id': animal_id, 'trait': trait_name, 'value': value, 'action': 'phenotype_recorded'}
            current_session_context['last_subject_type'] = 'phenotype_record_added'


    elif primary_intent == PI_RUN_ANALYSIS:
        if subject == SUBJ_INBREEDING:
            sire = entities.get('sire_id', '[Sire]')
            dam = entities.get('dam_id', '[Dam]')
            animal = entities.get('animal_id', None)
            if animal:
                action_details = f"Conceptual call: genetic_evaluator.calculate_inbreeding_for_animal(animal_id='{animal}')"
                response = f"The inbreeding coefficient for animal {animal} is [simulated_inbreeding_value]."
            else:
                action_details = f"Conceptual call: mating_planner.calculate_expected_inbreeding(sire_id='{sire}', dam_id='{dam}')"
                response = f"The expected inbreeding for progeny of {sire} and {dam} is [simulated_progeny_inbreeding_value]."

        elif subject == SUBJ_MATES:
            targets = entities.get('target_animals_ids', ['[TargetAnimals]'])
            criteria = entities.get('selection_criteria_raw', '[DefaultCriteria]')
            inbreeding_limit = entities.get('max_inbreeding_threshold', '[DefaultLimit]')
            action_details = f"Conceptual call: mating_planner.suggest_mates(targets={targets}, criteria='{criteria}', max_inbreeding={inbreeding_limit}, exclusions={entities.get('exclusions_raw')})"
            response = f"Generating mate suggestions for {', '.join(targets)} using criteria '{criteria}' with max inbreeding {inbreeding_limit}. (Results would appear in 'Suggested Matings Display' - see ui_conceptualization.md)"
            current_session_context['last_query_results'] = {'type': 'mate_suggestion', 'targets': targets, 'count': 3} # dummy
            current_session_context['last_subject_type'] = 'mate_suggestion_list'


    elif primary_intent == PI_CONTEXTUAL_FOLLOW_UP:
        last_subject = current_session_context.get('last_subject_type')
        last_results = current_session_context.get('last_query_results')
        requested_fields = entities.get('requested_fields_for_context', [])

        if last_subject == 'animal_list' and last_results and last_results.get('ids'):
            # Example: "For these animals, show their sire IDs"
            # Here, 'these animals' refers to last_results['ids']
            ids_to_query = last_results['ids']
            fields_to_show = requested_fields if requested_fields else ['SireID', 'DamID'] # Default follow-up
            action_details = f"Conceptual call: data_manager.get_animal_details_for_list(animal_ids={ids_to_query}, fields={fields_to_show})"
            response = f"For the previously listed {len(ids_to_query)} animals, their {', '.join(fields_to_show)} are: [simulated_data_list]. (This could update the 'Animal List View' or show a modal - ui_conceptualization.md)"
            # Update context if needed, or clear if this is a terminal action for the context
            current_session_context['last_query_results'] = {'ids': ids_to_query, 'data_shown': fields_to_show} # Update context
            current_session_context['last_subject_type'] = 'animal_list_followup'

        elif last_subject == 'animal_detail_added' and last_results and last_results.get('eartag'):
             if "who are its offspring" in parsed_output['original_command'].lower() : # Example specific follow-up
                animal_id = last_results['eartag'] # Assume eartag is the ID for simplicity here
                action_details = f"Conceptual call: data_manager.get_offspring(parent_id='{animal_id}')"
                response = f"Animal {animal_id} currently has no recorded offspring. (Offspring would be listed in 'Animal Detail View' - Pedigree section)"
             else:
                response = "I can provide more details for the last animal added. What would you like to know?"
        else:
            response = "I'm not sure which previous results you're referring to. Could you please clarify?"
            action_details = "Contextual follow-up attempted, but context was unclear or not available."

    elif primary_intent == PI_SYSTEM_COMMAND and subject == SUBJ_TRAIT:
        trait_name = entities.get("TraitName", "[UnknownTrait]")
        action_details = f"Conceptual call: data_manager.add_trait(trait_data={entities})"
        response = f"Okay, I've conceptually defined a new trait: {trait_name} with details {entities}. (Manageable in 'Trait Management View' - ui_conceptualization.md)"


    # Fallback for unhandled intents/subjects
    elif primary_intent == PI_UNKNOWN:
        response = "I'm sorry, I couldn't determine the main purpose of your request. Could you please rephrase?"
        action_details = "Intent could not be determined."
    elif subject == SUBJ_NONE and primary_intent not in [PI_CONTEXTUAL_FOLLOW_UP, PI_CLARIFICATION_RESPONSE]:
        response = f"I understood you want to {primary_intent.lower().replace('_',' ')}, but I'm not sure about which subject (e.g., animal, phenotype). Please specify."
        action_details = f"Intent {primary_intent} understood, but subject is unclear."


    return response, action_details

# --- Main Test Loop (Updated) ---
if __name__ == '__main__':
    print("Enhanced NLP Interface Designer - Conceptual Test\n")

    # Subset of commands from enhanced_nlp_commands.txt + contextual examples
    test_commands = [
        "List all active female animals with birth weight > 4.2 kg and weaning weight between 28kg and 32kg, order by eartag.",
        "For these animals, show their breed and sire IDs.", # Contextual
        "What is the average fleece diameter for Merino ewes born after 2023-01-01?",
        "Add new animal eartag L999 sex male birthdate 2024-05-01 breed Suffolk sire R100 dam E200",
        "Record weaning weight 30.5 kg for animal L999 on 2024-08-15.",
        "What is its current age?", # Contextual, but needs age calculation logic not in this NLP sim
        "Suggest mates for ewe E001 using selection index: 0.6*WW_EBV + 0.4*FW_EBV and keep inbreeding under 3%.",
        "Calculate inbreeding for progeny of sire S005 and dam D015.",
        "Show details for animal S010", # To set context for next command
        "What are its EBVs for fleece weight?", # Contextual on S010
        "Define a new trait: 'Scrotal Circumference', abbreviation SC, unit cm, category 'Reproduction'.",
        "list animals with no recorded weaning weight", # Test ambiguity: Which animals? (should ask for clarification if no context)
        "L001", # Test clarification response
        "list animals with eartag ZZZ", # Test no results found (simulated)
    ]

    for i, command in enumerate(test_commands):
        print(f"--- Turn {i+1} ---")
        print(f"User: {command}")

        parsed = parse_command(command, session_context)
        print(f"Parsed: {parsed}")

        simulated_response, conceptual_action = dispatch_action(parsed, session_context)
        print(f"Action: {conceptual_action}")
        print(f"System: {simulated_response}")
        print(f"Session Context: {session_context}\n")

        # Simulate "no results found" for a specific command to test response
        if "ZZZ" in command:
            session_context['last_query_results'] = {'ids': [], 'count': 0, 'filters': parsed.get('entities')}
            print(f"System (after actual query): No animals found matching your criteria for eartag ZZZ. (This would be reflected in 'Animal List View')")
            print(f"Session Context (updated): {session_context}\n")

```
