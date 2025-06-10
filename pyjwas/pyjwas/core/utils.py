from typing import List
# Forward declaration for MME if needed for type hinting, assuming it's complex
# from .definitions import MME # This would create circular import if MME uses utils

# To avoid circular import, if MME is truly needed, pass its necessary components
# or use 'Any' type hint. For get_effect_names, we only need mme.model_terms.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .definitions import MME, ModelTerm


def get_effect_names(mme: 'MME') -> List[str]:
    """
    Gets a list of full names for all estimated effects in the MME.
    Corresponds to Julia's getNames(mme::MME).
    Args:
        mme: The MME object.
    Returns:
        A list of strings representing effect names.
    """
    names = []
    for term in mme.model_terms: # Iterate through ModelTerm objects
        for level_name in term.names: # term.names are like "A1", "A2" or "covariate_name"
            # term.trm_str is like "y1:A" or "y1:cov"
            names.append(f"{term.trm_str}:{level_name}")

    # This function currently only processes mme.model_terms.
    # If names for random effects or genotype markers are needed,
    # further logic would be required here, depending on how they are stored
    # and how the solution vector is structured.
    return names
