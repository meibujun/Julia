import math

def calculate_mean(data_list: list[float]) -> float:
    """
    Calculates the arithmetic mean of a list of numbers.

    Args:
        data_list: A list of numbers (float or int).

    Returns:
        The arithmetic mean of the numbers in the list.

    Raises:
        ValueError: If the input list is empty.
    """
    if not data_list:
        raise ValueError("Input list cannot be empty to calculate mean.")
    return sum(data_list) / len(data_list)

def calculate_variance(data_list: list[float]) -> float:
    """
    Calculates the sample variance of a list of numbers.
    Uses N-1 in the denominator (ddof=1).

    Args:
        data_list: A list of numbers (float or int).

    Returns:
        The sample variance of the numbers in the list.

    Raises:
        ValueError: If the input list has fewer than 2 elements.
    """
    n = len(data_list)
    if n < 2:
        raise ValueError("Input list must contain at least two elements to calculate sample variance.")

    mean = calculate_mean(data_list)
    sum_squared_diff = sum((x - mean) ** 2 for x in data_list)
    return sum_squared_diff / (n - 1)

def calculate_std_dev(data_list: list[float]) -> float:
    """
    Calculates the sample standard deviation of a list of numbers.
    This is the square root of the sample variance.

    Args:
        data_list: A list of numbers (float or int).

    Returns:
        The sample standard deviation of the numbers in the list.

    Raises:
        ValueError: If the input list has fewer than 2 elements.
    """
    if len(data_list) < 2:
        # Error message consistent with calculate_variance for the underlying reason
        raise ValueError("Input list must contain at least two elements to calculate sample standard deviation.")

    variance = calculate_variance(data_list)
    return math.sqrt(variance)
