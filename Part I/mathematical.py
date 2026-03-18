import math

import pandas as pd


def _numeric_columns(data: pd.DataFrame):
    """
    Return only numeric columns from the DataFrame.

    :param data: The input DataFrame.
    :return: A DataFrame containing only numeric columns.
    """
    return data.select_dtypes(include="number")


def count(data: pd.DataFrame):
    """
    Count non-NaN numeric values for each column.

    :param data: The input DataFrame.
    :return: A Series containing the count of valid numeric values per column.
    """
    numeric_df = _numeric_columns(data)
    results = {}

    for col in numeric_df.columns:
        c = 0
        for x in numeric_df[col]:
            if pd.notna(x):
                c += 1
        results[col] = c

    return pd.Series(results)


def mean(data: pd.DataFrame):
    """
    Compute the arithmetic mean for each numeric column,
    ignoring NaN values.

    :param data: The input DataFrame.
    :return: A Series containing the mean value per column.
             Returns 0 for columns with no valid numeric values.
    """
    numeric_df = _numeric_columns(data)
    results = {}

    for col in numeric_df.columns:
        total = 0
        count_val = 0

        for x in numeric_df[col]:
            if pd.notna(x):
                total += x
                count_val += 1

        if count_val == 0:
            results[col] = 0
        else:
            results[col] = total / count_val

    return pd.Series(results)


def std(data: pd.DataFrame):
    """
    Compute the sample standard deviation for each numeric column,
    ignoring NaN values, to match pandas DataFrame.std().

    Formula (sample std):
        sqrt( sum((x - mean)^2) / (N - 1) )

    :param data: The input DataFrame.
    :return: A Series containing the sample standard deviation per column.
             Returns 0 for columns with fewer than 2 valid numeric values.
    """
    numeric_df = _numeric_columns(data)
    results = {}

    for col in numeric_df.columns:
        values = [x for x in numeric_df[col] if pd.notna(x)]
        n = len(values)

        if n < 2:
            # std undefined for 0 or 1 value → return 0
            results[col] = 0
            continue

        # Compute mean manually
        total = 0
        for v in values:
            total += v
        m = total / n

        # Compute variance manually (sample variance, divide by n-1)
        variance_sum = 0
        for v in values:
            diff = v - m
            variance_sum += diff * diff

        results[col] = math.sqrt(variance_sum / (n - 1))

    return pd.Series(results)


def min(data: pd.DataFrame):
    """
    Compute the minimum value for each numeric column,
    ignoring NaN values.

    :param data: The input DataFrame.
    :return: A Series containing the minimum value per column.
             Returns 0 for columns with no valid numeric values.
    """
    numeric_df = _numeric_columns(data)
    results = {}

    for col in numeric_df.columns:
        min_val = None

        for x in numeric_df[col]:
            if pd.notna(x):
                if min_val is None or x < min_val:
                    min_val = x

        results[col] = 0 if min_val is None else min_val

    return pd.Series(results)


def max(data: pd.DataFrame):
    """
    Compute the maximum value for each numeric column,
    ignoring NaN values.

    :param data: The input DataFrame.
    :return: A Series containing the maximum value per column.
             Returns 0 for columns with no valid numeric values.
    """
    numeric_df = _numeric_columns(data)
    results = {}

    for col in numeric_df.columns:
        max_val = None

        for x in numeric_df[col]:
            if pd.notna(x):
                if max_val is None or x > max_val:
                    max_val = x

        results[col] = 0 if max_val is None else max_val

    return pd.Series(results)


def _percentile(data: pd.DataFrame, q: float):
    """
    Compute a percentile for each numeric column using linear interpolation.

    :param data: The input DataFrame.
    :param q: The percentile to compute (between 0 and 1).
    :return: A Series containing the percentile value per column.
             Returns 0 for columns with no valid numeric values.
    """
    numeric_df = data.select_dtypes(include="number")
    results = {}

    for col in numeric_df.columns:
        # Extract non-NaN values manually
        values = []
        for x in numeric_df[col]:
            if pd.notna(x):
                values.append(x)

        n = len(values)
        if n == 0:
            results[col] = 0
            continue

        # Sort the values
        values.sort()

        # Compute position for percentile
        pos = q * (n - 1)
        lower = int(math.floor(pos))
        upper = int(math.ceil(pos))

        if lower == upper:
            # Exact index, no interpolation needed
            results[col] = values[lower]
        else:
            # Linear interpolation between the two closest values
            weight = pos - lower
            results[col] = (values[lower] +
                            (values[upper] -
                             values[lower]) * weight)

    return pd.Series(results)


def percentile_25(data: pd.DataFrame):
    """
    Compute the 25th percentile (first quartile) for each numeric column.

    :param data: The input DataFrame.
    :return: A Series containing the 25th percentile per column.
    """
    return _percentile(data, 0.25)


def percentile_50(data: pd.DataFrame):
    """
    Compute the 50th percentile (median) for each numeric column.

    :param data: The input DataFrame.
    :return: A Series containing the median per column.
    """
    return _percentile(data, 0.50)


def percentile_75(data: pd.DataFrame):
    """
    Compute the 75th percentile (third quartile) for each numeric column.

    :param data: The input DataFrame.
    :return: A Series containing the 75th percentile per column.
    """
    return _percentile(data, 0.75)


def missing(data: pd.DataFrame):
    """
    Count all "NaN" values for each column.

    :param data: The input DataFrame.
    :return: A series containing the count of NaN values per column.
    """
    numeric_df = _numeric_columns(data)
    results = {}

    for col in numeric_df.columns:
        c = 0
        for x in numeric_df[col]:
            if pd.isna(x):
                c += 1
        results[col] = c

    return pd.Series(results)


def var(data: pd.DataFrame):
    """
    Calculates the variance of each column of the dataset.

    :param data: The input DataFrame.
    :return: A series containing the variance of each column.
    """
    numeric_df = _numeric_columns(data)
    results = {}

    full_count = count(data)
    full_mean = mean(data)

    for col in numeric_df.columns:
        # Verifies if we have more than one value to prevent division by 0
        if full_count[col] <= 1:
            results[col] = float('nan')
            continue

        # Calculate the squared deviation
        sum_deviation = 0
        for x in numeric_df[col]:
            if pd.notna(x):
                sum_deviation += (x - full_mean[col]) ** 2

        results[col] = sum_deviation / (full_count[col] - 1)

    return pd.Series(results)


def iqr(data: pd.DataFrame):
    """
    Calculates the interquartile range of each column of the dataset.

    :param data: The input DataFrame.
    :return: A series containing the iqr of each column.
    """
    return percentile_75(data) - percentile_25(data)


def skew(data: pd.DataFrame):
    """
    Calculates the skewness of each column of the dataset.

    :param data: The input DataFrame.
    :return: A series containing the skewness of each column.
    """
    numeric_df = _numeric_columns(data)
    results = {}

    counts = count(data)
    means = mean(data)
    stds = std(data)

    for col in numeric_df.columns:
        # Verifies if we have more than two values and the std in not null
        # to prevent division by 0
        if counts[col] <= 2 or stds[col] == 0:
            results[col] = float('nan')
            continue

        # Sum of all the cubes
        cube_sum = 0
        for x in numeric_df[col]:
            if pd.notna(x):
                cube_sum += ((x - means[col]) / stds[col]) ** 3

        # Calculate the skewness using the formula
        skewness = ((counts[col] / ((counts[col] - 1) * (counts[col] - 2)))
                    * cube_sum)

        results[col] = skewness

    return pd.Series(results)
