#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np
    cleaned_data = []
    errors = np.array([abs(p[0] - n[0])
                       for p, n in zip(predictions, net_worths)])
    arg = np.argsort(errors)
    errors = errors[arg]
    ages = ages[arg]
    net_worths = net_worths[arg]
    cleaned_data = [(a, n, e) for a, n, e in zip(ages, net_worths, errors)]

    # your code goes here
    clean_lenght = int(0.9 * len(cleaned_data))
    return cleaned_data[:clean_lenght]
