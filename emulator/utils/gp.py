# Author: Arrykrishna Mootoovaloo
# Collaborators: Alan Heavens, Andrew Jaffe, Florent Leclercq
# Email : a.mootoovaloo17@imperial.ac.uk
# Affiliation : Imperial Centre for Inference and Cosmology
# Status : Under Development

'''
Perform all additional operations such as prediction, interpolation, gradient calculation for GPs
'''


def pred_normal(input_pred: list) -> float:
    '''
    For each GP we have to calculate the mean prediction

    :param: input_pred (list or tuple): array for the test point and whole gp module

    :param: mean_pred (float) : the mean prediction from the GP
    '''

    testpoint, gp = input_pred[0], input_pred[1]

    mean_pred = gp.prediction(testpoint).reshape(1,)

    return mean_pred[0]


def prediction(input_pred: list) -> float:
    '''
    For each GP we have to calculate the mean prediction

    :param: input_pred (list or tuple): array for the test point and whole gp module

    :param: mean_pred (float) : the mean prediction from the GP
    '''

    testpoint, gp = input_pred[0], input_pred[1]

    mean_pred = gp.pred_original_function(testpoint).reshape(1,)

    return mean_pred[0]


def gradient(input_pred: list) -> float:
    '''
    For each GP we have to calculate the mean prediction

    :param: input_pred (list or tuple): array for the test point and whole gp module

    :param: mean_pred (float) : the mean prediction from the GP
    '''

    testpoint, gp, order = input_pred[0], input_pred[1], input_pred[2]

    if order == 1:
        first_der = gp.derivatives(testpoint, order)

        return first_der

    else:
        first_der, second_der = gp.derivatives(testpoint, order)

        return first_der, second_der
