"""Pharmacodynamic response models

Pharmacodynamic response models, including concentration-response,
inhibitor-response, and various receptor-response models that can be used for
empirical fitting of response data.

List of Functions:
    * concentration_response
    * dose_response
    * inhibitor_response
  receptor occupation:
    * hill_langmuir_equation
  receptor-response:
    * hill_equation
    * clark_equation
    * operational_model
    * delcastillo_katz_model
    * buchwald_threeparameter_model
"""

def concentration_response(c, emin, emax, ec50, n):
    """Non-linear (sigmoidal) concentration-response equation.

    Args:
        c (float, numpy.array): The input concentration of an effector in
            concentration units.
        emin (float): The minimimun/baseline response when c=0 in response units.
            Bounds fot fitting: 0 <= emin <= inf
        emax (float): The maximum resonse in response units.
            Bounds fot fitting: 0 <= emax <= inf
        ec50 (float): The concentration corresponding to a half-maximal
            response in concentration units.
            Bounds fot fitting: 0 <= ec50 <= inf
        n (int, float): The Hill coefficient (or Hill slope).
            Bounds for fitting: 0 <= n <= inf

    Returns:
        float, numpy.array : The response for the given concentration(s) in
            response units.
    """
    return emin + (emax-emin) / (1 + (ec50/c)**n)

def dose_response(d, emin, emax, ec50, n):
    """Non-linear (sigmoidal) dose-response equation.

    Note that the dose-response equation is functionally identical to the
    concentration-response response equation but with the (drug) concentration
    replaced with a (drug) dose.

    Args:
        d (float, numpy.array): The input dose in dose units.
        emin (float): The minimimun/baseline response when d=0 in response units.
            Bounds fot fitting: 0 <= emin <= inf
        emax (float): The maximum resonse in response units.
            Bounds fot fitting: 0 <= emax <= inf
        ec50 (float): The dose corresponding to a half-maximal
            response in dose units.
            Bounds fot fitting: 0 <= ec50 <= inf
        n (int, float): The Hill coefficient (or Hill slope).
            Bounds for fitting: 0 <= n <= inf

    Returns:
        float, numpy.array : The response for the given dose(s) in
            response units.
    """
    return concentration_response(d, emin, emax, ec50, n)

def inhibitor_response(ic, emin, emax, ic50, n):
    """Non-linear (sigmoidal) inhibitor-response equation.

    Note that the inhibitor-response equation is functionally identical to the
    concentration-response response equation but with the (agonist)
    concentration replaced with an inhibitor concentration and a negative Hill
    coefficient.

    Args:
        ic (float, numpy.array): The inhibitor concentration in concentration
            units.
        emin (float): The minimimun response value to which the respsonse
            can be reduced to by the inhibitor; emin is in response units.
            Bounds fot fitting: 0 <= emin <= inf
        emax (float): The maximum/baseline resonse when ic=0 in response units.
            Bounds fot fitting: 0 <= emax <= inf
        ic50 (float): The inhibitor concentration corresponding to a
            half-maximal (halway between emax and emin) response in
            concentration units.
            Bounds fot fitting: 0 <= ic50 <= inf
        n (int, float): The Hill coefficient (or Hill slope).
            Bounds for fitting: 0 <= n <= inf

    Returns:
        float, numpy.array : The response for the given inhibitor
        concentration(s) in response units.
    """
    return concentration_response(ic, emin, emax, ic50, -n)

def hill_langmuir_equation(l, kd):
    """Hill-Langmuir receptor occupation equation.

    Args:
        l (float, numpy.array): The input concentration of an ligand in
            concentration units.
        kd (float): The ligand-receptor dissociation constant (or its
            effective value) in concentration units.
                Bounds fot fitting: 0 <= kd <= inf

    Returns:
        float, numpy.array : The fractional receptor occupation for the given
            ligand concentration; unitless, range [0,1].
    """
    return  l / (l + kd)

def hill_equation(l, emax, kd, n):
    """Hill receptor-response equation.

    Args:
        l (float, numpy.array): The input concentration of an ligand in
            concentration units.
        emax (float): The maximum response in response units.
            Bounds fot fitting: 0 <= emax <= inf
        kd (float): The ligand-receptor dissociation constant (or its
            effective value) in concentration units.
            Bounds fot fitting: 0 <= kd <= inf
        n (int, float): The Hill coefficient (or Hill slope).
            Bounds for fitting: 0 <= n <= inf

    Returns:
        float, numpy.array : The response for the given ligand concentration(s)
            in response units.
    """
    return emax * l**n / (l**n + kd**n)

def clark_equation(l, emax, kd):
    """Clark equation for receptor-response.

    The Clark equation corresponds to single-state receptor activation model
    with a linear effect response:
        L + R <--kd--> LR* ---> Effect,
    such that Effect is directly proportional to the receptor occupation LR*.
    Note that the Clark equation is equivalent to the Hill receptor-response
    equation with n = 1.

    Args:
        l (float, numpy.array): The input concentration of an ligand in
            concentration units.
        emax (float): The maximum response in response units.
            Bounds fot fitting: 0 <= emax <= inf
        kd (float): The ligand-receptor dissociation constant in concentration
            units. Bounds fot fitting: 0 <= kd <= inf

    Returns:
        float, numpy.array : The response for the given ligand concentration(s)
            in response units.

    References:
        1. Clark, A.J., 1926. The reaction between acetyl choline and muscle
            cells. The Journal of physiology, 61(4), pp.530-546.
            https://doi.org/10.1113/jphysiol.1926.sp002314
        2. Buchwald, P., 2017. A three‐parameter two‐state model of receptor
            function that incorporates affinity, efficacy, and signal
            amplification. Pharmacology research & perspectives, 5(3),
            p.e00311. https://doi.org/10.1002/prp2.311
    """
    return emax * l / (l + kd)

def operational_model(l, emax, kd, tau):
    """Operational (Black-Leff) model of receptor-response.
    The operational model corresponds to single-state receptor activation model
    with a (non-linear) hyperbolic effect response:
        L + R <--kd--> LR* --tau-> Effect,
    where the agonist ligand's effect is amplified to some extent, which is
    accounted for by the efficacy term tau.
    Note that the observed/effective dissociation constant and Emax values are:
        Kobs = kd / (tau + 1)
        Emax_obs = (emax * tau) / (tau + 1)    

    Args:
        l (float, numpy.array): The input concentration of an ligand in
            concentration units.
        emax (float): The maximum response in response units.
            Bounds fot fitting: 0 <= emax <= inf
        kd (float): The ligand-receptor dissociation constant (or its
            effective value) in concentration units.
            Bounds fot fitting: 0 <= kd <= inf
        tau (float): Efficacy of the agonist ligand.
            Bounds for fitting: 0 <= tau <= inf

    Returns:
        float, numpy.array : The response for the given ligand concentration(s)
            in response units.

    References:
        1. Black, J.W. and Leff, P., 1983. Operational models of
            pharmacological agonism. Proceedings of the Royal society of London.
            Series B. Biological sciences, 220(1219), pp.141-162.
            https://doi.org/10.1098/rspb.1983.0093
        2. Buchwald, P., 2017. A three‐parameter two‐state model of receptor
            function that incorporates affinity, efficacy, and signal
            amplification. Pharmacology research & perspectives, 5(3),
            p.e00311. https://doi.org/10.1002/prp2.311
    """
    return emax * tau * l / ((tau + 1) * l + kd)

def delcastillo_katz_model(l, emax, kd, tau):
    """Del Castillo-Katz model of receptor-response.
    The Del Castillo-Katz model corresponds to a two-state receptor activation
    model with a linear effect response:
        L + R <--kd--> LR <--tau--> LR* ---> Effect,
    Here tau represents the ratio of active receptor-complex to inactive
    receptor-complex: tau = [LR*]/[LR]

    Although the underlying receptor activation model and definition of tau
    is different, the del Castillo-Katz model is functionally identical to the
    Operational model and tau can still be thought of as encoding
    the ligand's efficacy.
    Note that the observed/effective dissociation constant and Emax values are:
        Kobs = kd / (tau + 1)
        Emax_obs = (emax * tau) / (tau + 1)

    Args:
        l (float, numpy.array): The input concentration of an ligand in
            concentration units.
        emax (float): The maximum response in response units.
            Bounds fot fitting: 0 <= emax <= inf
        kd (float): The ligand-receptor dissociation constant (or its
            effective value) in concentration units.
            Bounds fot fitting: 0 <= kd <= inf
        tau (float): Efficacy of the agonist ligand.
            Bounds for fitting: 0 <= tau <= inf

    Returns:
        float, numpy.array : The response for the given ligand concentration(s)
            in response units.

    References:
        1. Castillo, J.D. and Katz, B., 1957. Interaction at end-plate
            receptors between different choline derivatives. Proceedings of the
            Royal Society of London. Series B-Biological Sciences, 146(924),
            pp.369-381. https://doi.org/10.1098/rspb.1957.0018
        2. Buchwald, P., 2017. A three‐parameter two‐state model of receptor
            function that incorporates affinity, efficacy, and signal
            amplification. Pharmacology research & perspectives, 5(3),
            p.e00311. https://doi.org/10.1002/prp2.311
    """
    return emax * tau * l / ((tau + 1) * l + kd)

def buchwald_threeparameter_model(l, emax, kd, epsilon, gamma):
    """Three-parameter two-state model of receptor-response by Buchwald.
    The three-parameter two-state model of receptor-response by Buchwald
    corresponds to a two-state receptor activation model that allows for
    non-linear response via efficacy and amplification:
        L + R <--kd--> LR <--epsilon--> LR* --gamma--> Effect,
    Here epsilon encodes the efficacy of the ligand (similar to tau in the
    del Castillo-Katz model) and gamma encodes amplification of
    the ligand's effect. Note than when epsilon=1 and gamma=1 this model
    reduces to the Clark equation.
    Note that the observed/effective dissociation constant and Emax values are:
        Kobs = kd / (epsilon*gamma + 1 - epsilon)
        Emax_obs = (emax * epsilon * gamma) / (epsilon*gamma + 1 - epsilon)

    Args:
        l (float, numpy.array): The input concentration of an ligand in
            concentration units.
        emax (float): The maximum response in response units.
            Bounds fot fitting: 0 <= emax <= inf
        kd (float): The ligand-receptor dissociation constant (or its
            effective value) in concentration units.
            Bounds fot fitting: 0 <= kd <= inf
        epsilon (float): Efficacy of the agonist ligand.
            Bounds fot fitting: 0 <= kd <= 1
        gamma (float): Amplification of agonist effect.
            Bounds for fitting: 1 <= gamma <= inf

    Returns:
        float, numpy.array : The response for the given ligand concentration(s)
            in response units.

    References:
        1. Buchwald, P., 2017. A three‐parameter two‐state model of receptor
            function that incorporates affinity, efficacy, and signal
            amplification. Pharmacology research & perspectives, 5(3),
            p.e00311. https://doi.org/10.1002/prp2.311

    """
    return emax * epsilon * gamma * l / ((epsilon*gamma + 1 - epsilon) * l + kd)
