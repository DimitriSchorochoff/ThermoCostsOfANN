__author__ = "Schorochoff Dimitri"
__version__ = "1.1"

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers
import numpy as np
from tqdm import tqdm

_K = 1.380649 * 10 ** (-23)
_J2EV = 1 / (1.602176634 * 10 ** (-19))
_ROOMTEMP = 293.15
_ENTROPY2EV = _K * _ROOMTEMP * _J2EV


#########################################################################
###                        MAIN FUNCTIONS                          ###
#########################################################################

def samples2Landauer(model, inputs, bin_size=0.01, verbose=True):
    """
    samples2Landauer(model = keras.Sequential([layers.Input((n1)),layers.Dense(n2),]),
     (s1, s2, ...), bin_size=0.01)

    Compute the total Landauer cost in eV that yield a basic keras model
    when evaluating an input of a given distribution.

    Parameters
    ----------
    model : a tf.keras.Model class
        the model over which the Landauer cost is computed. It must only be composed of
        tf.keras.layers.Dense, keras.layers.Conv2D, tf.keras.layers.MaxPooling2D,
        tf.keras.layers.AveragePooling2D, tf.keras.layers.Flatten and/or
        tf.keras.layers.Reshape layers
    inputs : ndarray
        An arrays of samples for the given distribution. Each sample s1, s2,... must have
        the same shape as the model input.
    bin_size : float, optional
        Define the size of the bins that are used to compute discretized entropy.
    verbose: boolean, optional
        Define whether or not to show a progressbar

    Returns
    -------
    res : float
        The total Landauer cost.
    """

    h = samples2LandauerPerNeuron(model, inputs, bin_size=bin_size, verbose=verbose)
    return sum_matrix(h, get_shapes(model))


def samples2Mismatch(model, inputs, opti_inputs, bin_size=0.01, use_approx=True, verbose=True):
    """
    samples2Mismatch(model = keras.Sequential([layers.Input((n1)),layers.Dense(n2),]),
     (s1, s2, ...), (q1, q2, ...), bin_size=0.01)

    Compute the total Mismatch cost in eV that yield a basic keras model
    when evaluating an input of a given distribution if the entropy production
    is minimized by a given optimal distribution.

    Parameters
    ----------
    model : a tf.keras.Model class
        the model over which the Mismatch cost is computed. It must only be composed of
        tf.keras.layers.Dense, keras.layers.Conv2D, tf.keras.layers.MaxPooling2D,
        tf.keras.layers.AveragePooling2D, tf.keras.layers.Flatten and/or
        tf.keras.layers.Reshape layers
    inputs : ndarray
        An array of samples for the given distribution. Each sample s1, s2,...
        must have the same shape as the model input.
    opti_inputs : ndarray
        An array of samples for the given optimal distribution. Each sample q1, q2,...
        must have the same shape as the model input.
    bin_size : float, optional
        Define the size of the bins that are used to compute discretized entropy.
    use_approx: bool, optional
        If True, approximate the cost by considering every distribution as independant.
    verbose: boolean, optional
        Define whether or not to show a progressbar
    Returns
    -------
    res : float
        The total Mismatch cost.
    """
    h = samples2MismatchPerNeuron(model, inputs, opti_inputs, bin_size=bin_size, use_approx=use_approx, verbose=verbose)
    return sum_matrix(h, get_shapes(model))


def samples2LandauerPerNeuron(model, inputs, bin_size=0.01, verbose=True):
    """
    Compute the Landauer cost in eV of each neuron that yield a basic keras model
    when evaluating an input of a given distribution.

    Parameters
    ----------
    See documentation of samples2Landauer

    Returns
    -------
    res : ndarray
        An array (l1,l2,...) containing the Landauer cost of each neuron.
        l1 has the same shape as layer 1 and so on.
    """
    s = get_shapes(model)
    h = _samples2Histos(model, inputs, bin_size=bin_size, verbose=verbose)
    _histos2Entropy(h, s, entropy_func=CAE_entropy)
    _entropy2Landauer(model, h)

    return h


def samples2MismatchPerNeuron(model, inputs, opti_inputs, bin_size=0.01, use_approx=True, verbose=True):
    """
    Compute the Mismatch cost in eV of each neuron that yield a basic keras model
    when evaluating an input of a given distribution.

    Parameters
    ----------
    See documentation of samples2Mismatch

    Returns
    -------
    res : ndarray
        An array (l1,l2,...) containing the Mismatch cost of each neuron.
        l1 has the same shape as layer 1 and so on.
    """
    s = get_shapes(model)
    # Compute the KL divergence of final neuron
    h_nocorr = _samples2Histos(model, inputs, bin_size=bin_size, verbose=verbose)
    opti_h_nocorr = _samples2Histos(model, opti_inputs, bin_size=bin_size, verbose=verbose)
    _histos2KLDivergence(h_nocorr, opti_h_nocorr, s)

    if use_approx:
        _entropy2Landauer(model, opti_h_nocorr)
        opti_h_nocorr[0] = []  # Input layer has no Mismatch cost
        return opti_h_nocorr

    else:
        # Compute KL divergence of the set initial neurons if they are dependant
        h = _samples2HistosCorrelated(model, inputs, bin_size=bin_size)
        opti_h = _samples2HistosCorrelated(model, opti_inputs, bin_size=bin_size)
        _histos2KLDivergence(h, opti_h, s)

        # Compute the Mismatch cost from KL divergence
        map_matrixes(opti_h, opti_h_nocorr, s, _KL2Mismatch)

        opti_h[0] = []  # Input layer has no Mismatch cost

        return opti_h


def heuristicLandauer(model, std):
    """
    Compute estimation based on a heuristic of the Landauer cost of each layer in eV
    that yields a basic keras model when the input follows a Normal distribution.

    Parameters
    ----------
    model : a tf.keras.Model class
        the model over which the Landauer cost is computed. It must only be composed of
        tf.keras.layers.Dense, keras.layers.Conv2D, tf.keras.layers.MaxPooling2D,
        tf.keras.layers.AveragePooling2D, tf.keras.layers.Flatten and/or
        tf.keras.layers.Reshape layers
    std : float
        The standard deviation of the input Normal distribution

    Returns
    -------
    res : ndarray
        The Landauer cost in eV of each layer.
    """
    costs = [0]
    S_Normal = entropy_Normal(std)
    shapes = get_shapes(model)
    for i in range(len(model.layers)):
        previousShape = shapes[i]
        shape = shapes[i + 1]
        layer = model.layers[i]

        if isinstance(layer, tf.keras.layers.Dense):
            costs.append(shape[1]*previousShape[1]*S_Normal*_ENTROPY2EV)

        elif isinstance(layer, tf.keras.layers.Conv2D):
            K = layer.kernel_size[0]
            costs.append(shape[3] * (previousShape[1]-K+1)**2 * K**2 * S_Normal * _ENTROPY2EV)

        elif isinstance(layer, tf.keras.layers.MaxPooling2D) or\
                isinstance(layer, tf.keras.layers.AveragePooling2D):
            P = layer.pool_size[0]
            costs.append(previousShape[1]*previousShape[2]//(P**2) * (P**2 - 1/P)* S_Normal * _ENTROPY2EV)

        elif isinstance(layer, tf.keras.layers.Flatten) or\
                isinstance(layer, tf.keras.layers.Reshape):
            costs.append(0)

    return costs

def heuristicMismatch(model, mean, std, mean_opt, std_opt):
    """
    Compute estimation based on a heuristic of the Landauer cost of each layer in eV
    that yields a basic keras model when the input follows a Normal distribution.

    Parameters
    ----------
    model : a tf.keras.Model class
        the model over which the Landauer cost is computed. It must only be composed of
        tf.keras.layers.Dense, keras.layers.Conv2D, tf.keras.layers.MaxPooling2D,
        tf.keras.layers.AveragePooling2D, tf.keras.layers.Flatten and/or
        tf.keras.layers.Reshape layers
    mean : float
        The mean of the input Normal distribution
    std : float
        The standard deviation of the input Normal distribution
    mean_opt: float
        The mean of the optimal Normal distribution
    std_opt: float
        The standard deviation of the optimal Normal distribution

    Returns
    -------
    res : ndarray
        The Mismatch cost in eV of each layer.
    """
    costs = [0]
    D_Normal = KL_divergence_Normal(mean, std, mean_opt, std_opt)
    shapes = get_shapes(model)
    for i in range(len(model.layers)):
        previousShape = shapes[i]
        shape = shapes[i + 1]
        layer = model.layers[i]

        if isinstance(layer, tf.keras.layers.Dense):
            costs.append(shape[1]*previousShape[1]*D_Normal*_ENTROPY2EV)

        elif isinstance(layer, tf.keras.layers.Conv2D):
            K = layer.kernel_size[0]
            costs.append(shape[3] * (previousShape[1]-K+1)**2 * K**2 * D_Normal * _ENTROPY2EV)

        elif isinstance(layer, tf.keras.layers.MaxPooling2D) or\
                isinstance(layer, tf.keras.layers.AveragePooling2D):
            P = layer.pool_size[0]
            costs.append(previousShape[1]*previousShape[2]//(P**2) * (P**2 - 1/P)* D_Normal * _ENTROPY2EV/3)

        elif isinstance(layer, tf.keras.layers.Flatten) or\
                isinstance(layer, tf.keras.layers.Reshape):
            costs.append(0)

    return costs


#########################################################################
###                        ENTROPY FUNCTIONS                          ###
#########################################################################

def CAE_entropy(counts):
    """
    This is a Python translation of
    https://github.com/cran/entropy/blob/master/R/entropy.ChaoShen.R

    Return the CAE entropy of a histogram.

    Parameters
    ----------
    counts: ndarray. A list containing the number of occurence of every bin

    Returns
    -------
    res : float
    The CAE entropy of the histogram
    """
    counts = counts[counts > 0]
    n = np.sum(counts)
    if (n == 0): return 0
    p = counts / n

    f1 = np.count_nonzero(counts == 1)
    if (f1 == n): f1 = n - 1

    C = 1 - f1 / n
    pa = C * p
    la = (1 - (1 - pa) ** n)

    return -np.sum(pa * np.log(pa) / la)


def KL_divergence(countP, countQ):
    """
    Return the KL divergence between two given histograms.

    Parameters
    ----------
    countP,countQ: ndarray. A list containing the number of occurence of every bin

    Returns
    -------
    res : float
    The KL divergence between the two given histograms.
    """
    p = countP[countP > 0]
    p = p / np.sum(countP)

    q = countQ[countQ > 0]
    q = q / np.sum(countQ)

    return np.sum(p * np.log(p / q))

def _KL2Mismatch(KL_i, KL_f):
    return (KL_i - KL_f) * _ENTROPY2EV

def entropy_Normal(sigma):
    """
    Return the analytical entropy of a Normal distribution of standard deviation sigma
    """
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

def KL_divergence_Normal(mu1, sigma1, mu2, sigma2):
    """
    Return the analytical KL divergence between a Normal distribution of mean mu1 and
    standard deviation sigma1 and another Normal distribution of mean mu2 and standard deviation
    sigma2
    """
    return np.log(sigma2/sigma1) + (sigma1**2+ (mu2 - mu1)**2)/(2*sigma2**2) - 0.5

#########################################################################
###                  MULTIDIMENSION ARRAY FUNCTIONS                   ###
#########################################################################

def get_shapes(model):
    """
    Give an array of shapes corresponding to the layers of a given model

    Parameters
    ----------
    model : a tf.keras.Model class

    Returns
    -------
    res : ndarray
        Return an array (s1,s2,...) where s1 is a tuple containing the shape of layer 1.
        The input layer is counted as the first layer
    """
    s = [(1, *(model.input.shape[1:]))]
    for layer in model.layers:
        s.append((1, *(layer.output_shape[1:])))

    return s


def index_multidim(shape):
    """
    Give a generator giving all the index needed to iterate over
    a multidimensional array of the given shape

    Parameters
    ----------
    shape : a tuple (d1,d2,...) where d1 correspond to the first dimension and so on.

    Returns
    -------
    res : generator
        Return a generator giving d1*d2*... indexes.
        An index is a tuple of the lenght as the shape.
    """

    def index_multidim_recursive(x, t):
        if (len(t) == 0):
            yield x
            return

        for i in range(t[0]):
            yield from index_multidim_recursive((*x, i), t[1:])

    yield from index_multidim_recursive([], shape)


def get_matrix(matrix, ind):
    """
    Return a value stored at a specific index of a multidimensional array

    Parameters
    ----------
    matrix : a multidimensional matrix having a depth of N.

    ind : a tuple of length N indicating which part of matrix should be returned

    Returns
    -------
    res : value
        Return the value stored in the array the type depends on what is stored
    """
    for i in ind:
        matrix = matrix[i]
    return matrix


def set_matrix(matrix, ind, v):
    """
    Store a value at a specific index of a multidimensional array

    Parameters
    ----------
    matrix : a multidimensional array having a depth of N.

    ind : a tuple of length N indicating which part of matrix should be returned

    v : the value to store
    """
    for i in ind[:-1]:
        matrix = matrix[i]
    matrix[ind[-1]] = v


def sum_matrix(matrixes, shapes):
    """
    Return the sum of all values contained inside a list of multidimensional arrays

    Parameters
    ----------
    matrixes : a list of N multidimensional array having depth of (d1,d2,...,dN).
    Arrays must only contains float/int

    shapes : a list of N tuples. The first tuples gives the shape of
    the first multidimensional array and must therefore have length d1.

    Returns
    -------
    res : float/int
        Return sum of all values stored in matrixes
    """
    res = 0
    for i in range(len(matrixes)):
        if len(matrixes[i]) == 0: continue

        for ind in index_multidim(shapes[i]):
            res += get_matrix(matrixes[i], ind)

    return res


def map_matrixes(matrixes1, matrixes2, shapes, map):
    """
    Iterate over every index of a given list of shape. For every index store
    and apply the operation map

    Parameters
    ----------
    matrixes1: a list of N multidimensional array having depth of (d1,d2,...,dN).
    Arrays must only contains float/int

    matrixes2: same as matrixes1

    shapes : a list of N tuples. The first tuples gives the shape of
    the first multidimensional array and must therefore have length d1.

    map : func. Mapping function applied.
    It must take 2 float/int as argument

    Returns
    -------
    Store result of the mapping in matrixes1
    """
    if matrixes1 == [] or matrixes2 == []: return []

    for i in range(len(matrixes1)):
        if len(matrixes1[i]) == 0 or len(matrixes2[i]) == 0: continue
        for ind in index_multidim(shapes[i]):
            v = map(get_matrix(matrixes1[i], ind), get_matrix(matrixes2[i], ind))
            set_matrix(matrixes1[i], ind, v)


#########################################################################
###                     HISTOGRAM FUNCTIONS                           ###
#########################################################################

def _init_histogram_matrix(shapes):
    """
    Initialize a list of multidimensional array of given shapes
    filled with empy dictionary

    Parameters
    ----------
    shapes: a list of N tuples.
    The first tuple gives the shape of the first array and so on

    Returns
    -------
    res : ndarray
        Return a list of N multidimensional arrays with shape
        corresponding to shapes
    """
    lst = [np.full(s, {}) for s in shapes]

    for i in range(len(lst)):
        for ind in index_multidim(shapes[i]):
            # Otherwise all dictionary have the same ref
            set_matrix(lst[i], ind, {})

    return lst

def _add2histogram(histo, datapoint, bin_size=0.01):
    """
    Add a datapoint to a given histogram.
    Convert the value of the datapoint to its corresponding bin number
    Bounds of bin number i are [ (i-0.5)*bin_size, (i+0.5)*bin_size )

    Parameters
    ----------
    histo: dictionnary. The histogram to add the datapoint.

    datapoint: tuple. The datapoint (x1,x2,...) to add.
    x1 correspond to it's position in the first dimension and so on.

    bin_size : float, optional
    Define the size of the bins that are used to compute discretized entropy.
    """
    position = tuple([round(d / bin_size) for d in datapoint])

    if position in histo:
        histo[position] += 1
    else:
        histo[position] = 1

def _add2layer_histo(histos, shape, data, bin_size=0.01):
    """
    Iterate through every histogram in a multidimensional array and
    add the corresponding datapoint in another multidim array

    Parameters
    ----------
    histos: ndarray. A multidimensional array containing the histograms (dictionary)

    shape: tuple. The shape of the multidimensional array

    data: ndarray. A multidimensional array of datapoint to add.
    In datapoint (x1,x2,...) x1 correspond to its position in the first dimension and so on.

    bin_size : float, optional
    Define the size of the bins that are used to compute discretized entropy.
    """
    for ind in index_multidim(shape):
        _add2histogram(get_matrix(histos, ind), (get_matrix(data, ind),), bin_size=bin_size)


def _match_histos(histo1, histo2):
    """
    Convert two histograms from dictionary format to array format.
    Keep only the keys that match and align the index of their corresponding
    value in the arrays

    Parameters
    ----------
    histo1: dictionnary. The first histogram to match

    histo2: dictionnary. The second histogram to match

    Returns
    -------
    res1 : ndarray
        Return a histogram corresponding to histo1 as array format.
        The value at index i is the number of occurence of key K_i

    res2 : ndarray
        Same as res1 except it is corresponding to histo2
    """

    inter = list(set(histo1.keys()).intersection(set(histo2.keys())))

    newH1 = np.zeros(len(inter))
    newH2 = np.zeros(len(inter))

    for k in range(len(inter)):
        newH1[k] = histo1[inter[k]]
        newH2[k] = histo2[inter[k]]

    return newH1, newH2

#########################################################################
###                     PIPELINE FUNCTIONS                            ###
#########################################################################

def _input_outputs_func(model):
    """
    Code inspired from
    https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer

    Return a function f corresponding to a keras model.

    Parameters
    ----------
    model : a keras model. The model you want to know the neuron's output of.

    Returns
    -------
    res : a function having the following specification

        Function f(input)
        This function is based of a keras model M from an input
        it return the output of each neuron.

        Parameters
        ----------
        input : ndarray. The input of the model you want to know the neurons's output of.

        Returns
        -------
        res : ndarray
        An array (l1,l2,...) containing the output (float) of each neuron.
        l1 has the same shape as layer 1 and so on.
    """
    inp = model.input  # input placeholder
    outputs = [model.input] + [layer.output for layer in model.layers]
    functor = K.function([inp], outputs)  # evaluation function

    return functor


def _samples2Histos(model, inputs, bin_size=0.01, verbose=True):
    """
    Run a set of samples through a keras model and store for each neuron a histogram
    storing the occurence of the neuron output in each bin.

    Parameters
    ----------
    See documentation of samples2Landauer

    Returns
    -------
    res : ndarray
        An array (l1,l2,...) containing a dictionary of each neuron.
        l1 has the same shape as layer 1 and so on. Each key of a dictionary is
        a bin number and the corresponding value is the number of samples that
        felt within the bounds of the bin.
        Bounds of bin number i are [ (i-0.5)*bin_size, (i+0.5)*bin_size )
    """
    shapes = get_shapes(model)
    histos = _init_histogram_matrix(shapes)
    functor = _input_outputs_func(model)

    with tqdm(total=len(inputs), disable= not verbose) as pbar:
        for i in range(len(inputs)):
            inp = inputs[i]
            out = functor([np.array([inp])])

            for i in range(len(out)):
                _add2layer_histo(histos[i], shapes[i], out[i], bin_size=bin_size)

            if verbose: pbar.update(1)

    return histos


def _samples2HistosCorrelated(model, inputs, bin_size=0.01, verbose=True):
    """
    Run a set of samples through a keras model and store for each neuron a histogram
    storing the occurence of the set of linked initial neurons's output in
    each multidimensional bin.

    Parameters
    ----------
    See documentation of samples2Landauer

    Returns
    -------
    res : ndarray
        An array (l1,l2,...) containing a dictionary of each neuron.
        l1 has the same shape as layer 1 and so on. Each key of a dictionary is
        a tuple of bin number (b1,b2,...). b1 is the bin number corresponding to
        the output of the first initial neuron. The value corresponding to the key
        is the number ofsamples that felt within the bounds of this set of bins.
        Bounds of bin number i are [ (i-0.5)*bin_size, (i+0.5)*bin_size )

        Note: l1 is set to [] has it has no prior initial neurons.
    """
    shapes = get_shapes(model)
    histos = _init_histogram_matrix(shapes)
    functor = _input_outputs_func(model)

    with tqdm(total=len(inputs), disable=not verbose) as pbar:
        for k in range(len(inputs)):  # Iterate through all sample
            inp = inputs[k]
            out = functor([np.array([inp])])  # Compute the neuron output in the model

            for i in range(len(model.layers) - 1, -1, -1):  # Iterate through all layers
                previousShape = shapes[i]
                shape = shapes[i + 1]
                layer = model.layers[i]

                if isinstance(layer, tf.keras.layers.Dense):  # Check the layer type
                    for ind in index_multidim(shape):  # Iterate through all final neuron
                        prev_result = []

                        # Iterate through all linked initial neuron
                        # Add their output to a list
                        for prev_index in index_multidim(previousShape):
                            prev_result.append(get_matrix(out[i], prev_index))

                        # Add the list of output to the histogram
                        _add2histogram(get_matrix(histos[i + 1], ind), tuple(prev_result),
                                       bin_size=bin_size)

                elif isinstance(layer, tf.keras.layers.Conv2D):
                    kernel = layer.kernel_size
                    stride = layer.strides

                    for ind in index_multidim(shape):
                        prev_result = []

                        for n_filter in range(previousShape[3]):  # All filter index
                            for m in range(kernel[0]):  # 2nd dim of the kernel
                                for n in range(kernel[1]):  # 1st dim of the kernel
                                    prev_index = (ind[0], ind[1] * stride[0] + m,
                                                  ind[2] * stride[1] + n, n_filter)
                                    prev_result.append(get_matrix(out[i], prev_index))

                        _add2histogram(get_matrix(histos[i + 1], ind), tuple(prev_result),
                                       bin_size=bin_size)

                elif isinstance(layer, tf.keras.layers.MaxPooling2D) or \
                        isinstance(layer,tf.keras.layers.AveragePooling2D):
                    # Main difference with convolution: only one filter in previous layer
                    kernel = layer.pool_size
                    stride = layer.strides

                    for ind in index_multidim(shape):
                        prev_result = []

                        for m in range(kernel[0]):
                            for n in range(kernel[1]):
                                prev_index = (ind[0], ind[1] * stride[0] + m,
                                              ind[2] * stride[1] + n, ind[3])
                                prev_result.append(get_matrix(out[i], prev_index))

                        _add2histogram(get_matrix(histos[i + 1], ind), tuple(prev_result),
                                       bin_size=bin_size)

                elif isinstance(layer, tf.keras.layers.Flatten) or\
                        isinstance(layer, tf.keras.layers.Reshape):
                    pass  # There is no initial neuron when reshapping

            if verbose: pbar.update(1)

    histos[0] = []  # Input layer has no prior initial neuron
    return histos


def _histos2Entropy(histos, shapes, entropy_func=CAE_entropy, bin_size=0.01, epsilon=0):
    """
    Convert every histogram of a keras model contained in a multidimensional array
    to their respective entropy

    Parameters
    ----------
    histos: ndarray. Often the output of _samples2Histos().
    An array (l1,l2,...) containing a dictionary of each neuron of a keras model.
    l1 has the same shape as layer 1 and so on. Each key of a dictionary is
    a bin number and the corresponding value is the number of samples that felt
    within the bounds of the bin.

    shapes: a list of N tuples. The first tuples gives the shapes of l1 and so on.

    entropy_func: func. An entropy function with specification as below

        Entropy function f(counts)

        Return the entropy of a histogram.

        Parameters
        ----------
        counts: ndarray. A list containing the number of occurence of every bin

        Returns
        -------
        res : float
        The entropy of the histogram

    bin_size : float, optional
    Define the size of the bins that are used to compute discretized entropy.

    epsilon: float, optional
    Define the threshold under which entropy is considered as degenerate distribution's entropy
    In this case we don't add the correction factor as it would make estimation worth by default
    we don't use this estimation

    Returns
    -------
    res : ndarray
        An array (l1,l2,...) containing the entropy (float) of each neuron.
        l1 has the same shape as layer 1 and so on.

    Note: the entropy provided still need to be adjusted by substracting log(bin_size)
    to it. This isn't done here as this often cancel itself when computing
    the Landauer cost.
    """
    for i in range(len(histos)):
        for ind in index_multidim(shapes[i]):
            histo = get_matrix(histos[i], ind)

            entropy = entropy_func(np.array(list(histo.values())))
            if np.abs(entropy) >= epsilon: entropy += np.log(bin_size)
            set_matrix(histos[i], ind, entropy)


def _histos2KLDivergence(histos1, histos2, shapes):
    """
    Compare and compute the KL divergence of every histogram of
    two multidimensional arrays. Store the result in histos2

    Parameters
    ----------
    histos1: ndarray. Often the output of _samples2HistosCorrelated().
    An array (l1,l2,...) containing a dictionary of each neuron.
    l1 has the same shape as layer 1 and so on. Each key of a dictionary is a tuple of
    bin number (b1,b2,...). b1 is the bin number corresponding to the output of the
    first initial neuron. The value corresponding to the key is the number of samples
    that felt within the bounds of this set bins.

    histos2: ndarray. Similar as histos1 however the distribution sampled
    by the histograms might differ.

    shapes: a list of N tuples. The first tuples gives the shapes of l1 and so on.

    Returns
    -------
    For every histogram in histos2, store the KL divergence between the histogram
    at same index in histos1 and this histogram.
    """
    for i in range(len(histos1)):
        if len(histos1[i]) == 0 or len(histos2[i]) == 0: continue
        for ind in index_multidim(shapes[i]):

            histo1 = get_matrix(histos1[i], ind)
            histo2 = get_matrix(histos2[i], ind)

            histo1, histo2 = _match_histos(histo1, histo2)
            divergence = KL_divergence(histo1, histo2)
            set_matrix(histos2[i], ind, divergence)


def _entropy2Landauer(model, entropies):
    """
    Convert every entropy of a keras model contained in a multidimensional array
    to their respective Landauer costs

    Parameters
    ----------
    model : a tf.keras.Model class
        the model over which the Landauer cost is computed. It must only be composed of
        tf.keras.layers.Dense, keras.layers.Conv2D, tf.keras.layers.MaxPooling2D,
        tf.keras.layers.AveragePooling2D, tf.keras.layers.Flatten and/or
         tf.keras.layers.Reshape layers

    entropies: ndarray. Often the output of _histos2Entropy().
    An array (l1,l2,...) containing the entropy (float) of each neuron of a keras model.
    l1 has the same shape as layer 1 and so on.

    Returns
    -------
    res : ndarray
        An array (l1,l2,...) containing the Landauer cost (float) of each neuron.
        l1 has the same shape as layer 1 and so on.

    Note: l1 is set to [] as input layer has no Landauer cost
    """
    shapes = get_shapes(model)
    # iterate through all layer backward, so we don't erase entropy we will use later
    for i in range(len(model.layers) - 1, -1, -1):
        previousShape = shapes[i]
        shape = shapes[i + 1]
        layer = model.layers[i]

        if isinstance(layer, tf.keras.layers.Dense):
            # iterate through all index of the current layer
            for ind in index_multidim(shape):
                entropy = get_matrix(entropies[i + 1], ind)
                prev_entropy = 0

                # In dense layer all neurons in previous layer are initial
                # We sum their entropy
                for prev_index in index_multidim(previousShape):
                    prev_entropy += get_matrix(entropies[i], prev_index)

                # Set Landauer cost where current entropy was stored.
                set_matrix(entropies[i + 1], ind, max(0, prev_entropy - entropy)*_ENTROPY2EV)

        elif isinstance(layer, tf.keras.layers.Conv2D):
            kernel = layer.kernel_size
            stride = layer.strides

            for ind in index_multidim(shape):
                entropy = get_matrix(entropies[i + 1], ind)
                prev_entropy = 0

                # We iterate through all initial layer and sum their entropy
                for n_filter in range(previousShape[3]):  # All filter index
                    for m in range(kernel[0]):  # 2nd dim of the kernel
                        for n in range(kernel[1]):  # 1st dim of the kernel
                            # Get the index matching kernel position in previous layer
                            # and take stride into account
                            prev_index = (ind[0], ind[1] * stride[0] + m,
                                          ind[2] * stride[1] + n, n_filter)
                            prev_entropy += get_matrix(entropies[i], prev_index)

                set_matrix(entropies[i + 1], ind, max(0, prev_entropy - entropy)*_ENTROPY2EV)

        elif isinstance(layer, tf.keras.layers.MaxPooling2D) or\
                isinstance(layer, tf.keras.layers.AveragePooling2D):
            # Main difference with convolution: only one filter in previous layer
            kernel = layer.pool_size
            stride = layer.strides

            for ind in index_multidim(shape):
                entropy = get_matrix(entropies[i + 1], ind)
                prev_entropy = 0

                for m in range(kernel[0]):
                    for n in range(kernel[1]):
                        prev_index = (ind[0], ind[1] * stride[0] + m,
                                      ind[2] * stride[1] + n, ind[3])
                        prev_entropy += get_matrix(entropies[i], prev_index)

                set_matrix(entropies[i + 1], ind, max(0, prev_entropy - entropy)*_ENTROPY2EV)

        elif isinstance(layer, tf.keras.layers.Flatten) or\
                isinstance(layer, tf.keras.layers.Reshape):
            entropies[i + 1] = []  # No Landauer cost for reshaping

    entropies[0] = []  # Inputs alone have no Landauer costs


# This is an example of use for this library
if __name__ == "__main__":
    n = (4, 4, 1)
    model = keras.Sequential(
        [
            layers.Input((n)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(2)
        ]
    )

    n_samples = 10000
    bin_size = 0.01

    inputs = np.random.normal(0, 1, (n_samples, *(model.input.shape[1:])))
    opt_inputs = np.random.normal(0.5, 1.5, (n_samples, *(model.input.shape[1:])))

    #h = samples2MismatchPerNeuron(model, inputs, opt_inputs, bin_size=bin_size)
    h = samples2Landauer(model, inputs, bin_size=bin_size)
    #h = heuristicLandauer(model, 1)

    print(h)