from numpy.core.numeric import normalize_axis_tuple


def moveaxis(x, source, destination):
    # simply copied from np.moveaxis
    dim = len(x.shape)
    source = normalize_axis_tuple(source, dim, 'source')
    destination = normalize_axis_tuple(destination, dim, 'destination')
    if len(source) != len(destination):
        raise ValueError('`source` and `destination` arguments must have the same number of elements')

    order = [n for n in range(dim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    return x.permute(order)
