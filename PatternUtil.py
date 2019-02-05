def to_nparray(letter):
    """
    Converts a string of chars representing a letter into a numpy array
    """
    from numpy import array
    return array([+1 if c=="X" else -1 for c in letter.replace('\n', '')])

def to_nparray_bin(letter):
    """
    Converts a string of chars representing a letter (binary) into a numpy array
    """
    from numpy import array
    return array([+1 if c=="X" else -1 for c in letter])

def display_pattern(pattern):
    """
    Displays the pattern from a numpy array
    """
    from pylab import imshow, cm, show
    imshow(pattern.reshape((5,5)), cmap=cm.binary, interpolation='nearest')
    show()