import matplotlib
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import extract
import numpy as np


# refer to
# http://matplotlib.org/examples/pylab_examples/histogram_percent_demo.html
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def plot_hist(y, title=None, percent=False):
    plt.hist(y, bins=5, normed=percent)
    if(percent):
        formatter = FuncFormatter(to_percent)
        plt.gca().yaxis.set_major_formatter(formatter)
    if title:
        plt.title(title)

def main():
    (X,y) = extract.generate_labelled_data()
    names = extract.get_feature_names()
    print(names)
    y = (map(lambda x:int(x),y))
    plot_hist(y, "Distribution of consort success")
    plt.show()
    fig, axes = plt.subplots(nrows=5, ncols=3)
    plt.tight_layout()
    for i in range(np.shape(X)[1]):
        plt.subplot(5,3,i+1)
        plot_hist(X[:,i], "Distribution of "+names[i])
    plt.show()

if __name__=="__main__":
    main()
