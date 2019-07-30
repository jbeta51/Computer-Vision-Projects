import numpy as np


def verticalSeam(cumulative):
    #  find lowest cumulative value along bottom row
    out = list()
    # last row
    out.append(np.argmin(cumulative[-1]))
    for i in range(len(cumulative) - 1):
        # row we look at
        y = -(len(out)+1)
        if out[-1] == 0:
            choices = [cumulative[y, out[-1]], cumulative[y, out[-1] + 1]]
            out.append(out[-1] + np.argmin(choices))
        elif out[-1] == cumulative.shape[1] - 1:
            choices = [cumulative[y, out[-1]], cumulative[y, out[-1] - 1]]
            out.append(out[-1] - np.argmin(choices))
        else:
            cumulative[y, out[-1] - 1]
            cumulative[y, out[-1]]
            cumulative[y, out[-1] + 1]
            choices = [cumulative[y, out[-1] - 1], cumulative[y, out[-1]], cumulative[y, out[-1] + 1]]
            out.append(out[-1] + np.argmin(choices) - 1)

    out.reverse()
    return np.array(out)
