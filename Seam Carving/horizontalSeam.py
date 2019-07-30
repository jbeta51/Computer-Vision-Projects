import numpy as np

def horizontalSeam(cumulative):
    #  find lowest cumulative value along right col
    out = list()
    # right col
    out.append(np.argmin(cumulative[:, -1]))
    for i in range(len(cumulative[0]) - 1):
        # col we look at
        x = -(len(out)+1)
        if out[-1] == 0:
            choices = [cumulative[out[-1], x], cumulative[out[-1] + 1, x]]
            out.append(out[-1] + np.argmin(choices))
        elif out[-1] == cumulative.shape[0] - 1:
            choices = [cumulative[out[-1], x], cumulative[out[-1] - 1, x]]
            out.append(out[-1] - np.argmin(choices))
        else:
            choices = [cumulative[out[-1] - 1, x], cumulative[out[-1], x], cumulative[out[-1] + 1, x]]
            out.append(out[-1] + np.argmin(choices) - 1)

    out.reverse()
    return np.array(out)
