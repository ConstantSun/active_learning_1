def get_acquisition_func(i: int):
    """
    Returns the func of the func.

    Args:
        i: (todo): write your description
    """
    switcher = {
        0: "category",
        1: "mean",
        2: "std",
        3: "random",
    }
    return switcher.get(i, "category")


print(get_acquisition_func(2))

from models.pan_regnety120 import PAN

model = PAN(is_dropout=False)

print("dropout" + str(model.is_dropout))
