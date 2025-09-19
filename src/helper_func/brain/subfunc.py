from src.utils.load_imports.loading import *
# from src.utils.load_imports.load_regmethods import *

#Brain Processing
def gen_maskeddata(brain_data:np.ndarray, mask:np.ndarray):
    _,_,s = brain_data.shape
    ones_array = np.ones(s)
    expanded_mask = mask[:, :, np.newaxis] * ones_array
    masked_data = expanded_mask * brain_data
    return masked_data

