import numpy as np
import torch
import torch.nn.functional as F

# def mask_np(array, null_val):
#     if np.isnan(null_val):
#         return (~np.isnan(null_val)).astype('float32')
#     else:
#         return np.not_equal(array, null_val).astype('float32')
#
#
# def masked_mape_np(y_true, y_pred, null_val=np.nan):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mask = mask_np(y_true, null_val)
#         mask /= mask.mean()
#         mape = np.abs((y_pred - y_true) / y_true)
#         mape = np.nan_to_num(mask * mape)
#         return np.mean(mape) * 100
#
#
# def masked_rmse_np(y_true, y_pred, null_val=np.nan):
#     mask = mask_np(y_true, null_val)
#     mask /= mask.mean()
#     mse = (y_true - y_pred) ** 2
#     return np.sqrt(np.mean(np.nan_to_num(mask * mse)))
#
#
# def masked_mae_np(y_true, y_pred, null_val=np.nan):
#     mask = mask_np(y_true, null_val)
#     mask /= mask.mean()
#     mae = np.abs(y_true - y_pred)
#     return np.mean(np.nan_to_num(mask * mae))



def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    else:
        return (array != null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean(axis=1, keepdims=True)  # Normalize mask for each row
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return mape * 100  # Return MAPE for each element


def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean(axis=1, keepdims=True)  # Normalize mask for each row
    mse = (y_true - y_pred) ** 2
    # print(mse.shape)
    # mse=np.mean(mse, axis=0, keepdims=True)
    # print(mse.shape)
    mse=np.nan_to_num(mask * mse)
    mse=np.mean(mse, axis=0, keepdims=True)
    result=np.sqrt(mse)
    return result  # Return RMSE for each element


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean(axis=1, keepdims=True)  # Normalize mask for each row
    mae = np.abs(y_true - y_pred)
    return np.nan_to_num(mask * mae)  # Return MAE for each element




# def mask_np(array, null_val):
#     if np.isnan(null_val):
#         return (~np.isnan(null_val)).astype('float32')
#     else:
#         return np.not_equal(array, null_val).astype('float32')
#
#
# def masked_mape_np(y_true, y_pred, null_val=np.nan):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         mask = mask_np(y_true, null_val)
#         mask /= mask.mean()
#         mape = np.abs((y_pred - y_true) / y_true)
#         mape = np.nan_to_num(mask * mape)
#         return mape * 100
#
#
# def masked_rmse_np(y_true, y_pred, null_val=np.nan):
#     mask = mask_np(y_true, null_val)
#     mask /= mask.mean()
#     mse = (y_true - y_pred) ** 2
#     return np.sqrt(np.nan_to_num(mask * mse))
#
#
# def masked_mae_np(y_true, y_pred, null_val=np.nan):
#     mask = mask_np(y_true, null_val)
#     mask /= mask.mean()
#     mae = np.abs(y_true - y_pred)
#     return np.nan_to_num(mask * mae)


