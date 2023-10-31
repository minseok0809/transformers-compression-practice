from .bitpack_utils import pack_tensor, unpack_tensor
import torch
import numpy as np

int_type = [torch.uint8, torch.int8, torch.int16, torch.short, torch.int32, torch.int, torch.int64, torch.long]

def get_offset(tensor):
    """
    A function to return the offset if input has negative value.

    """
   
    # assert tensor.dtype in int_type
    min_value = torch.min(tensor)
    if min_value >= 0:
        return 0
    else:
        return -min_value


def get_bitwidth(tensor):
    """
    A function to return the smallest bitwidth that can represent the tensor.
    Specifically, if the max value of the tensor is smaller than 2 ** k - 1,
    then the tensor can be packed with k bit.

    """
    # assert tensor.dtype in int_type
    max_bitwidth = 8
    max_value = torch.max(tensor)

    for k in range(1, max_bitwidth + 1):
        if max_value < 2 ** k - 1:
            return k

    print("Error: The tensor is more than 8-bit")
    return max_bitwidth


def save_quantized_state_dict(state_dict, output_path):
    """
    A function to pack and save the quantized state_dict.
    The codes inside the for loop can be directly applied on a single quantized tensor with ultra low-precison.

    """

    quantized_layer = ['bert.embeddings.word_embeddings.weight', 
                       'attention.self.query.weight',
                       'attention.self.key.weight',
                       'attention.self.value.weight',
                       'attention.output.dense.weight',
                       'intermediate.dense.weight',
                       'output.dense.weight',
                       'bert.pooler.dense.weight']
    
    bit_dict = {}
    shape_dict = {}
    offset_dict = {}
    original_size = 0
    packed_size = 0

    for k, v in state_dict.items():
        # shape_dict[k] = v.shape
        original_size += v.nelement() * v.element_size()

        v_tensor = v.cpu()
        
        # offset_dict[k] = get_offset(v_tensor).to(torch.uint8)
        offset_dict[k] = get_offset(v_tensor)
        v_tensor = v_tensor + offset_dict[k]

        bitwidth = get_bitwidth(v_tensor)
        # bit_dict[k] = bitwidth

        if any(attr in str(k) for attr in quantized_layer):
            np_packed_tensor = pack_tensor(v_tensor.numpy(), bitwidth)
            state_dict[k] = torch.tensor(np_packed_tensor)
        packed_size += state_dict[k].nelement() * state_dict[k].element_size()

    print()
    print("Original Size: ", round(original_size / (1024 * 1024), 1), "MB")
    print("Packed Size: ", round(packed_size / (1024 * 1024), 1), "MB")
    print("Total Compression Ratio: ", original_size / packed_size)
    print()
    
    torch.save(state_dict, output_path, pickle_protocol=2)
    return state_dict

def load_quantized_state_dict(input_path, target_device):
    """
    A function to load and unpack the packed state_dict.
    The codes inside the for loop can be directly applied on a single packed tensor.

    """
    state_dict, bit_dict, shape_dict, offset_dict = torch.load(input_path, map_location=target_device)

    for k, v in state_dict.items():
        bitwidth = bit_dict[k]
        target_shape = shape_dict[k]
        offset = offset_dict[k]

        if bitwidth != -1:
            if target_device != torch.device('cpu'):
                np_tensor = unpack_tensor(v.cpu().numpy(), bitwidth, target_shape)
            else:
                np_tensor = unpack_tensor(v.numpy(), bitwidth, target_shape)
            state_dict[k] = torch.tensor(np_tensor).to(torch.int32) - offset
        else:
            state_dict[k] = v

    return state_dict
