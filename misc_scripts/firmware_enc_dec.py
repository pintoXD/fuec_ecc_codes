# import fuec_encoder_decoder


#This decoder does not correct errors, it's just a simple word extractor from a 12-bit Hamming code.
def hamming_decoder_48_32(data_in):
    if len(data_in) != 12:
        raise ValueError("Input data must be a 12 characters string.")
    
    parsed_data_in = data_in[data_in.find('X"') + 2:data_in.find('"')]
    
    return parsed_data_in

    
    


def first():
    aux = 'X"00f300002537"'
    print(len(aux))
    print(hamming_decoder_48_32(aux))


if __name__ == "__main__":
    first()