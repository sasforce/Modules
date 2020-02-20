import numpy as np


def modulate(bits):
    """generates a 16-QAM symbol of unit energy"""
    assert bits.size == 4
    # real part quadrant
    rpq = 4*bits[0]-2
    # imaginary part quadrant
    ipq = 4*bits[1]-2

    # last two bits determine the position in the quadrant
    # to get gary coding we need to rotate the sub qpsk symbos
    rp = 2*bits[2]-1
    rp = rp*np.sign(rpq)
    ip = 2*bits[3]-1
    ip = ip*np.sign(ipq)

    symbol = np.complex(rpq+rp, ipq+ip)

    # normalize
    symbol = symbol/np.sqrt(10)

    return symbol


def demodulate(symbol):
    """demodulates a 16-QAM symbol of unit energy"""
    symbol = symbol * np.sqrt(10)

    def compute_bit(level):
        if level >= 0:
            return 1
        else:
            return 0

    cb = compute_bit
    rp = symbol.real
    ip = symbol.imag
    # quadrant bits
    rbq = cb(rp)
    ibq = cb(ip)

    # position bits, remove information from the quadrant bits
    rbp = rp-(4*rbq-2)
    # undo the rotation
    rbp = cb(np.sign(rp)*rbp)
    ibp = ip-(4*ibq-2)
    ibp = cb(np.sign(ip)*ibp)

    bits = np.array([rbq, ibq, rbp, ibp])

    return bits


def bi2dec(code):
    """Translate the binary to integer"""
    return 8*code[0]+4*code[1]+2*code[2]+code[3]


def dec2bi(integer):
    """Translate the integer to binary"""
    return [int(x) for x in list(bin(integer)[2:])]


def vec_mod(bits):
    """vector wrapper for qam16.modulate
        Bits->Signal
    """
    M = bits.size/4
    bits = np.array_split(bits, M)
    symbols = [modulate(x) for x in bits]
    # reshape symbols
    symbols = np.hstack(symbols)
    return symbols


def vec_demod(signals, out_type):
    """vector wrapper for qam16.demodulate
        Signal->Bits
    """
    M = signals.size
    signals = np.array_split(signals, M)
    bits = [demodulate(x) for x in signals]
    symbols = [bi2dec(x) for x in bits]
    bits = np.hstack(bits)
    symbols = np.hstack(symbols)
    if out_type == 'binary':
        return bits
    elif out_type == 'integer':
        return symbols
    else:
        return bits, symbols


def symbols2bits(integer_arr):
    """Translate a integer array into a binary array"""
    bit_arr = [dec2bi(x) for x in integer_arr]
    return np.hstack(bit_arr)


def bits2symbols(bit_arr):
    """Translate a binary array into a integer array"""
    M = bit_arr.size/4
    bit_arr = np.array_split(bit_arr, M)
    integer_arr = [bi2dec(x) for x in bit_arr]
    return np.hstack(integer_arr)
