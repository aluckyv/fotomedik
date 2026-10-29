from fotomedik.jpeg.bitreader import BitReader, MarkerHit


def test_reads_msb_first():
    br = BitReader(bytes([0b1010_0000, 0xFF, 0x00]), 0)
    assert br.read_bit() == 1
    assert br.read_bit() == 0
    assert br.read_bit() == 1
    assert br.read_bits(5) == 0
    assert br.bit_pos == 8
    assert br.byte_pos == 1


def test_byte_stuffing_ff00_yields_ff():
    br = BitReader(bytes([0xFF, 0x00, 0x12]), 0)
    assert br.read_bits(8) == 0xFF
    assert br.read_bits(8) == 0x12
    # stuffed zero byte does not count as data bits but position is in the stream
    assert br.byte_pos == 3


def test_marker_stops_reading():
    br = BitReader(bytes([0xAB, 0xFF, 0xD0, 0x12]), 0)
    assert br.read_bits(8) == 0xAB
    try:
        br.read_bits(4)
    except MarkerHit as m:
        assert m.marker == 0xD0
        assert m.offset == 1
    else:
        raise AssertionError("expected MarkerHit")
    # after a hit the reader is positioned at the marker
    assert br.marker_at() == (0xD0, 1)


def test_fill_bytes_before_marker():
    # FF FF D1: fill byte then marker
    br = BitReader(bytes([0xFF, 0xFF, 0xD1]), 0)
    assert br.marker_at() == (0xD1, 0)


def test_eof_raises():
    br = BitReader(bytes([0x80]), 0)
    br.read_bits(8)
    try:
        br.read_bit()
    except EOFError:
        pass
    else:
        raise AssertionError("expected EOFError")


def test_start_offset_and_end():
    data = b"\x00" * 10 + bytes([0xF0]) + b"\x00" * 5
    br = BitReader(data, 10, end=11)
    assert br.read_bits(4) == 0xF
    assert br.read_bits(4) == 0x0
    try:
        br.read_bit()
    except EOFError:
        pass
    else:
        raise AssertionError("expected EOFError at end bound")


def test_align_and_skip_marker():
    br = BitReader(bytes([0b1100_0000, 0xFF, 0xD2, 0x55]), 0)
    br.read_bits(2)
    br.align_to_byte()
    assert br.byte_pos == 1
    assert br.marker_at() == (0xD2, 1)
    br.skip_marker()
    assert br.read_bits(8) == 0x55


def test_peek_does_not_advance():
    br = BitReader(bytes([0b1011_0000]), 0)
    assert br.peek_bits(3) == 0b101
    assert br.bit_pos == 0
    assert br.read_bits(3) == 0b101


def test_tell_is_exact_across_stuffing():
    data = bytes([0x12, 0xFF, 0x00, 0x34, 0x56])
    br = BitReader(data, 0)
    assert br.tell() == (0, 0)
    br.read_bits(3)
    assert br.tell() == (0, 3)
    br.read_bits(7)  # 5 bits from byte 0, 2 from the FF at offset 1
    assert br.tell() == (1, 2)
    br.read_bits(6)  # finish FF -> next real byte is 0x34 at offset 3 (00 is stuffing)
    assert br.tell() == (3, 0)
    br.peek_bits(16)  # peeking must not disturb tell()
    assert br.tell() == (3, 0)
    br.read_bits(9)
    assert br.tell() == (4, 1)


def test_receive_extend():
    # category 3, bits 010 -> -5 ; bits 110 -> 6
    br = BitReader(bytes([0b0101_1000]), 0)
    assert br.receive_extend(3) == -5
    assert br.receive_extend(3) == 6
    assert br.receive_extend(0) == 0
