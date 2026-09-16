"""
Unit tests for source/communication/message.py.

Tests message types and data structures used for board communication.
"""

import pytest
from source.communication.message import MsgType, Datatuple


class TestMsgType:
    """Test suite for MsgType enum"""

    # =========================================================================
    # Enum Value Tests
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.communication
    def test_msgtype_values(self):
        """Test that MsgType enum has correct byte values"""
        assert MsgType.EVENT.value == b"E"
        assert MsgType.STATE.value == b"S"
        assert MsgType.PRINT.value == b"P"
        assert MsgType.HARDW.value == b"H"
        assert MsgType.VARBL.value == b"V"
        assert MsgType.WARNG.value == b"!"
        assert MsgType.ERROR.value == b"!!"
        assert MsgType.STOPF.value == b"X"
        assert MsgType.ANLOG.value == b"A"
        assert MsgType.THRSH.value == b"T"

    @pytest.mark.unit
    @pytest.mark.communication
    def test_msgtype_enum_members(self):
        """Test that all expected MsgType members exist"""
        expected_members = [
            'EVENT', 'STATE', 'PRINT', 'HARDW', 'VARBL',
            'WARNG', 'ERROR', 'STOPF', 'ANLOG', 'THRSH'
        ]

        for member_name in expected_members:
            assert hasattr(MsgType, member_name)
            assert isinstance(getattr(MsgType, member_name), MsgType)

    @pytest.mark.unit
    @pytest.mark.communication
    def test_msgtype_unique_values(self):
        """Test that all MsgType values are unique"""
        values = [member.value for member in MsgType]
        assert len(values) == len(set(values))

    # =========================================================================
    # from_byte Tests
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.communication
    def test_from_byte_valid_event(self):
        """Test from_byte with valid EVENT byte"""
        result = MsgType.from_byte(b"E")

        assert result == MsgType.EVENT
        assert isinstance(result, MsgType)

    @pytest.mark.unit
    @pytest.mark.communication
    def test_from_byte_valid_state(self):
        """Test from_byte with valid STATE byte"""
        result = MsgType.from_byte(b"S")

        assert result == MsgType.STATE

    @pytest.mark.unit
    @pytest.mark.communication
    def test_from_byte_valid_print(self):
        """Test from_byte with valid PRINT byte"""
        result = MsgType.from_byte(b"P")

        assert result == MsgType.PRINT

    @pytest.mark.unit
    @pytest.mark.communication
    def test_from_byte_all_types(self):
        """Test from_byte with all valid message types"""
        test_cases = [
            (b"E", MsgType.EVENT),
            (b"S", MsgType.STATE),
            (b"P", MsgType.PRINT),
            (b"H", MsgType.HARDW),
            (b"V", MsgType.VARBL),
            (b"!", MsgType.WARNG),
            (b"!!", MsgType.ERROR),
            (b"X", MsgType.STOPF),
            (b"A", MsgType.ANLOG),
            (b"T", MsgType.THRSH),
        ]

        for byte_value, expected_type in test_cases:
            result = MsgType.from_byte(byte_value)
            assert result == expected_type

    @pytest.mark.unit
    @pytest.mark.communication
    def test_from_byte_invalid_returns_input(self):
        """Test from_byte with invalid byte returns input"""
        invalid_byte = b"Z"
        result = MsgType.from_byte(invalid_byte)

        assert result == invalid_byte
        assert not isinstance(result, MsgType)

    @pytest.mark.unit
    @pytest.mark.communication
    def test_from_byte_empty_byte(self):
        """Test from_byte with empty byte"""
        result = MsgType.from_byte(b"")

        assert result == b""

    # =========================================================================
    # get_subtype Tests
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.communication
    def test_get_subtype_varbl_get(self):
        """Test get_subtype for VARBL type with 'g' subtype"""
        result = MsgType.VARBL.get_subtype('g')

        assert result == 'get'

    @pytest.mark.unit
    @pytest.mark.communication
    def test_get_subtype_varbl_all_subtypes(self):
        """Test all VARBL subtypes"""
        test_cases = [
            ('g', 'get'),
            ('s', 'user_set'),
            ('a', 'api_set'),
            ('p', 'print'),
            ('t', 'run_start'),
            ('e', 'run_end'),
        ]

        for char, expected in test_cases:
            result = MsgType.VARBL.get_subtype(char)
            assert result == expected

    @pytest.mark.unit
    @pytest.mark.communication
    def test_get_subtype_event_all_subtypes(self):
        """Test all EVENT subtypes"""
        test_cases = [
            ('i', 'input'),
            ('t', 'timer'),
            ('p', 'publish'),
            ('u', 'user'),
            ('a', 'api'),
            ('s', 'sync'),
        ]

        for char, expected in test_cases:
            result = MsgType.EVENT.get_subtype(char)
            assert result == expected

    @pytest.mark.unit
    @pytest.mark.communication
    def test_get_subtype_print_all_subtypes(self):
        """Test all PRINT subtypes"""
        test_cases = [
            ('t', 'task'),
            ('a', 'api'),
            ('u', 'user'),
            ('s', 'trigger'),
        ]

        for char, expected in test_cases:
            result = MsgType.PRINT.get_subtype(char)
            assert result == expected

    @pytest.mark.unit
    @pytest.mark.communication
    def test_get_subtype_thrsh_all_subtypes(self):
        """Test all THRSH subtypes"""
        test_cases = [
            ('s', 'run_start'),
            ('t', 'task'),
        ]

        for char, expected in test_cases:
            result = MsgType.THRSH.get_subtype(char)
            assert result == expected

    @pytest.mark.unit
    @pytest.mark.communication
    def test_get_subtype_underscore_returns_none(self):
        """Test get_subtype with underscore character"""
        result = MsgType.VARBL.get_subtype('_')

        assert result is None

    @pytest.mark.unit
    @pytest.mark.communication
    def test_get_subtype_invalid_character(self):
        """Test get_subtype with invalid character raises KeyError"""
        with pytest.raises(KeyError):
            MsgType.VARBL.get_subtype('z')

    @pytest.mark.unit
    @pytest.mark.communication
    def test_get_subtype_event_input(self):
        """Test EVENT subtype 'input'"""
        result = MsgType.EVENT.get_subtype('i')

        assert result == 'input'

    @pytest.mark.unit
    @pytest.mark.communication
    def test_get_subtype_event_timer(self):
        """Test EVENT subtype 'timer'"""
        result = MsgType.EVENT.get_subtype('t')

        assert result == 'timer'

    # =========================================================================
    # Subtype Coverage Tests
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.communication
    def test_msgtype_with_subtypes_coverage(self):
        """Test that types with subtypes are properly defined"""
        # Types that should have subtypes
        types_with_subtypes = [MsgType.VARBL, MsgType.EVENT, MsgType.PRINT, MsgType.THRSH]

        for msg_type in types_with_subtypes:
            # Should not raise KeyError for underscore
            result = msg_type.get_subtype('_')
            assert result is None


class TestDatatuple:
    """Test suite for Datatuple namedtuple"""

    # =========================================================================
    # Creation Tests
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_creation_all_fields(self):
        """Test creating Datatuple with all fields"""
        dt = Datatuple(time=100, type="EVENT", subtype="input", content="data")

        assert dt.time == 100
        assert dt.type == "EVENT"
        assert dt.subtype == "input"
        assert dt.content == "data"

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_creation_defaults(self):
        """Test creating Datatuple with default values"""
        dt = Datatuple()

        assert dt.time is None
        assert dt.type is None
        assert dt.subtype is None
        assert dt.content is None

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_creation_partial(self):
        """Test creating Datatuple with partial fields"""
        dt = Datatuple(time=50, type="STATE")

        assert dt.time == 50
        assert dt.type == "STATE"
        assert dt.subtype is None
        assert dt.content is None

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_named_arguments(self):
        """Test creating Datatuple with named arguments"""
        dt = Datatuple(content="test", time=200, subtype="user", type="PRINT")

        assert dt.time == 200
        assert dt.type == "PRINT"
        assert dt.subtype == "user"
        assert dt.content == "test"

    # =========================================================================
    # Field Access Tests
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_field_access(self):
        """Test accessing Datatuple fields"""
        dt = Datatuple(time=100, type="EVENT", subtype="input", content="data")

        # Access by attribute
        assert dt.time == 100
        assert dt.type == "EVENT"

        # Access by index
        assert dt[0] == 100
        assert dt[1] == "EVENT"
        assert dt[2] == "input"
        assert dt[3] == "data"

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_immutability(self):
        """Test that Datatuple is immutable"""
        dt = Datatuple(time=100, type="EVENT")

        with pytest.raises(AttributeError):
            dt.time = 200

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_unpacking(self):
        """Test unpacking Datatuple"""
        dt = Datatuple(time=100, type="EVENT", subtype="input", content="data")

        time, type_, subtype, content = dt

        assert time == 100
        assert type_ == "EVENT"
        assert subtype == "input"
        assert content == "data"

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_length(self):
        """Test Datatuple has correct number of fields"""
        dt = Datatuple()

        assert len(dt) == 4

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_fields_attribute(self):
        """Test Datatuple _fields attribute"""
        expected_fields = ('time', 'type', 'subtype', 'content')

        assert Datatuple._fields == expected_fields

    # =========================================================================
    # Equality and Comparison Tests
    # =========================================================================

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_equality(self):
        """Test Datatuple equality comparison"""
        dt1 = Datatuple(time=100, type="EVENT", subtype="input", content="data")
        dt2 = Datatuple(time=100, type="EVENT", subtype="input", content="data")

        assert dt1 == dt2

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_inequality(self):
        """Test Datatuple inequality"""
        dt1 = Datatuple(time=100, type="EVENT")
        dt2 = Datatuple(time=200, type="EVENT")

        assert dt1 != dt2

    @pytest.mark.unit
    @pytest.mark.communication
    def test_datatuple_with_different_types(self):
        """Test Datatuple with different data types"""
        dt = Datatuple(
            time=123.456,
            type=MsgType.EVENT,
            subtype="input",
            content={"key": "value"}
        )

        assert isinstance(dt.time, float)
        assert isinstance(dt.type, MsgType)
        assert isinstance(dt.content, dict)

    # =========================================================================
    # Integration Tests
    # =========================================================================

    @pytest.mark.integration
    @pytest.mark.communication
    def test_msgtype_and_datatuple_integration(self):
        """Test using MsgType with Datatuple"""
        msg_type = MsgType.from_byte(b"E")
        subtype = msg_type.get_subtype('i')

        dt = Datatuple(
            time=100,
            type=msg_type,
            subtype=subtype,
            content="input event data"
        )

        assert dt.type == MsgType.EVENT
        assert dt.subtype == "input"
        assert isinstance(dt.type, MsgType)

    @pytest.mark.integration
    @pytest.mark.communication
    def test_message_parsing_workflow(self):
        """Test complete message parsing workflow"""
        # Simulate receiving a message byte
        byte_value = b"V"
        subtype_char = 'g'

        # Parse message type
        msg_type = MsgType.from_byte(byte_value)
        assert msg_type == MsgType.VARBL

        # Get subtype
        subtype = msg_type.get_subtype(subtype_char)
        assert subtype == 'get'

        # Create Datatuple
        dt = Datatuple(
            time=1234567,
            type=msg_type,
            subtype=subtype,
            content="variable_name"
        )

        # Verify complete message
        assert dt.type == MsgType.VARBL
        assert dt.subtype == 'get'
        assert dt.content == "variable_name"

    @pytest.mark.integration
    @pytest.mark.communication
    def test_multiple_message_types_workflow(self):
        """Test handling multiple message types"""
        messages = [
            (b"E", 'i', "button_press"),
            (b"S", '_', "state_change"),
            (b"P", 'u', "debug message"),
            (b"V", 's', "var_value"),
        ]

        datatuples = []
        for byte_val, subtype_char, content in messages:
            msg_type = MsgType.from_byte(byte_val)
            subtype = msg_type.get_subtype(subtype_char) if subtype_char != '_' else None

            dt = Datatuple(
                time=len(datatuples),
                type=msg_type,
                subtype=subtype,
                content=content
            )
            datatuples.append(dt)

        assert len(datatuples) == 4
        assert datatuples[0].type == MsgType.EVENT
        assert datatuples[1].type == MsgType.STATE
        assert datatuples[2].type == MsgType.PRINT
        assert datatuples[3].type == MsgType.VARBL
