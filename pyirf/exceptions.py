class IRFException(Exception):
    pass


class MissingColumns(IRFException):
    """
    Exception to be raised when a table is missing a required column.
    """

    def __init__(self, columns):
        super().__init__(f"Table is missing required columns {columns}")


class WrongColumnUnit(IRFException):
    """
    Exception to be raised when a column of a table has the wrong unit.
    """

    def __init__(self, column, unit, expected):
        super().__init__(
            f'Unit {unit} of column "{column}"'
            f' has incompatible unit "{unit}", expected {expected}'
            f" required column {column}"
        )
