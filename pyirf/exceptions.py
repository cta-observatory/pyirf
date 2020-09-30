class IRFException(Exception):
    pass


class MissingColumns(IRFException):
    def __init__(self, columns):
        super().__init__(f'Table is missing required columns {columns}')


class WrongColumnUnit(IRFException):
    def __init__(self, column, unit, expected):
        super().__init__(
            f'Unit {unit} of column "{column}"'
            f' has incompatible unit "{unit}", expected {expected}'
            f' required column {column}'
        )
