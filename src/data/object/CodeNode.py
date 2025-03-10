import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.data.object.TestCase import TestCase


@dataclass
class CodeNode:
    """Node chính chứa thông tin về code và các test case."""
    _fm: str  # focal method
    _fc: str = ""  # focal class
    _path_fm: str = ""
    _datatest: List[TestCase] = field(default_factory=list)
    _m: List[str] = field(default_factory=list)  # methods
    _c: List[str] = field(default_factory=list)  # constructors
    _f: List[str] = field(default_factory=list)  # fields

    # Getters và setters cho fm
    @property
    def fm(self) -> str:
        return self._fm

    @fm.setter
    def fm(self, value: str):
        self._fm = value

    # Getters và setters cho fc
    @property
    def fc(self) -> str:
        return self._fc

    @fc.setter
    def fc(self, value: str):
        self._fc = value

    # Getters và setters cho path_fm
    @property
    def path_fm(self) -> str:
        return self._path_fm

    @path_fm.setter
    def path_fm(self, value: str):
        self._path_fm = value

    # Getters và setters cho datatest
    @property
    def datatest(self) -> List[TestCase]:
        return self._datatest

    @datatest.setter
    def datatest(self, value: List[TestCase]):
        self._datatest = value

    # Getters và setters cho m
    @property
    def m(self) -> List[str]:
        return self._m

    @m.setter
    def m(self, value: List[str]):
        self._m = value

    # Getters và setters cho c
    @property
    def c(self) -> List[str]:
        return self._c

    @c.setter
    def c(self, value: List[str]):
        self._c = value

    # Getters và setters cho f
    @property
    def f(self) -> List[str]:
        return self._f

    @f.setter
    def f(self, value: List[str]):
        self._f = value