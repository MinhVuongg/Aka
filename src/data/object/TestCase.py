import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TestCase:
    """Cấu trúc cho một test case."""
    _id: int
    _td: str
    _simplified_t: str
    _isAutomated: bool
    _dt: Dict[str, Any] = field(default_factory=dict)

    # Getter và setter cho id
    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = value

    # Getter và setter cho td (test data)
    @property
    def td(self) -> str:
        return self._td

    @td.setter
    def td(self, value: str):
        self._td = value

    # Getter và setter cho simplified_t
    @property
    def simplified_t(self) -> str:
        return self._simplified_t

    @simplified_t.setter
    def simplified_t(self, value: str):
        self._simplified_t = value

    # Getter và setter cho isAutomated
    @property
    def isAutomated(self) -> bool:
        return self._isAutomated

    @isAutomated.setter
    def isAutomated(self, value: bool):
        self._isAutomated = value

    # Getter và setter cho dt (test details)
    @property
    def dt(self) -> Dict[str, Any]:
        return self._dt

    @dt.setter
    def dt(self, value: Dict[str, Any]):
        self._dt = value

    # Các phương thức tiện ích để thao tác với dt
    def add_detail(self, key: str, value: Any):
        """Thêm một mục mới vào chi tiết kiểm thử"""
        self._dt[key] = value

    def get_detail(self, key: str, default: Any = None) -> Any:
        """Lấy giá trị của một mục chi tiết"""
        return self._dt.get(key, default)

    def has_detail(self, key: str) -> bool:
        """Kiểm tra xem một mục chi tiết có tồn tại không"""
        return key in self._dt