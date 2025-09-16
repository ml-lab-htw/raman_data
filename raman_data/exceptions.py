from typing import Optional

class ChecksumError(Exception):
    def __init__(
        self, 
        expected_checksum: Optional[str] = None,
        actual_checksum: Optional[str] = None
    ) -> None:
        
        super().__init__()
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum
        
    def __str__(self) -> str:
        return f"Expected: {self.expected_checksum} but got {self.actual_checksum}"
    