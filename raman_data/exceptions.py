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
    

class CorruptedZipFileError(Exception):
    def __init__(
        self, 
        zip_file_path: Optional[str] = None,
        zip_file_name: Optional[str] = None
    ) -> None:
        
        super().__init__()
        self.zip_file_path = zip_file_path
        self.zip_file_name = zip_file_name
        
    def __str__(self) -> str:
         return f"The file {self.zip_file_name} at {self.zip_file_path} seams to be corrupted or in a other way not usable!"
    