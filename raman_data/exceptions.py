"""
Custom exceptions for the raman_data package.

This module defines custom exception classes used throughout the package
to handle specific error conditions related to data integrity and file handling.
"""

from typing import Optional


class ChecksumError(Exception):
    """
    Raised when a file's checksum does not match the expected value.

    This exception is typically raised during file download operations
    when the integrity verification fails.

    Attributes:
        expected_checksum (str | None): The expected hash value of the file.
        actual_checksum (str | None): The computed hash value of the downloaded file.

    Example:
        >>> raise ChecksumError(expected_checksum="abc123", actual_checksum="def456")
        ChecksumError: Expected: abc123 but got def456
    """

    def __init__(
        self, 
        expected_checksum: Optional[str] = None,
        actual_checksum: Optional[str] = None
    ) -> None:
        """
        Initialize the ChecksumError.

        Args:
            expected_checksum: The expected hash value of the file.
            actual_checksum: The computed hash value of the downloaded file.
        """
        super().__init__()
        self.expected_checksum = expected_checksum
        self.actual_checksum = actual_checksum
        
    def __str__(self) -> str:
        return f"Expected: {self.expected_checksum} but got {self.actual_checksum}"
    

class CorruptedZipFileError(Exception):
    """
    Raised when a ZIP file is corrupted or cannot be processed.

    This exception is typically raised when attempting to extract
    a ZIP file that is malformed or incomplete.

    Attributes:
        zip_file_path (str | None): The full path to the corrupted ZIP file.
        zip_file_name (str | None): The name of the corrupted ZIP file.

    Example:
        >>> raise CorruptedZipFileError(zip_file_path="/path/to/file.zip", zip_file_name="file.zip")
        CorruptedZipFileError: The file file.zip at /path/to/file.zip seems to be corrupted...
    """

    def __init__(
        self, 
        zip_file_path: Optional[str] = None,
        zip_file_name: Optional[str] = None
    ) -> None:
        """
        Initialize the CorruptedZipFileError.

        Args:
            zip_file_path: The full path to the corrupted ZIP file.
            zip_file_name: The name of the corrupted ZIP file.
        """
        super().__init__()
        self.zip_file_path = zip_file_path
        self.zip_file_name = zip_file_name
        
    def __str__(self) -> str:
         return f"The file {self.zip_file_name} at {self.zip_file_path} seems to be corrupted or otherwise unusable!"
