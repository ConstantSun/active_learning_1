from abc import abstractmethod


class Parser:
    def __init__(self):
        """
        Initialize the object

        Args:
            self: (todo): write your description
        """
        pass

    @abstractmethod
    def _filename(self, index, basename=False, absolute=False):
        """
        Return the filename.

        Args:
            self: (todo): write your description
            index: (str): write your description
            basename: (str): write your description
            absolute: (str): write your description
        """
        pass

    def filename(self, index, basename=False, absolute=False):
        """
        Return the filename of the filename.

        Args:
            self: (todo): write your description
            index: (str): write your description
            basename: (str): write your description
            absolute: (str): write your description
        """
        return self._filename(index, basename=basename, absolute=absolute)

    def filenames(self, basename=False, absolute=False):
        """
        Return a list of filenames.

        Args:
            self: (todo): write your description
            basename: (str): write your description
            absolute: (todo): write your description
        """
        return [self._filename(index, basename=basename, absolute=absolute) for index in range(len(self))]

