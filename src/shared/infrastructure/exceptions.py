class SingletonException(RuntimeError):
    """
        Exception raised when a singleton class is instantiated more than once.
    """
    def __int__(self):
        super().__init__("This class is a singleton. Instantiate it via get_instance() method")
