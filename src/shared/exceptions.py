class FlippedRobotException(RuntimeError):
    """
        Exception raised when the robot is flipped.
    """
    def __int__(self):
        super().__init__("Robot is flipped")
