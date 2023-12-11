class FlippedRobotException(RuntimeError):
    """
        Exception raised when the robot is flipped.
    """
    def __int__(self):
        super().__init__("Robot is flipped")


class WallHitException(RuntimeError):
    """
        Exception raised when the robot hits a wall.
    """
    def __int__(self):
        super().__init__("The robot has hit a wall")
