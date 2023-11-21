from typing import Optional

from shared.data_types import CameraReadingData, LidarReadingData, SonarsReadingsData


class State:

    def __init__(self, camera_reading: CameraReadingData, lidar_reading: Optional[LidarReadingData] = None,
                 sonars_readings: Optional[SonarsReadingsData] = None):

        # Class attributes
        self.__camera_reading: CameraReadingData = camera_reading

        # Optional arguments initialization
        if lidar_reading is not None:
            self.__lidar_reading: LidarReadingData = lidar_reading
        if sonars_readings is not None:
            self.__sonars_readings: SonarsReadingsData = sonars_readings

    # Properties
    def camera_reading(self) -> CameraReadingData:
        """
            Getter for the camera_reading private object.
        """
        return self.__camera_reading

    def _camera_reading(self, camera_reading: CameraReadingData) -> None:
        """
            Setter for the camera_reading private object.
            :param camera_reading: new camera_reading object to store.
        """
        self.__camera_reading = camera_reading

    camera_reading = property(fget=camera_reading, fset=_camera_reading)

    def lidar_reading(self) -> LidarReadingData:
        """
            Getter for the lidar_reading private object.
        """
        # Attribute initialization checking
        if self.__lidar_reading is None:
            raise AttributeError("Attribute self.__lidar_reading not initialized")

        return self.__lidar_reading

    def _lidar_reading(self, lidar: LidarReadingData) -> None:
        """
            Setter for the lidar_reading private object.
            :param lidar: new lidar_reading object to store.
        """
        self.__lidar_reading = lidar

    lidar_reading = property(fget=lidar_reading, fset=_lidar_reading)

    def sonars_readings(self) -> SonarsReadingsData:
        """
            Getter for the sonars_readings private object.
        """
        # Attribute initialization checking
        if self.__sonars_readings is None:
            raise AttributeError("Attribute self.__sonars_readings not initialized")

        return self.__sonars_readings

    def _sonars_readings(self, sonars: SonarsReadingsData) -> None:
        """
            Setter for the sonars_readings private object.
            :param sonars: new sonars_readings object to store.
        """
        self.__sonars_readings = sonars

    sonars_readings = property(fget=sonars_readings, fset=_sonars_readings)
