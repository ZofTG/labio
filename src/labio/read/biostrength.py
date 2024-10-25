"""products module"""

#! IMPORTS


import copy
from os.path import exists
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


#! CLASSES


class Product:
    """Product class object"""

    # * class variables

    _spring_correction: float = 1
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = 0
    _camme_ratio: float = 1
    _lever_number: int = 1
    _lever_radius_m: float = 0.054
    _rom_correction_coefs: list[float] = [0, 0, 0]
    _rm1_coefs: list[float] = [1, 0]

    _position_motor_rad: NDArray[np.floating[Any]]
    _load_lever_kgf: NDArray[np.floating[Any]]
    _time_s: NDArray[np.floating[Any]]

    # * attributes

    @property
    def time_s(self):
        """return the time of each sample"""
        return self._time_s[1:-1].astype(float)

    @property
    def position_motor_rad(self):
        """return the raw postition in radians"""
        return self._position_motor_rad[1:-1].astype(float)

    @property
    def pulley_radius_m(self):
        """pulley radius coefficient in m for each time sample"""
        return np.tile(self._spring_correction, len(self.time_s))

    @property
    def lever_weight_kgf(self):
        """lever weight coefficient in kgf for each time sample"""
        return np.tile(self._lever_weight_kgf, len(self.time_s))

    @property
    def camme_ratio(self):
        """camme ratio coefficient for each time sample"""
        return np.tile(self._camme_ratio, len(self.time_s))

    @property
    def spring_correction(self):
        """spring correction coefficient for each time sample"""
        return np.tile(self._spring_correction, len(self.time_s))

    @property
    def load_motor_nm(self):
        """return the motor load in Nm"""
        return (
            (self.load_lever_kgf - self.lever_weight_kgf)
            * G
            * self.pulley_radius_m
            / self.spring_correction
            * self.camme_ratio
        )

    @property
    def lever_number(self):
        """number of levers"""
        return np.tile(self._lever_number, len(self.time_s))

    @property
    def rom_correction_coefs(self):
        """rom correction coefficients with higher order first"""
        return self._rom_correction_coefs

    @property
    def position_lever_deg(self):
        """return the calculated position of the lever in degrees"""
        rad = self.position_motor_rad
        rad += np.polyval(self.rom_correction_coefs, self.load_motor_nm)
        return (rad * 180 / np.pi / self.lever_number).astype(float)

    @property
    def lever_radius_m(self):
        """radius of the lever(s) in m for each sample"""
        return np.tile(self._lever_radius_m, len(self.time_s)).astype(float)

    @property
    def position_lever_m(self):
        """return the calculated position of the lever in meters"""
        return (self.position_lever_deg / 180 * np.pi * self.lever_radius_m).astype(
            float
        )

    @property
    def load_lever_kgf(self):
        """return the calculated lever weight"""
        return self._load_lever_kgf[1:-1].astype(float)

    @property
    def speed_motor_rads(self):
        """
        return the calculated speed at the motor level in rad for each sample
        """
        num = self._position_motor_rad[:-2] - self._position_motor_rad[2:]
        den = self._time_s[:-2] - self._time_s[2:]
        return (num / den).astype(float)

    @property
    def speed_lever_degs(self):
        """
        return the calculated speed at the lever level in deg/s for each sample
        """
        rad = self._position_motor_rad
        rad += np.polyval(self.rom_correction_coefs, self.load_motor_nm)
        deg = rad * 180 / np.pi / self._lever_number
        deg = deg * self._lever_radius_m / self._pulley_radius_m
        num = deg[:-2] - deg[2:]
        den = self._time_s[:-2] - self._time_s[2:]
        return (num / den).astype(float)

    @property
    def speed_lever_ms(self):
        """
        return the calculated speed at the lever level in m/s for each sample
        """
        speed = self.speed_lever_degs / 180 * np.pi / self.lever_radius_m
        return speed.astype(float)

    @property
    def power_w(self):
        """return the calculated power"""
        return self.load_motor_nm * self.speed_motor_rads

    @property
    def rm1_coefs(self):
        """1RM coefficients with higher order first"""
        return self._rm1_coefs

    @property
    def name(self):
        """the name of the product"""
        return type(self).__name__

    # * methods

    def copy(self):
        """make a copy of the object"""
        return copy.deepcopy(self)

    def as_dataframe(self):
        """return a summary table containing the resulting data"""
        out = {
            ("Time", "s"): self.time_s,
            ("Lever Load", "kgf"): self.load_lever_kgf,
            ("Motor Load", "Nm"): self.load_motor_nm,
            ("Lever Position", "m"): self.position_lever_m,
            ("Lever Position", "deg"): self.position_lever_deg,
            ("Motor Position", "rad"): self.position_motor_rad,
            ("Lever Speed", "m/s"): self.speed_lever_ms,
            ("Lever Speed", "deg/s"): self.speed_lever_degs,
            ("Motor Speed", "rad/s"): self.speed_motor_rads,
            ("Power", "W"): self.power_w,
        }
        return pd.DataFrame(out)

    # * constructors

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        load_kgf: NDArray[np.floating],
    ):
        # check the entries
        try:
            self._time_s = np.array([time_s]).astype(float).flatten()
        except Exception as exc:
            raise ValueError(
                "time must be castable to a numpy array of floats"
            ) from exc
        try:
            self._position_motor_rad = (
                np.array([motor_position_rad]).astype(float).flatten()
            )
        except Exception as exc:
            raise ValueError(
                "motor_position_rad must be castable to a numpy array of floats"
            ) from exc
        try:
            self._load_lever_kgf = np.array([load_kgf]).astype(float).flatten()
            self._load_lever_kgf += self._lever_weight_kgf
        except Exception as exc:
            raise ValueError(
                "motor_load_nm must be castable to a numpy array of floats"
            ) from exc

        # check the length of each element
        if (
            not len(self.time_s)
            == len(self.position_motor_rad)
            == len(self.load_motor_nm)
        ):
            msg = "time_s, motor_position_rad and motor_load_nm must all have "
            msg += "the same number of samples."
            raise ValueError(msg)

    @classmethod
    def from_file(cls, file: str):
        """
        read raw data from file

        Parameters
        ----------
        file : str
            the path to the file
        """

        # check the inputs
        msg = "incorrect file."
        assert isinstance(file, str), msg
        assert exists(file), msg
        assert file.endswith(".txt") or file.endswith(".csv"), msg

        # get the data
        obj = pd.read_csv(file, sep="|")
        col = obj.columns[[0, 2, 5]]
        obj = obj[col].astype(str).map(lambda x: x.replace(",", "."))
        time, load, pos = [i.astype(float) for i in obj.values.T]

        # return
        return cls(time, pos, load)  # type: ignore


class ChestPress(Product):
    """Chest press class object"""

    _spring_correction: float = 1.15
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = 4.0
    _camme_ratio: float = 0.74
    _lever_number: int = 1  # ? TO BE CHECKED might be 1
    _lever_radius_m: float = 0.87489882
    _rom_correction_coefs: list[float] = [
        -0.0000970270993668,
        0.0284363503605837,
        -0.1454105176656738,
    ]
    _rm1_coefs: list[float] = [0.96217, 2.97201]

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class LegPress(Product):
    """Leg Press class object"""

    _spring_correction: float = 1
    _pulley_radius_m: float = 0.08175
    _lever_weight_kgf: float = 9.0 + 0.17 * 85
    _camme_ratio: float = 1
    _lever_number: int = 1
    _lever_radius_m: float = 1
    _rom_correction_coefs: list[float] = [
        -0.0000594298355666,
        0.0155680740573513,
        -0.0022758912872085,
    ]
    _rm1_coefs: list[float] = [0.65705, 9.17845]

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class LegPressREV(LegPress):
    """Leg Press REV class object"""

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class AdjustablePulleyREV(Product):
    """Adjustable Pulley REV class object"""

    _spring_correction: float = 1
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = 0.01
    _camme_ratio: float = 0.25
    _lever_number: int = 2
    _lever_radius_m: float = 0.054
    _rom_correction_coefs: list[float] = [0, 0, 0]
    _rm1_coefs: list[float] = [1, 0]

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class LegExtension(Product):
    """Leg Extension class object"""

    _spring_correction: float = 0.79
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = 9.0 + 0.17 * 85
    _camme_ratio: float = 0.738
    _lever_number: int = 1
    _lever_radius_m: float = 1  # ? TO BE CHECKED
    _rom_correction_coefs: list[float] = [
        0.1237962826137063,
        -0.0053627811034270,
        0.0003232899485875,
    ]
    _rm1_coefs: list[float] = [0.7351, 6]

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


class LegExtensionREV(LegExtension):
    """Leg Extension REV class object"""

    _spring_correction: float = 1
    _pulley_radius_m: float = 0.054
    _lever_weight_kgf: float = 0.875
    _camme_ratio: float = 0.689
    _lever_number: int = 1
    _lever_radius_m: float = 0.21
    _rom_correction_coefs: list[float] = [
        0.000201694,
        -0.030051020,
        0.03197279,
    ]
    _rm1_coefs: list[float] = [0.7351, 6]

    def __init__(
        self,
        time_s: NDArray[np.floating],
        motor_position_rad: NDArray[np.floating],
        motor_load_nm: NDArray[np.floating],
    ):
        super().__init__(time_s, motor_position_rad, motor_load_nm)


#! CONSTANTS


G = 9.80665

__all__ = [
    "PRODUCTS",
    "ChestPress",
    "LegPress",
    "LegExtension",
    "LegPressREV",
    "AdjustablePulleyREV",
    "LegExtensionREV",
]

PRODUCTS = {
    "CHEST PRESS": ChestPress,
    "LEG PRESS": LegPress,
    "LEG EXTENSION": LegExtension,
    "LEG PRESS REV": LegPressREV,
    "ADJUSTABLE PULLEY REV": AdjustablePulleyREV,
    "LEG EXTENSION REV": LegExtensionREV,
}
