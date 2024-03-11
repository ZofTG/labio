"""products module"""

#! imports


import copy
from os.path import exists
from abc import ABCMeta, abstractmethod
from types import NoneType
from .core import G, der1, der2, polyval, symmetry


#! exports


__all__ = [
    "PRODUCTS",
    "ChestPress",
    "LegPress",
    "LegPressREV",
    "LowRow",
    "VerticalTraction",
    "Product",
]


#! classes


class Product(metaclass=ABCMeta):
    """(Abstract) Product class object"""

    _raw_position_rad: list[float] | None
    _raw_torque_nm: list[float] | None
    _raw_time_s: list[float] | None
    _raw_symmetry: list[float] | None

    def __init__(
        self,
        time: list[float] | None = None,
        position: list[float] | None = None,
        torque: list[float] | None = None,
        symm: list[float] | None = None,
    ):
        # check the entries
        entries = [time, position, torque, symm]
        types = (int, float, NoneType)
        n_samples = []
        for entry in entries:
            if isinstance(entry, list):
                float_ok = list(map(lambda x: isinstance(x, types), entry))
                if not all(float_ok):
                    msg = f"all values in {entry} must be int or float."
                    raise ValueError(msg)
                n_samples += [len(entry)]
            elif not isinstance(entry, NoneType):
                raise TypeError(f"{entry} must be a list or None.")
        if len(n_samples) > 1:
            if not all(i == n_samples[0] for i in n_samples):
                raise ValueError("all entries must have the same length.")

        self.set_raw_time_s(time)
        self.set_raw_position_rad(position)
        self.set_raw_torque_nm(torque)
        self.set_raw_symmetry(symm)

    @property
    def position_m(self):
        """return the calculated position"""
        if self._raw_position_rad is None or self._raw_torque_nm is None:
            return None
        rad = self._raw_position_rad
        trq = self._raw_torque_nm
        pos = list(map(self._get_lever_position, rad, trq))
        samples = len(pos)
        return [pos[i] for i in range(1, samples - 1)]

    @property
    def lever_kgf(self):
        """return the calculated lever weight"""
        pos = self.position_m
        if pos is None:
            return None
        return list(map(self._get_lever_weight, pos))

    @property
    def inertia_kgf(self):
        """return the calculated inertia"""
        tmr = self._raw_time_s
        trq = self._raw_torque_nm
        pos = self._raw_position_rad
        if tmr is None or trq is None or pos is None:
            return None
        n_samples = len(tmr)
        tm0 = [tmr[i] for i in range(n_samples - 2)]
        tm1 = [tmr[i] for i in range(1, n_samples - 1)]
        tm2 = [tmr[i] for i in range(2, n_samples)]
        pos = list(map(self._get_lever_position, pos, trq))
        ps0 = [pos[i] for i in range(n_samples - 2)]
        ps1 = [pos[i] for i in range(1, n_samples - 1)]
        ps2 = [pos[i] for i in range(2, n_samples)]
        return list(map(self._get_lever_inertia, tm0, tm1, tm2, ps0, ps1, ps2))

    @property
    def symmetry(self):
        """return the calculated symmetry"""
        if self._raw_symmetry is None:
            return None
        n_samples = len(self._raw_symmetry)
        return [self._raw_symmetry[i] for i in range(1, n_samples - 1)]

    @property
    def motor_kgf(self):
        """return the motor load"""
        if self._raw_torque_nm is None:
            return None
        n_samples = len(self._raw_torque_nm)
        trq = [self._raw_torque_nm[i] for i in range(1, n_samples - 1)]
        return list(map(self._get_load, trq))

    @property
    def power_w(self):
        """return the calculated power"""
        frz = self.load_kgf
        spd = self.speed_ms
        if frz is None or spd is None:
            return None
        return list(map(lambda x, y: x * y * G, frz, spd))

    @property
    def time_s(self):
        """return the calculated time"""
        if self._raw_time_s is None:
            return None
        n_samples = len(self._raw_time_s)
        return [self._raw_time_s[i] for i in range(1, n_samples - 1)]

    @property
    def speed_ms(self):
        """return the calculated speed"""
        trq = self._raw_torque_nm
        pos = self._raw_position_rad
        tim = self._raw_time_s
        if tim is None or pos is None or trq is None:
            return None
        n_samples = len(tim)
        tm0 = [tim[i] for i in range(n_samples - 2)]
        tm2 = [tim[i] for i in range(2, n_samples)]
        pos = list(map(self._get_lever_position, pos, trq))
        ps0 = [pos[i] for i in range(n_samples - 2)]
        ps2 = [pos[i] for i in range(2, n_samples)]
        return list(map(der1, ps0, ps2, tm0, tm2))

    @property
    def load_kgf(self):
        """return the calculated load"""
        mtr = self.motor_kgf
        lvr = self.lever_kgf
        if mtr is None or lvr is None:
            return None
        return list(map(lambda x, y: x + y, mtr, lvr))

    def slice(
        self,
        start: int | None,
        stop: int | None,
    ):
        """
        slice the object in a subset

        Parameters
        ----------
        start: int | None,
            the starting point of the slice. If None, the slice will consider
            the first sample as start.

        stop: int | None,
            the ending point of the slice. If None, the slice will consider
            the last sample as stop.

        inplace: bool = False,
            if True the actual object is sliced. If False a sliced copy is
            returned.

        Returns
        -------
        sliced: Product
            a slice of the object.
        """
        if self.is_empty():
            return self.copy()
        init = max(0, 0 if start is None else (start - 1))
        end = min(
            len(self.time_s) - 1,  # type: ignore
            len(self.time_s) if stop is None else (stop + 1),  # type: ignore
        )  # type: ignore
        raw_index = list(range(init, end))
        obj = self.copy()
        trq = [self._raw_torque_nm[i] for i in raw_index]  # type: ignore
        obj.set_raw_torque_nm(trq)  # type: ignore
        pos = [self._raw_position_rad[i] for i in raw_index]  # type: ignore
        obj.set_raw_position_rad(pos)  # type: ignore
        tim = [self._raw_time_s[i] for i in raw_index]  # type: ignore
        obj.set_raw_time_s(tim)  # type: ignore
        sym = [self._raw_symmetry[i] for i in raw_index]  # type: ignore
        obj.set_raw_symmetry(sym)  # type: ignore
        return obj

    def set_raw_torque_nm(
        self,
        torque_nm: list[float] | None,
    ):
        """
        set the input torque in Nm

        Parameters
        ----------
        torque_nm: list[float] | None
            set the target torque
        """
        self._raw_torque_nm = torque_nm

    def set_raw_position_rad(
        self,
        position_rad: list[float] | None,
    ):
        """
        set the input position in rad

        Parameters
        ----------
        position_rad: list[float] | None
            set the target position
        """
        self._raw_position_rad = position_rad

    def set_raw_time_s(
        self,
        time_s: list[float] | None,
    ):
        """
        set the input time in s

        Parameters
        ----------
        time_s: list[float] | None
            set the target time
        """
        self._raw_time_s = time_s

    def set_raw_symmetry(
        self,
        symm: list[float] | None,
    ):
        """
        set the input symmetry

        Parameters
        ----------
        symmetry: list[float] | None
            set the target symmetry
        """
        self._raw_symmetry = symm

    def values(self):
        """return the values contained by the object"""
        return iter(
            [
                self.time_s,
                self.position_m,
                self.load_kgf,
                self.power_w,
                self.speed_ms,
                self.symmetry,
                self.inertia_kgf,
                self.lever_kgf,
            ]
        )

    def keys(self):
        """return the keys contained by the object"""
        return iter(
            [
                "time_s",
                "position_m",
                "load_kgf",
                "power_w",
                "speed_ms",
                "symmetry_%",
                "inertia_kgf",
                "lever_kgf",
            ]
        )

    def items(self):
        """return the pairs of key, value contained by the object"""
        objs = [(i, j) for i, j in zip(list(self.keys()), list(self.values()))]
        return iter(objs)

    def copy(self):
        """make a copy of the object"""
        return copy.deepcopy(self)

    @classmethod
    def from_file(
        cls,
        file: str,
        has_cells: bool = False,
    ):
        """
        read raw data from file

        Parameters
        ----------
        file : str
            the path to the file

        has_cells : bool
            if True load cell data are searched within the input file

        Returns
        -------
        raw: dict[str, list[float]]
            a dictionary with keys "time", "position", "torque", "symmetry"

        Raises
        ------
        RuntimeError
            in case something went wrong during the data processing

        AssertionError
            in case the provided file was not correct.

        Note
        ----
        This function is for internal use only and serves as wrapper to
        simplify the implementation of the "from_file" method.
        """

        # check the inputs
        msg = "incorrect file."
        assert isinstance(file, str), msg
        assert exists(file), msg
        assert file.endswith(".txt") or file.endswith(".csv"), msg

        # get the data
        with open(file, "r", encoding="utf-8") as buf:
            obj = [i.replace(",", ".").split("|") for i in buf.readlines()]

        # get the relevant indices
        if has_cells:
            lbls = ["Cell1", "Cell2"]
            cells = [i for i, v in enumerate(obj[0]) if v in lbls]
        else:
            cells = []
        indices = [5, 2, 0] + cells

        # extract the raw data
        arr: list[list[float]] = []
        obj.pop(0)
        for i in indices:
            arr += [[float(obj[j][i]) for j in range(len(obj))]]
        if len(cells) > 0:
            pos, load, time, cell1, cell2 = arr
            sym = list(map(symmetry, cell1, cell2))
        else:
            pos, load, time = arr
            sym = [None for i in range(len(pos))]

        # merge into dict
        labels = ["time", "position", "torque", "symm"]
        values = [time, pos, load, sym]
        return cls(**dict(zip(labels, values)))

    @property
    @abstractmethod
    def rom_correction_coefs(
        self,
    ) -> list[float]:
        """
        the correction coefficients of the position readings of the motor
        according to the torque generated by the motor.

        Returns
        -------
        coefs: list[float]
            a list of 3 float defining the coefficients of the 2nd order
            polynomial which corrects the rom positioning according to
            the raw torque (in Nm) of read by the motor.
        """
        return NotImplementedError

    @property
    @abstractmethod
    def pulley_radius(
        self,
    ) -> float:
        """
        return the radius of the pulley from which the motor is directly
        linked to.

        Returns
        -------
        radius: float
            the radius of the pulley in m.
        """
        return NotImplementedError

    @property
    @abstractmethod
    def lever_radius(
        self,
    ) -> float:
        """
        return the radius of the levers moved by the user.

        Returns
        -------
        radius: float
            the radius of the lever moved by the user in m.
        """
        return NotImplementedError

    @property
    @abstractmethod
    def lever_com_radius(
        self,
    ) -> float:
        """
        return the radius generated by the CoM of the lever with respect to
        the fuclrum around with it rotates.

        Returns
        -------
        radius: float
            the radius of the lever moved by the user in m.
        """
        return NotImplementedError

    @property
    @abstractmethod
    def lever_number(
        self,
    ) -> int:
        """
        return the number of levers of the product
        """
        return NotImplementedError

    @property
    @abstractmethod
    def load_conversion_coefs(
        self,
    ) -> list[float]:
        """
        the coefficients to be applied to the torque in Nm to extract the
        generated load in kgf.

        Returns
        -------
        coefs: list[float]
            a list of 2 floats defining the coefficients of the 1st order
            polynomial which converts the raw torque positioning to the output
            load.
        """
        return NotImplementedError

    @property
    @abstractmethod
    def rm1_coefs(
        self,
    ):
        """
        return the conversion coefficient for the 1RM estimation.
        """
        return NotImplementedError

    @abstractmethod
    def _get_cam_correction(
        self,
        position: float,
        rom0: float,
        rom1: float,
    ) -> float:
        """
        return the isotonic cam correction for the actual position.

        Parameters
        ----------
        position: float
            the istantaneous position at which the correction is required.

        rom0: float
            the lower end of the user's ROM.

        rom1: float
            the upper end of the user's ROM.

        Returns
        -------
        correction: float
            the correction to be applied to the output force according to the
            isotonic cam profile.
        """
        return NotImplementedError

    @abstractmethod
    def _get_lever_weight(
        self,
        position: float | int,
    ) -> float:
        """
        return the lever weight according to the actual position in m

        Parameters
        ----------
        position : list[float] | float | int
            the instantaneous (corrected) lever position in m

        Returns
        -------
        weight: float
            the weight of the lever
        """
        return NotImplementedError

    @property
    @abstractmethod
    def name(self):
        """the name of the product"""
        return NotImplementedError

    def _get_lever_position(
        self,
        position: float | int,
        torque: float | int,
    ):
        """
        return the position in mm

        Parameters
        ----------
        position : list[float | int]
            the raw postion in rad

        torque : list[float | int]
            the measured torque from the inverter

        Returns
        -------
        lever_position: list[float]
            the lever position in m
        """
        cor = max(polyval(self.rom_correction_coefs, abs(torque)), 0)
        return (cor + position) * self.pulley_radius / self.lever_radius

    def _get_lever_inertia(
        self,
        time0: float | int,
        time1: float | int,
        time2: float | int,
        pos0: float | int,
        pos1: float | int,
        pos2: float | int,
    ) -> float:
        """
        return the lever inertial force in kgf according to the actual
        position in m

        Parameters
        ----------
        time0 : float | int
            the instant t-1

        time1 : float | int
            the actual instant

        time2 : float | int
            the next time instant

        pos0 : float | int
            the instantaneous (corrected) lever position in m at the instant t-1

        pos1 : float | int
            the instantaneous (corrected) lever position in m at actual instant

        pos2 : float | int
            the instantaneous (corrected) lever position in m at next time
            instant.

        Returns
        -------
        inertial_force: float
            the inertial force in kgf
        """
        ang_acc = der2(pos0, pos1, pos2, time0, time1, time2) / self.lever_radius
        inertia = self._get_lever_weight(pos1) * ang_acc * self.lever_com_radius
        return inertia * self.lever_number

    def _get_load(
        self,
        torque: float | int,
    ):
        """
        convert the load corresponding to the torque read by the motor.

        Parameters
        ----------
        torque : float | int,
            the measured torque from the inverter

        Returns
        -------
        load: float,
            the load corresponding to the entered torque in kgf
        """
        return polyval(self.load_conversion_coefs, torque)

    def is_empty(self):
        """
        check whether the product contains data
        """
        return any(v is None for v in self.values())


class ChestPress(Product):
    """Chest press class object"""

    _CAMME_RATIO = 0.74
    _SPRING_CORRECTION = 1.15

    @property
    def rom_correction_coefs(
        self,
    ):
        """
        the correction coefficients of the position readings of the motor
        according to the torque generated by the motor.

        Returns
        -------
        coefs: list[float]
            a list of 3 float defining the coefficients of the 2nd order
            polynomial which corrects the rom positioning according to
            the raw torque (in Nm) of read by the motor.
        """
        return [-0.0000970270993668, 0.0284363503605837, -0.1454105176656738]

    @property
    def pulley_radius(
        self,
    ):
        """
        return the radius of the pulley from which the motor is directly
        linked to.

        Returns
        -------
        radius: float
            the radius of the pulley in m.
        """
        return 0.054

    @property
    def lever_radius(
        self,
    ):
        """
        return the radius of the levers moved by the user.

        Returns
        -------
        radius: float
            the radius of the lever moved by the user in m.
        """
        return 0.87489882

    @property
    def lever_com_radius(
        self,
    ):
        """
        return the radius generated by the CoM of the lever with respect to
        the fuclrum around with it rotates.

        Returns
        -------
        radius: float
            the radius of the lever moved by the user in m.
        """
        return 0.592188472

    @property
    def load_conversion_coefs(
        self,
    ):
        """
        the coefficients to be applied to the torque in Nm to extract the
        generated load in kgf.

        Returns
        -------
        coefs: list[float]
            a list of 2 floats defining the coefficients of the 1st order
            polynomial which converts the raw torque positioning to the output
            load.
        """
        coefs = self._CAMME_RATIO / self._SPRING_CORRECTION
        coefs = coefs / self.pulley_radius / G
        return [coefs, 0]

    @property
    def lever_number(
        self,
    ):
        """
        return the number of levers of the product
        """
        return int(2)

    def _get_lever_weight(
        self,
        position: float | int,
    ):
        """
        return the lever weight according to the actual position in m

        Parameters
        ----------
        position : list[float] | float | int
            the instantaneous (corrected) lever position in m

        Returns
        -------
        weight: float
            the weight of the lever
        """
        return 4.0 * self.lever_number

    @classmethod
    def from_file(
        cls,
        file: str,
    ):
        """
        read raw data from file

        Parameters
        ----------
        file : str
            the path to the file
        """
        return super().from_file(file, False)

    def _get_cam_correction(
        self,
        position: float,
        rom0: float,
        rom1: float,
    ) -> float:
        """
        return the isotonic cam correction for the actual position.

        Parameters
        ----------
        position: float
            the istantaneous position at which the correction is required.

        rom0: float
            the lower end of the user's ROM.

        rom1: float
            the upper end of the user's ROM.

        Returns
        -------
        correction: float
            the correction to be applied to the output force according to the
            isotonic cam profile.
        """
        return 1.0

    @property
    def rm1_coefs(
        self,
    ):
        """
        return the conversion coefficient for the 1RM estimation.
        """
        return [0.96217, 2.97201]

    @property
    def name(self):
        """the name of the product"""
        return "CHEST PRESS"


class VerticalTraction(Product):
    """Vertical Traction class object"""

    _CAMME_RATIO = 0.6
    _SPRING_CORRECTION = 1.1

    @property
    def rom_correction_coefs(
        self,
    ):
        """
        the correction coefficients of the position readings of the motor
        according to the torque generated by the motor.

        Returns
        -------
        coefs: list[float]
            a list of 3 float defining the coefficients of the 2nd order
            polynomial which corrects the rom positioning according to
            the raw torque (in Nm) of read by the motor.
        """
        return [-0.0000593206445389, 0.0259836372360176, -0.0980811957930012]

    @property
    def pulley_radius(
        self,
    ):
        """
        return the radius of the pulley from which the motor is directly
        linked to.

        Returns
        -------
        radius: float
            the radius of the pulley in m.
        """
        return 0.054

    @property
    def lever_radius(
        self,
    ):
        """
        return the radius of the levers moved by the user.

        Returns
        -------
        radius: float
            the radius of the lever moved by the user in m.
        """
        return 0.7032

    @property
    def lever_com_radius(
        self,
    ):
        """
        return the radius generated by the CoM of the lever with respect to
        the fuclrum around which it rotates.

        Returns
        -------
        radius: float
            the radius of the lever moved by the user in m.
        """
        return 0.0464

    @property
    def lever_number(
        self,
    ):
        """
        return the number of levers of the product
        """
        return int(2)

    @property
    def load_conversion_coefs(
        self,
    ):
        """
        the coefficients to be applied to the torque in Nm to extract the
        generated load in kgf.

        Returns
        -------
        coefs: list[float]
            a list of 2 floats defining the coefficients of the 1st order
            polynomial which converts the raw torque positioning to the output
            load.
        """
        coefs = self._CAMME_RATIO / self._SPRING_CORRECTION
        coefs = coefs / self.pulley_radius / G
        return [coefs, 0]

    def _get_lever_weight(
        self,
        position: float | int,
    ):
        """
        return the lever weight according to the actual position in m

        Parameters
        ----------
        position : list[float] | float | int
            the instantaneous (corrected) lever position in m

        Returns
        -------
        weight: float
            the weight of the lever
        """
        return -3.0 * self.lever_number

    @classmethod
    def from_file(
        cls,
        file: str,
    ):
        """
        read raw data from file

        Parameters
        ----------
        file : str
            the path to the file
        """
        return super().from_file(file, False)

    def _get_cam_correction(
        self,
        position: float,
        rom0: float,
        rom1: float,
    ) -> float:
        """
        return the isotonic cam correction for the actual position.

        Parameters
        ----------
        position: float
            the istantaneous position at which the correction is required.

        rom0: float
            the lower end of the user's ROM.

        rom1: float
            the upper end of the user's ROM.

        Returns
        -------
        correction: float
            the correction to be applied to the output force according to the
            isotonic cam profile.
        """
        betas = [-0.000000000192, -0.000000366669, +0.000154455574]
        betas += [-0.048948857016, +0.105833758047]
        pos = min(max(position, rom0), rom1)
        return polyval(betas, pos)

    @property
    def rm1_coefs(
        self,
    ):
        """
        return the conversion coefficient for the 1RM estimation.
        """
        return [1.05070, 1.00000]

    @property
    def name(self):
        """the name of the product"""
        return "VERTICAL TRACTION"


class LowRow(Product):
    """Low Row class object"""

    _CAMME_RATIO = 0.64
    _SPRING_CORRECTION = 1.0

    @property
    def rom_correction_coefs(
        self,
    ):
        """
        the correction coefficients of the position readings of the motor
        according to the torque generated by the motor.

        Returns
        -------
        coefs: list[float]
            a list of 3 float defining the coefficients of the 2nd order
            polynomial which corrects the rom positioning according to
            the raw torque (in Nm) of read by the motor.
        """
        return [0, 0, 0]

    @property
    def pulley_radius(
        self,
    ):
        """
        return the radius of the pulley from which the motor is directly
        linked to.

        Returns
        -------
        radius: float
            the radius of the pulley in m.
        """
        return 0.054

    @property
    def lever_radius(
        self,
    ):
        """
        return the radius of the levers moved by the user.

        Returns
        -------
        radius: float
            the radius of the lever moved by the user in m.
        """
        return 0.7032

    @property
    def lever_com_radius(
        self,
    ):
        """
        return the radius generated by the CoM of the lever with respect to
        the fuclrum around with it rotates.

        Returns
        -------
        radius: float
            the radius of the lever moved by the user in m.
        """
        return 0.0464

    @property
    def lever_number(
        self,
    ):
        """
        return the number of levers of the product
        """
        return int(1)

    @property
    def load_conversion_coefs(
        self,
    ):
        """
        the coefficients to be applied to the torque in Nm to extract the
        generated load in kgf.

        Returns
        -------
        coefs: list[float]
            a list of 2 floats defining the coefficients of the 1st order
            polynomial which converts the raw torque positioning to the output
            load.
        """
        coefs = self._CAMME_RATIO / self._SPRING_CORRECTION
        coefs = coefs / self.pulley_radius / G
        return [coefs, 0]

    def _get_lever_weight(
        self,
        position: float | int,
    ):
        """
        return the lever weight according to the actual position in m

        Parameters
        ----------
        position : list[float] | float | int
            the instantaneous (corrected) lever position in m

        Returns
        -------
        weight: float
            the weight of the lever
        """
        return 5 * self.lever_number

    @classmethod
    def from_file(
        cls,
        file: str,
    ):
        """
        read raw data from file

        Parameters
        ----------
        file : str
            the path to the file
        """
        return super().from_file(file, False)

    def _get_cam_correction(
        self,
        position: float,
        rom0: float,
        rom1: float,
    ) -> float:
        """
        return the isotonic cam correction for the actual position.

        Parameters
        ----------
        position: float
            the istantaneous position at which the correction is required.

        rom0: float
            the lower end of the user's ROM.

        rom1: float
            the upper end of the user's ROM.

        Returns
        -------
        correction: float
            the correction to be applied to the output force according to the
            isotonic cam profile.
        """
        betas = [0.000000002839, -0.000002774527, 0.000233480221]
        betas += [0.059293138715, 8.971309619962]
        pos = min(max(position, rom0), rom1)
        return polyval(betas, pos)

    @property
    def rm1_coefs(
        self,
    ):
        """
        return the conversion coefficient for the 1RM estimation.
        """
        return [0.69234, 3.38309]

    @property
    def name(self):
        """the name of the product"""
        return "LOW ROW"


class LegPress(Product):
    """Leg Press class object"""

    _CAMME_RATIO = 1
    _SPRING_CORRECTION = 1

    @property
    def rom_correction_coefs(
        self,
    ):
        """
        the correction coefficients of the position readings of the motor
        according to the torque generated by the motor.

        Returns
        -------
        coefs: list[float]
            a list of 3 float defining the coefficients of the 2nd order
            polynomial which corrects the rom positioning according to
            the raw torque (in Nm) of read by the motor.
        """
        return [-0.0000594298355666, 0.0155680740573513, -0.0022758912872085]

    @property
    def pulley_radius(
        self,
    ):
        """
        return the radius of the pulley from which the motor is directly
        linked to.

        Returns
        -------
        radius: float
            the radius of the pulley in m.
        """
        return 0.08175

    @property
    def lever_radius(
        self,
    ):
        """
        return the radius of the levers moved by the user.

        Returns
        -------
        radius: float
            the radius of the lever moved by the user in m.
        """
        return 1.0

    @property
    def lever_com_radius(
        self,
    ):
        """
        return the radius generated by the CoM of the lever with respect to
        the fuclrum around with it rotates.

        Returns
        -------
        radius: float
            the radius of the lever moved by the user in m.
        """
        return 1.0

    @property
    def lever_number(
        self,
    ):
        """
        return the number of levers of the product
        """
        return int(1)

    @property
    def load_conversion_coefs(
        self,
    ):
        """
        the coefficients to be applied to the torque in Nm to extract the
        generated load in kgf.

        Returns
        -------
        coefs: list[float]
            a list of 2 floats defining the coefficients of the 1st order
            polynomial which converts the raw torque positioning to the output
            load.
        """
        coefs = self._CAMME_RATIO / self._SPRING_CORRECTION
        coefs = coefs / self.pulley_radius / G
        return [coefs, 0]

    def _get_lever_weight(
        self,
        position: float | int,
    ):
        """
        return the lever weight according to the actual position in m

        Parameters
        ----------
        position : list[float] | float | int
            the instantaneous (corrected) lever position in m

        Returns
        -------
        weight: float
            the weight of the lever
        """
        # l_weight = 77.4  # kg
        # rail_incline = 10  # degrees
        # true_weight = l_weight * float(sin(rail_incline / 180 * pi)) = 24.5
        return 24.5
        # return  9 + 85 * float(sin(10 / 180 * pi))

    @classmethod
    def from_file(
        cls,
        file: str,
    ):
        """
        read raw data from file

        Parameters
        ----------
        file : str
            the path to the file
        """
        return super().from_file(file, False)

    def _get_cam_correction(
        self,
        position: float,
        rom0: float,
        rom1: float,
    ) -> float:
        """
        return the isotonic cam correction for the actual position.

        Parameters
        ----------
        position: float
            the istantaneous position at which the correction is required.

        rom0: float
            the lower end of the user's ROM.

        rom1: float
            the upper end of the user's ROM.

        Returns
        -------
        correction: float
            the correction to be applied to the output force according to the
            isotonic cam profile.
        """
        pos = min(max(position, rom0), rom1) / (rom1 - rom0)
        betas = [0.4, 0.6]
        return polyval(betas, pos)

    @property
    def rm1_coefs(
        self,
    ):
        """
        return the conversion coefficient for the 1RM estimation.
        """
        return [0.65705, 9.17845]

    @property
    def name(self):
        """the name of the product"""
        return "LEG PRESS"


class LegPressREV(LegPress):
    """Leg Press REV class object"""

    @property
    def name(self):
        """the name of the product"""
        return "LEG PRESS REV"


#! constants


PRODUCTS = {
    "CHEST PRESS": ChestPress,
    "LEG PRESS": LegPress,
    "LEG PRESS REV": LegPressREV,
    "LOW ROW": LowRow,
    "VERTICAL TRACTION": VerticalTraction,
}
