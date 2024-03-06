"""
io._util

a library containing functions to be used for reading and writing files.
These functions are thought to be used internally and not directly from the
user.

Functions
---------
check_writing_file
    check the provided filename and rename it if required.

assert_file_extension
    check the validity of the input path file to be a str with the provided
    extension.
"""

#! IMPORTS


from datetime import date, datetime
from os.path import exists
from tkinter.messagebox import askyesno

import pandas as pd
from numpy import ndarray, unique

__all__ = [
    "check_entry",
    "check_writing_file",
    "assert_file_extension",
    "Participant",
]


#! FUNCTIONS


def check_entry(
    entry: object,
    mask: ndarray,
):
    """
    check a given object to be a pandas DataFrame with the "mask" structure of
    indices and columns.

    Parameters
    ----------
    entry : object
        the object to be checked

    mask : ndarray
        the column mask to be controlled. The mask has to match all the columns
        contained by levels at index > 1.

    Raises
    ------
    TypeError
        "entry must be a pandas DataFrame."
        In case the entry is not a pandas.DataFrame.

    TypeError
        "entry columns must be a pandas MultiIndex."
        In case the entry columns are not a pandas.MultiIndex instance.

    TypeError
        "entry columns must contain {mask}."
        In case the entry columns does not match with the provided mask.

    TypeError
        "entry index must be a pandas Index."
        In case the index of the entry is not a pandas.Index
    """
    if not isinstance(entry, pd.DataFrame):
        raise TypeError("entry must be a pandas DataFrame.")
    if not isinstance(entry.columns, pd.MultiIndex):
        raise TypeError("entry columns must be a pandas MultiIndex.")
    umask = unique(mask.astype(str), axis=0)
    for lbl in unique(entry.columns.get_level_values(0)):
        imask = entry[lbl].columns.to_frame().values.astype(str)
        imask = unique(imask, axis=0)
        if not (imask == umask).all():
            raise TypeError(f"entry columns must contain {mask}.")
    if not isinstance(entry.index, pd.Index):
        raise TypeError("entry index must be a pandas Index.")


def check_writing_file(
    file: str,
):
    """
    check the provided filename and rename it if required.

    Parameters
    ----------
    file : str
        the file path

    Returns
    -------
    filename: str
        the file (renamed if required).
    """
    ext = file.rsplit(".", 1)[-1]
    filename = file
    while exists(filename):
        msg = f"The {file} file already exist.\nDo you want to replace it?"
        yes = askyesno(title="Replace", message=msg)
        if yes:
            return filename
        filename = file.replace(f".{ext}", f"_1.{ext}")
    return filename


def assert_file_extension(
    path: str,
    ext: str,
):
    """
    check the validity of the input path file to be a str with the provided
    extension.

    Parameters
    ----------
    path : str
        the object to be checked

    ext : str
        the target file extension

    Raises
    ------
    err: AsserttionError
        in case the file is not a str or it does not exist or it does not have
        the provided extension.
    """
    assert isinstance(path, str), "path must be a str object."
    msg = path + f' must have "{ext}" extension.'
    assert path.rsplit(".", 1)[-1] == f"{ext}", msg


#! CLASSES


class Participant:
    """
    class containing all the data relevant to a participant.

    Parameters
    ----------
    surname: str | None = None
        the participant surname

    name: str | None = None
        the participant name

    gender: str | None = None
        the participant gender

    height: int | float | None = None
        the participant height

    weight: int | float | None = None
        the participant weight

    age: int | float | None = None
        the participant age

    birthdate: date | None = None
        the participant birth data

    recordingdate: date | None = None
        the the test recording date
    """

    # class variables
    _name = None
    _surname = None
    _gender = None
    _height = None
    _weight = None
    _birthdate = None
    _recordingdate = date  # type:ignore
    _units = {
        "fullname": "",
        "surname": "",
        "name": "",
        "gender": "",
        "height": "m",
        "weight": "kg",
        "bmi": "kg/m^2",
        "birthdate": "",
        "age": "years",
        "hrmax": "bpm",
        "recordingdate": "",
    }

    def __init__(
        self,
        surname: str | None = None,
        name: str | None = None,
        gender: str | None = None,
        height: int | float | None = None,
        weight: int | float | None = None,
        age: int | float | None = None,
        birthdate: date | None = None,
        recordingdate: date = datetime.now().date,  # type: ignore
    ):
        self.set_surname(surname)
        self.set_name(name)
        self.set_gender(gender)
        self.set_height((height / 100 if height is not None else height))
        self.set_weight(weight)
        self.set_age(age)
        self.set_birthdate(birthdate)
        self.set_recordingdate(recordingdate)

    def set_recordingdate(
        self,
        recordingdate: date | None,
    ):
        """
        set the test recording date.

        Parameters
        ----------
        recording_date: datetime.date | None
            the test recording date.
        """
        if recordingdate is not None:
            txt = "'recordingdate' must be a datetime.date or datetime.datetime."
            assert isinstance(recordingdate, (datetime, date)), txt
            if isinstance(recordingdate, datetime):
                self._recordingdate = recordingdate.date()
            else:
                self._recordingdate = recordingdate
        else:
            self._recordingdate = recordingdate

    def set_surname(
        self,
        surname: str | None,
    ):
        """
        set the participant surname.

        Parameters
        ----------
        surname: str | None,
            the surname of the participant.
        """
        if surname is not None:
            assert isinstance(surname, str), "'surname' must be a string."
        self._surname = surname

    def set_name(
        self,
        name: str | None,
    ):
        """
        set the participant name.

        Parameters
        ----------
        name: str | None
            the name of the participant.
        """
        if name is not None:
            assert isinstance(name, str), "'name' must be a string."
        self._name = name

    def set_gender(
        self,
        gender: str | None,
    ):
        """
        set the participant gender.

        Parameters
        ----------
        gender: str | None
            the gender of the participant.
        """
        if gender is not None:
            assert isinstance(gender, str), "'gender' must be a string."
        self._gender = gender

    def set_height(
        self,
        height: int | float | None,
    ):
        """
        set the participant height in meters.

        Parameters
        ----------
        height: int | float | None
            the height of the participant.
        """
        if height is not None:
            txt = "'height' must be a float or int."
            assert isinstance(height, (int, float)), txt
        self._height = height

    def set_weight(
        self,
        weight: int | float | None,
    ):
        """
        set the participant weight in kg.

        Parameters
        ----------
        weight: int | float | None
            the weight of the participant.
        """
        if weight is not None:
            txt = "'weight' must be a float or int."
            assert isinstance(weight, (int, float)), txt
        self._weight = weight

    def set_age(
        self,
        age: int | float | None,
    ):
        """
        set the participant age in years.


        Parameters
        ----------
        age: int | float | None,
            the age of the participant.
        """
        if age is not None:
            txt = "'age' must be a float or int."
            assert isinstance(age, (int, float)), txt
        self._age = age

    def set_birthdate(
        self,
        birthdate: date | None,
    ):
        """
        set the participant birth_date.

        Parameters
        ----------
        birth_date: datetime.date | None
            the birth date of the participant.
        """
        if birthdate is not None:
            txt = "'birth_date' must be a datetime.date or datetime.datetime."
            assert isinstance(birthdate, (datetime, date)), txt
            if isinstance(birthdate, datetime):
                self._birthdate = birthdate.date()
            else:
                self._birthdate = birthdate
        else:
            self._birthdate = birthdate

    @property
    def surname(self):
        """get the participant surname"""
        return self._surname

    @property
    def name(self):
        """get the participant name"""
        return self._name

    @property
    def gender(self):
        """get the participant gender"""
        return self._gender

    @property
    def height(self):
        """get the participant height in meter"""
        return self._height

    @property
    def weight(self):
        """get the participant weight in kg"""
        return self._weight

    @property
    def birthdate(self):
        """get the participant birth date"""
        return self._birthdate

    @property
    def recordingdate(self):
        """get the test recording date"""
        return self._recordingdate

    @property
    def bmi(self):
        """get the participant BMI in kg/m^2"""
        if self.height is None or self.weight is None:
            return None
        return self.weight / (self.height**2)

    @property
    def fullname(self):
        """
        get the participant full name.
        """
        return f"{self.surname} {self.name}"

    @property
    def age(self):
        """
        get the age of the participant in years
        """
        if self._age is not None:
            return self._age
        if self._birthdate is not None:
            return int((self._recordingdate - self._birthdate).days // 365)  # type: ignore
        return None

    @property
    def hrmax(self):
        """
        get the maximum theoretical heart rate according to Gellish.

        References
        ----------
        Gellish RL, Goslin BR, Olson RE, McDonald A, Russi GD, Moudgil VK.
            Longitudinal modeling of the relationship between age and maximal
            heart rate.
            Med Sci Sports Exerc. 2007;39(5):822-9.
            doi: 10.1097/mss.0b013e31803349c6.
        """
        if self.age is None:
            return None
        return 207 - 0.7 * self.age

    @property
    def units(self):
        """return the unit of measurement of the stored data."""
        return self._units

    def copy(self):
        """return a copy of the object."""
        return Participant(**{i: getattr(self, i) for i in self.units.keys()})

    @property
    def dict(self):
        """return a dict representation of self"""
        out = {}
        for i, v in self.units.items():
            out[i + ((" [" + v + "]") if v != "" else "")] = getattr(self, i)
        return out

    @property
    def series(self):
        """return a pandas.Series representation of self"""
        vals = [(i, v) for i, v in self.units.items()]
        vals = pd.MultiIndex.from_tuples(vals)
        return pd.Series(list(self.dict.values()), index=vals)

    @property
    def dataframe(self):
        """return a pandas.DataFrame representation of self"""
        return pd.DataFrame(self.series).T

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.dataframe.__str__()
