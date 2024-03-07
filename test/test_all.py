"""test the rslib library"""

#! IMPORTS


from os.path import dirname, join
from os import getcwd
from pandas import DataFrame
import sys

sys.path += [getcwd()]
import src as labio


#! CONSTANTS


PATH = join(dirname(__file__), "assets")
READ_PATH = join(PATH, "read")
WRITE_PATH = join(PATH, "write")


#! TEST CLASSES


def read_emt():
    try:
        emt_rd = labio.read_emt(join(READ_PATH, "read.emt"))
    except Exception:
        emt_rd = None
    assert isinstance(emt_rd, DataFrame)


def read_tdf():
    tdf = labio.read_tdf(join(READ_PATH, "read.tdf"))
    keys = [
        "CAMERA",
        "FORCE_PLATFORM",
        "GENERIC",
        "EMG",
        "EVENTS",
        "VOLUME",
        "VERSION",
    ]
    if isinstance(tdf, dict):
        for key, val in tdf.items():
            assert key in keys
            if key == "CAMERA":
                lbls = [
                    "TRACKED",
                    "RAW",
                    "PARAMETERS",
                    "CALIBRATION",
                    "CONFIGURATION",
                ]
            elif key == "FORCE_PLATFORM":
                lbls = [
                    "TRACKED",
                    "RAW",
                    "PARAMETERS",
                    "CALIBRATION",
                ]
            elif key == "GENERIC":
                lbls = ["DATA", "CALIBRATION"]
            else:
                lbls = []
            if val is not None and len(lbls) > 0:
                for inn in val:  # type: ignore
                    assert any(inn == i for i in lbls)


def read_trc():
    rdf = labio.read_trc(join(READ_PATH, "read.trc"))
    assert isinstance(rdf, DataFrame)


def write_trc():
    rdf = labio.read_trc(join(READ_PATH, "read.trc"))
    write_file = join(WRITE_PATH, "write.trc")
    labio.write_trc(write_file, rdf)
    wdf = labio.read_trc(write_file)
    assert isinstance(wdf, DataFrame)
    if isinstance(wdf, DataFrame):
        assert (rdf - wdf).sum().sum() == 0


def read_mot():
    rdf = labio.read_mot(join(READ_PATH, "read.mot"))
    assert isinstance(rdf, DataFrame)


def write_mot():
    rdf = labio.read_mot(join(READ_PATH, "read.mot"))
    write_file = join(WRITE_PATH, "write.mot")
    labio.write_mot(write_file, rdf)
    wdf = labio.read_mot(write_file)
    assert isinstance(wdf, DataFrame)
    if isinstance(wdf, DataFrame):
        assert (rdf - wdf).sum().sum() == 0


def read_cosmed_xlsx():
    rdf = labio.read_cosmed_xlsx(join(READ_PATH, "read.xlsx"))
    assert isinstance(rdf, tuple)
    assert len(rdf) == 2
    assert isinstance(rdf[0], DataFrame)
    assert isinstance(rdf[1], labio.Participant)
