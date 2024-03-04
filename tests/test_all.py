"""test the rslib library"""

#! IMPORTS


from os.path import dirname, join
import sys

sys.path += [join(dirname(dirname(__file__)), "src")]
import btspy


#! FUNCTIONS


def test_io():
    """test input-output module"""

    # setup the data path
    READ_PATH = join(dirname(__file__), "assets", "read")

    # read_emt
    print("\nTESTING EMT DATA READING")
    emt_rd = btspy.read_emt(join(READ_PATH, "read.emt"))
    print(emt_rd)

    # read_tdf
    print("TESTING TDF DATA READING")
    tdf_rd = btspy.read_tdf(join(READ_PATH, "read.tdf"))
    print(tdf_rd)


def test_all():
    """test all rslib functionalities"""
    test_io()


#! MAIN


if __name__ == "__main__":
    test_all()
    print("\n\nALL TESTS COMPLETED")
