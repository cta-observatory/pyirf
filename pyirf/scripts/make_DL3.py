"""Script to produce DL3 data from DL2 data and a configuration file.

Is it initially thought as a clean start based on old code for reproducing
EventDisplay DL3 data based on the latest release of the GADF format.

"""

import argparse


def main():

    # =========================================================================
    #                   READ INPUT FROM CLI AND CONFIGURATION FILE
    # =========================================================================

    # INPUT FROM CLI

    parser = argparse.ArgumentParser(description="Produce DL3 data from DL2.")

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="A configuration file like pyirf/resources/performance.yaml .",
    )

    parser.add_argument(
        "--obs_time",
        type=str,
        required=True,
        help="An observation time written as (value.unit), e.g. '50.h'",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Print debugging information."
    )

    # INPUT FROM THE CONFIGURATION FILE

    # =========================================================================

    args = parser.parse_args()

    print("MEH.")


if __name__ == "__main__":
    main()
