import os
from typing import List, Tuple


def intervals_to_rttm(intervals: List[Tuple[float, float]], rttm_filename: str):
    """
    Convert a list of speech intervals to RTTM format.

    Args:
        intervals (List[Tuple[float, float]]): List of voice segments as (start_time, end_time) pairs in seconds.

    Returns:
        str: RTTM formatted string.
        rttm_filename: Path to output RTTM file.

    Example:
        >>> rttm = intervals_to_rttm([(0.5, 1.2), (1.8, 3.4)])
        >>> print(rttm)
        SPEAKER audio 1 0.5 0.7 <NA> <NA> speaker <NA>
        SPEAKER audio 1 1.8 1.6 <NA> <NA> speaker <NA>
    """

    rttm = ""
    for start, duration in intervals:
        rttm += f"SPEAKER audio 1 {start} {duration} <NA> <NA> speaker <NA>\n"

    if os.path.exists(rttm_filename):
        raise FileExistsError(f"File {rttm_filename} already exists.")
    with open(rttm_filename, "w") as f:
        f.write(rttm)


def intervals_to_csv(
    intervals: List[Tuple[float, float]], csv_filename: str, delimiter=","
):
    """
    Convert a list of speech intervals to CSV format.

    Args:
        intervals (List[Tuple[float, float]]): List of voice segments as (start_time, end_time) pairs in seconds.
        csv_filename (str): Path to output CSV file.

    Example:
        >>> intervals_to_csv([(0.5, 1.2), (1.8, 3.4)], "output.csv")
    """

    if os.path.exists(csv_filename):
        raise FileExistsError(f"File {csv_filename} already exists.")
    with open(csv_filename, "w") as f:
        f.write("start_time{0}end_time\n".format(delimiter))
        for start, end in intervals:
            f.write("{0:>4.2f}{1}{2:>4.2f}\n".format(start, delimiter, end))


def intervals_to_textgrid(
    intervals: List[Tuple[float, float]],
    textgrid_filename: str,
    duration: float,
    tier_name: str = "speech",
) -> None:
    """Export voice activity intervals to Praat TextGrid format.

    Args:
        intervals (List[Tuple[float, float]]): List of (start, end) times in seconds.
        filename (str): Output TextGrid file path.
        duration (float): Total duration of audio in seconds.
        tier_name (str, optional): Name of the interval tier. Defaults to "speech".
    """
    if os.path.exists(textgrid_filename):
        raise FileExistsError(f"File {textgrid_filename} already exists.")
    with open(textgrid_filename, "w", encoding="utf-8") as f:
        # Write header
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write("xmin = 0\n")
        f.write(f"xmax = {duration:.6f}\n")
        f.write("tiers? <exists>\n")
        f.write("size = 1\n")
        f.write("item []:\n")

        # Write tier info
        f.write("    item [1]:\n")
        f.write('        class = "IntervalTier"\n')
        f.write(f'        name = "{tier_name}"\n')
        f.write("        xmin = 0\n")
        f.write(f"        xmax = {duration:.6f}\n")

        # Count intervals (including silence)
        total_intervals = len(intervals) * 2 + 1
        if intervals and intervals[0][0] == 0:
            total_intervals -= 1
        if intervals and intervals[-1][1] == duration:
            total_intervals -= 1

        f.write(f"        intervals: size = {total_intervals}\n")

        # Write intervals
        interval_idx = 1
        last_end = 0.0

        for start, end in intervals:
            # Add silence interval if needed
            if start > last_end:
                f.write(f"        intervals [{interval_idx}]:\n")
                f.write(f"            xmin = {last_end:.6f}\n")
                f.write(f"            xmax = {start:.6f}\n")
                f.write('            text = ""\n')
                interval_idx += 1

            # Add speech interval
            f.write(f"        intervals [{interval_idx}]:\n")
            f.write(f"            xmin = {start:.6f}\n")
            f.write(f"            xmax = {end:.6f}\n")
            f.write('            text = "speech"\n')
            interval_idx += 1
            last_end = end

        # Add final silence if needed
        if last_end < duration:
            f.write(f"        intervals [{interval_idx}]:\n")
            f.write(f"            xmin = {last_end:.6f}\n")
            f.write(f"            xmax = {duration:.6f}\n")
            f.write('            text = ""\n')
