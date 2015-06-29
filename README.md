# ReSplit
This is a recording splitter. It takes an input directory with PCM recordings and splits them using a voice activity detection mechanism into output dir.

It expects the PCM recordings to be 2 channel, where each channel contains one side of a dialog.
The input directory structure is preserved under the output directory, where each input file corresponds to an output folder of the same name in the output directory.

# Usage

  python recording_splitter.py where/is/input_dir where/is/output_dir