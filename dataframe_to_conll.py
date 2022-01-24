import pandas as pd

infile_path = "..\data\SEM-2012-SharedTask-CD-SCO-training-simple.v2_SHORT-out.txt"
outfile_path = "..\data\SEM-2012-SharedTask-CD-SCO-training-simple.v2_SHORT-out-CONLL.conll"
dataframe = pd.read_csv(infile_path)

def convert_dataframe_to_conll(dataframe, outfile_path, delimiter= '\t'):
    """
    Convert the rows of a dataframe into conll format and output a conll file

    :param dataframe: data as a pandas dataframe
    :param outfile_path: path of the output file
    :param delimiter: delimiter to separate the columns of the dataframe in the conll format
    """

    with open(outfile_path, 'w') as outputfile:
        for i in range(len(dataframe) - 1):
            row = dataframe.iloc[i]
            row = [str(j) for j in row]
            row = delimiter.join(row)

            if dataframe.loc[i + 1, "token_nr"] == 0:
                outputfile.write(row + "\n")
                outputfile.write("\n")
            else:
                outputfile.write(row + "\n")


convert_dataframe_to_conll(dataframe, outfile_path)