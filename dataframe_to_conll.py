import pandas as pd

infile_path = "..\data\SEM-2012-SharedTask-CD-SCO-dev-simple.v2-out.txt"
outfile_path = infile_path.replace(".txt", "-CONLL.conll")
dataframe = pd.read_csv(infile_path, encoding="utf-8")

def convert_dataframe_to_conll(dataframe, outfile_path, delimiter= '\t'):
    """
    Convert the rows of a dataframe into conll format and output a conll file

    :param dataframe: data as a pandas dataframe
    :param outfile_path: path of the output file
    :param delimiter: delimiter to separate the columns of the dataframe in the conll format
    """
    with open(outfile_path, 'w') as outputfile:
        for i in range(len(dataframe)):

            # Select all columns sent_id for each row
            row = dataframe.loc[i, dataframe.columns != "sent_id"]
            row = [str(j) for j in row]
            row = delimiter.join(row)

            if (dataframe.loc[i, "token_nr"] == 0) and (i != 0):
                outputfile.write("\n")
                outputfile.write(row + "\n")
            else:
                outputfile.write(row + "\n")


convert_dataframe_to_conll(dataframe, outfile_path)