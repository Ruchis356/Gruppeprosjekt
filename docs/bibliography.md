## AI usage in code
For detailed information on the use of AI, see the document "detailed AI use.md"
Comments are written in code for sections where AI was used more explicitly.

## Stack overflow
Nafiul Islam (response to 'How do I call a function from another .py file?') link: https://stackoverflow.com/questions/20309456/how-do-i-call-a-function-from-another-py-file
    Excerpts used (for several files, inluding unittests and main.ipynb): 
        from file import function
        function(a, b)
Sayan Sil (response to 'Writing a pandas DataFrame to CSV file') link: https://stackoverflow.com/questions/16923281/writing-a-pandas-dataframe-to-csv-file
    Excerpt used (at the end of main.ipynb):
         df.to_csv(file_name, encoding='utf-8', index=False)
root (response to 'How do I get the row count of a Pandas DataFrame?') link: https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
    Excerpt used (in main, data_import, and data_handling):
        len(df.index)
        df.shape[0]
        df[df.columns[0]].count()
mozway (response to 'Error "'DataFrame' object has no attribute 'append'"') link: https://stackoverflow.com/questions/75956209/error-dataframe-object-has-no-attribute-append
    Excerpt used (in data_import)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
Andy Hayden (response to'Python pandas - Does read_csv keep file open?') link: https://stackoverflow.com/questions/29416968/python-pandas-does-read-csv-keep-file-open
    Excerpt used (in data_import):
        with open("myfile.csv", "w") as f:

## Other sources
escaaaaa (The unittests of the 2nd mandatory assignment in the subject 'object oriented programming') link: https://github.com/escaaaaaa/oving2/tree/main/Enhetstesting
    The general structure and some specific constructs were used as a basis for the unittests in this program.