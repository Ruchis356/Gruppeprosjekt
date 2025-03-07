
Har filtrert bort de fleste filer og mapper (?)

Remove hardcoding where possible


## Ting vi mangler:

1. Enhetstesting - må skrives for hver funksjon (4 per i dag)
    - Har enhetstestene beskrivende navn som dokumenterer hva testene gjør?
- Tas det hensyn til både positive og negative tilfeller?
- Er testdekningen god nok?
- Har gode beskrivende navn på testene
- Har enhetstester for de viktigste funksjonene
- Har helt greie negative tester (viser at kandidaten har forstått hovedpoenget med positive/negative tester)

Bruk unittest eller pytest for testing. Vi burde teste hver "funksjon", dvs get_met, get_nilu, missing_value, og show_zeroes





## Areas of improvement:

1. Use list comprehensions to simplify some code?

Instead of a for loop:
zero_indices = [index for index in df.index if df.loc[index, col] == 0]

Look for other opportunities to replace loops with list comprehensions 
(e.g., when creating lists or filtering data)


2. pandasql to manipulate data with SQL-like queries?

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

query = """
SELECT * FROM df_weather WHERE Temperature > 20
"""
result = pysqldf(query)
print(result)