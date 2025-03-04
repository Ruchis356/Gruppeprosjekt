

First: 
- Merge branches
- Finish writing the readme file! (include a short description at the top)
- Go through assignment, general part and part 1
- Add missing parts of part 1




## Ting vi mangler:

1. Enhetstesting - må skrives for hver fil/notatbok (3 per i dag)
- Har enhetstestene beskrivende navn som dokumenterer hva testene gjør?
- Tas det hensyn til både positive og negative tilfeller?
- Er testdekningen god nok?
- Har gode beskrivende navn på testene
- Har enhetstester for de viktigste funksjonene
- Har helt greie negative tester (viser at kandidaten har forstått hovedpoenget med positive/negative tester)

Bruk unittest eller pytest for testing. Vi burde teste hver "funksjon", dvs get_met, get_nilu, missing_value, og show_zeroes

2. Versjonshåndtering/Versjonkontroll - Gå gjennom Git og kommenter hva vi ikke har gjort/skal gjøre fremover i README
- Er prosjektet underlagt versjonskontroll med sentral repro?
- Sjekkes det inn jevnlig?
- Gode commit-meldinger som beskriver kort hvilke endringer som er gjort/hvilke problem som er løst
- Har benyttet tags for å merke versjone (?)
- Har filtrert bort de fleste filer og mapper (?)
- Gode navngitte branches som sier hva den skal brukes til

3. Utdyp kildereferanser i selve koden(?):
- Kildereferanser bør inkludere informasjon om kildeautoritet, datakvalitet og tilgjengelighet, og bør presenteres i en klar og konsistent form.

4. Filhåndtering
- Leser fra tekstfil
- Begrenset eller ingen sjekk/kontroll av filformat/struktur
- Enkel håndtering av unntak
- Skriver til tekstfil
- Lukker filressurser på en trygg måte






## Areas of improvement:

1. Add to main? (Exploratory Data Analysis)

# Weather Data EDA
print("Weather Data Info:")
print(df_weather.info())
print("\nWeather Data Description:")
print(df_weather.describe())

# Air Quality Data EDA
print("\nAir Quality Data Info:")
print(df_quality.info())
print("\nAir Quality Data Description:")
print(df_quality.describe())


2. Add to handling?

Expand the missing_data method to include options for handling missing values (e.g., filling with a default value, interpolating, or dropping rows/columns).

def handle_missing_values(self, df, strategy='drop', fill_value=None):
    if strategy == 'drop':
        df = df.dropna()
    elif strategy == 'fill':
        df = df.fillna(fill_value)
    return df


3. Use list comprehensions to simplify some code?

Instead of a for loop:
zero_indices = [index for index in df.index if df.loc[index, col] == 0]

Look for other opportunities to replace loops with list comprehensions 
(e.g., when creating lists or filtering data)


4. pandasql to manipulate data with SQL-like queries?

from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

query = """
SELECT * FROM df_weather WHERE Temperature > 20
"""
result = pysqldf(query)
print(result)


