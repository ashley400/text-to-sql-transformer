# Correct examples

## Correct 1
- Question: How many schools did player number 3 play at?
- Gold SQL: `SELECT COUNT(School/Club Team) FROM table WHERE No. = 3`
- Predicted SQL: `SELECT COUNT(School/Club Team) FROM table WHERE No. = 3`

## Correct 2
- Question: What school did player number 21 play for?
- Gold SQL: `SELECT School/Club Team FROM table WHERE No. = 21`
- Predicted SQL: `SELECT School/Club Team FROM table WHERE No. = 21`

## Correct 3
- Question: Who is the player that wears number 42?
- Gold SQL: `SELECT Player FROM table WHERE No. = 42`
- Predicted SQL: `SELECT Player FROM table WHERE No. = 42`

## Correct 4
- Question: What player played guard for toronto in 1996-97?
- Gold SQL: `SELECT Player FROM table WHERE Position = 'guard' AND Years in Toronto = '1996-97'`
- Predicted SQL: `SELECT Player FROM table WHERE Position = 'guard' AND Years in Toronto = '1996-97'`

## Correct 5
- Question: What school/club team is Amir Johnson on?
- Gold SQL: `SELECT School/Club Team FROM table WHERE Player = 'amir johnson'`
- Predicted SQL: `SELECT School/Club Team FROM table WHERE Player = 'amir johnson'`

# Wrong examples

## Wrong 1
- Question: What position does the player who played for butler cc (ks) play?
- Gold SQL: `SELECT Position FROM table WHERE School/Club Team = 'butler cc (ks)'`
- Predicted SQL: `SELECT Position FROM table WHERE Player = 'butler cc (ks)'`
- Failure: wrong condition column

## Wrong 2
- Question: Who are all of the players on the Westchester High School club team?
- Gold SQL: `SELECT Player FROM table WHERE School/Club Team = 'westchester high school'`
- Predicted SQL: `SELECT Player FROM table WHERE School/Club Team = 'westchester'`
- Failure: wrong value

## Wrong 3
- Question: What are the total amount of numbers on the Toronto team in 2005-06?
- Gold SQL: `SELECT COUNT(No.) FROM table WHERE Years in Toronto = '2005-06'`
- Predicted SQL: `SELECT COUNT(Years in Toronto) FROM table WHERE No. = '2005-06'`
- Failure: wrong column (SELECT)

## Wrong 4
- Question: When did Jacques Chirac stop being a G8 leader?
- Gold SQL: `SELECT Ended time as senior G8 leader FROM table WHERE Person = 'jacques chirac'`
- Predicted SQL: `SELECT Office FROM table WHERE Ended time as senior G8 leader = 'jacques chirac stop'`
- Failure: wrong column (SELECT)

## Wrong 5
- Question: When did the Prime Minister of Italy take office?
- Gold SQL: `SELECT Entered office as Head of State or Government FROM table WHERE Office = 'prime minister of italy'`
- Predicted SQL: `SELECT Began time as senior G8 leader FROM table WHERE Office = 'italy'`
- Failure: wrong column (SELECT)

