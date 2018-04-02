from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

intxt = ""
while True and intxt.lower() != "quit":
    intxt = input("input: ")
    print(str(analyzer.polarity_scores(intxt)))
